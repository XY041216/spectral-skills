#!/usr/bin/env python3
"""Install or update the complete Spectral Skills bundle for Codex."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SKILLS = REPO_ROOT / "skills"
SOURCE_RUNTIME = REPO_ROOT / "spectral_core"
SKILL_NAMES = (
    "spectral-reader",
    "spectral-qc",
    "spectral-splitter",
    "spectral-preprocess",
    "spectral-feature",
    "spectral-modeling",
    "spectral-optimizer",
    "spectral-report",
    "spectral-workflow",
)
SOURCE_SHARED_NAME = "_shared"
INSTALLED_SHARED_NAME = "_spectral_shared"
INSTALLED_RUNTIME_NAME = "spectral_core"
MANAGED_NAMES = (*SKILL_NAMES, INSTALLED_SHARED_NAME, INSTALLED_RUNTIME_NAME)
IGNORED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.pyc",
    "*.pyo",
}
SHARED_REFERENCE = re.compile(r"\.\./_shared/")


def _default_destination() -> Path:
    override = os.environ.get("CODEX_SKILLS_DIR")
    return Path(override).expanduser() if override else Path.home() / ".codex" / "skills"


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in IGNORED_NAMES or any(Path(name).match(pattern) for pattern in ("*.pyc", "*.pyo")):
            ignored.add(name)
    return ignored


def _validate_sources() -> list[str]:
    errors: list[str] = []
    for skill_name in SKILL_NAMES:
        skill_dir = SOURCE_SKILLS / skill_name
        if not (skill_dir / "SKILL.md").is_file():
            errors.append(f"missing skill entry: {skill_dir / 'SKILL.md'}")
    if not (SOURCE_SKILLS / SOURCE_SHARED_NAME).is_dir():
        errors.append(f"missing shared resources: {SOURCE_SKILLS / SOURCE_SHARED_NAME}")
    if not (SOURCE_RUNTIME / "__init__.py").is_file():
        errors.append(f"missing shared runtime: {SOURCE_RUNTIME / '__init__.py'}")
    if (SOURCE_SKILLS / "spectral-core" / "SKILL.md").exists():
        errors.append("spectral-core must not be exposed as a user-facing skill")
    return errors


def _rewrite_installed_manifest(skill_dir: Path) -> None:
    manifest = skill_dir / "manifest.yaml"
    if not manifest.is_file():
        return
    original = manifest.read_text(encoding="utf-8")
    rewritten = SHARED_REFERENCE.sub(f"../{INSTALLED_SHARED_NAME}/", original)
    if rewritten != original:
        manifest.write_text(rewritten, encoding="utf-8", newline="\n")


def _stage_bundle(staging_root: Path) -> None:
    for skill_name in SKILL_NAMES:
        destination = staging_root / skill_name
        shutil.copytree(SOURCE_SKILLS / skill_name, destination, ignore=_copy_ignore)
        _rewrite_installed_manifest(destination)
    shutil.copytree(
        SOURCE_SKILLS / SOURCE_SHARED_NAME,
        staging_root / INSTALLED_SHARED_NAME,
        ignore=_copy_ignore,
    )
    shutil.copytree(SOURCE_RUNTIME, staging_root / INSTALLED_RUNTIME_NAME, ignore=_copy_ignore)


def verify_install(destination: Path, *, run_smoke: bool = True) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []

    for skill_name in SKILL_NAMES:
        skill_dir = destination / skill_name
        entry = skill_dir / "SKILL.md"
        if entry.is_file():
            checked.append(str(entry))
        else:
            errors.append(f"missing installed skill entry: {entry}")

        manifest = skill_dir / "manifest.yaml"
        if manifest.is_file():
            text = manifest.read_text(encoding="utf-8")
            if "../_shared/" in text:
                errors.append(f"unrewritten shared reference: {manifest}")
            for relative in re.findall(r"\.\./_spectral_shared/[^\s]+", text):
                candidate = (skill_dir / relative.rstrip("'\"",)).resolve()
                if not candidate.exists():
                    errors.append(f"missing installed shared reference: {candidate}")

    shared_dir = destination / INSTALLED_SHARED_NAME
    runtime_dir = destination / INSTALLED_RUNTIME_NAME
    if not shared_dir.is_dir():
        errors.append(f"missing installed shared resources: {shared_dir}")
    elif (shared_dir / "SKILL.md").exists():
        errors.append(f"shared resources must not be discoverable as a skill: {shared_dir}")
    else:
        checked.append(str(shared_dir))

    if not (runtime_dir / "__init__.py").is_file():
        errors.append(f"missing installed runtime: {runtime_dir}")
    elif (runtime_dir / "SKILL.md").exists():
        errors.append(f"shared runtime must not be discoverable as a skill: {runtime_dir}")
    else:
        checked.append(str(runtime_dir))

    obsolete_core = destination / "spectral-core"
    if (obsolete_core / "SKILL.md").exists():
        warnings.append(
            f"obsolete spectral-core skill found: {obsolete_core}; remove it after confirming this bundle works"
        )

    smoke: dict[str, Any] | None = None
    if run_smoke and not errors:
        command = [
            sys.executable,
            str(destination / "spectral-reader" / "scripts" / "server_health.py"),
            "--json",
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            cwd=destination,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        try:
            smoke = json.loads(completed.stdout)
        except json.JSONDecodeError:
            smoke = {"stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode != 0 or not isinstance(smoke, dict) or not smoke.get("ok"):
            errors.append(f"installed runtime smoke test failed: {smoke}")
        else:
            checked.append("spectral-reader server_health smoke test")

    return {
        "ok": not errors,
        "destination": str(destination),
        "skills": list(SKILL_NAMES),
        "shared": INSTALLED_SHARED_NAME,
        "runtime": INSTALLED_RUNTIME_NAME,
        "checked": checked,
        "errors": errors,
        "warnings": warnings,
        "smoke": smoke,
    }


def _safe_destination(path: Path) -> Path:
    destination = path.expanduser().resolve()
    anchor = Path(destination.anchor).resolve()
    if destination == anchor:
        raise ValueError(f"refusing to install into filesystem root: {destination}")
    if destination == REPO_ROOT or REPO_ROOT in destination.parents:
        raise ValueError(f"refusing to install inside the source repository: {destination}")
    return destination


def _replace_bundle(staging_root: Path, destination: Path) -> None:
    backup_root = destination / f".spectral-skills-backup-{uuid.uuid4().hex}"
    backup_root.mkdir()
    installed: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        for name in MANAGED_NAMES:
            target = destination / name
            staged = staging_root / name
            backup = backup_root / name
            if target.exists():
                target.replace(backup)
                backed_up.append((backup, target))
            staged.replace(target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        for backup, target in reversed(backed_up):
            if backup.exists():
                backup.replace(target)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def install_bundle(destination: Path) -> dict[str, Any]:
    source_errors = _validate_sources()
    if source_errors:
        return {"ok": False, "destination": str(destination), "errors": source_errors}

    destination = _safe_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".spectral-skills-stage-", dir=destination))
    try:
        _stage_bundle(staging_root)
        staged_check = verify_install(staging_root, run_smoke=True)
        if not staged_check["ok"]:
            return staged_check
        _replace_bundle(staging_root, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return verify_install(destination, run_smoke=True)


def _pull_latest() -> None:
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "pull", "--ff-only"],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install or update all Spectral Skills plus their shared runtime for Codex."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_default_destination(),
        help="Codex skills directory (default: CODEX_SKILLS_DIR or ~/.codex/skills).",
    )
    parser.add_argument("--pull", action="store_true", help="Run git pull --ff-only before installing.")
    parser.add_argument("--check", action="store_true", help="Verify an existing installation without changing it.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result.")
    args = parser.parse_args(argv)

    try:
        if args.pull and not args.check:
            _pull_latest()
        result = verify_install(args.destination) if args.check else install_bundle(args.destination)
    except Exception as exc:
        result = {"ok": False, "destination": str(args.destination), "errors": [str(exc)]}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["ok"]:
        print(f"Installed {len(SKILL_NAMES)} Spectral Skills into {result['destination']}")
        print(f"Shared resources: {result['shared']}; runtime: {result['runtime']}")
        for warning in result.get("warnings", []):
            print(f"Warning: {warning}")
        print("Restart Codex to load the updated skills.")
    else:
        print("Spectral Skills installation failed:", file=sys.stderr)
        for error in result.get("errors", []):
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

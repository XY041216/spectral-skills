"""Reader IO helpers for JSON and path handling."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def safe_read_text(path: str | Path, *, encoding: str = "utf-8-sig") -> str:
    """Read text with UTF-8 and UTF-8 BOM support."""

    return Path(path).read_text(encoding=encoding)


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from UTF-8 or UTF-8-with-BOM text."""

    target = Path(path)
    try:
        payload = json.loads(safe_read_text(target))
    except Exception as exc:
        raise ValueError(normalize_json_error(exc, target)) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {target}")
    return payload


def write_json_file(path: str | Path, data: Any, *, ensure_ascii: bool = False, no_bom: bool = True, indent: int = 2) -> Path:
    """Write JSON as UTF-8 without BOM by default.

    The write is atomic within the target directory. This prevents partially
    written JSON or two concatenated JSON documents when workflow tools update
    the same contract-like file repeatedly.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8" if no_bom else "utf-8-sig"
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return target


def normalize_json_error(error: Exception, path: str | Path) -> str:
    return f"Could not parse JSON file {Path(path)}: {error}"


def resolve_path(
    path_value: str | Path | None,
    *,
    base_dir: str | Path | None = None,
    read_plan_dir: str | Path | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve absolute or relative paths without encoding conversion."""

    original = "" if path_value is None else str(path_value)
    current_dir = Path(cwd) if cwd is not None else Path.cwd()
    attempted: list[dict[str, str]] = []
    if not original:
        return {
            "original_path": original,
            "resolved_path": None,
            "base_dir_used": None,
            "exists": False,
            "attempted_paths": [],
            "status": "missing",
        }

    candidate = Path(original)
    if candidate.is_absolute():
        resolved = candidate
        attempted.append({"base": "absolute", "path": str(resolved)})
        return _path_report(original, resolved, "absolute", attempted)

    for label, base in [
        ("source_base_dir", base_dir),
        ("read_plan_dir", read_plan_dir),
        ("cwd", current_dir),
    ]:
        if base is None:
            continue
        resolved = Path(base) / candidate
        attempted.append({"base": label, "path": str(resolved)})
        if resolved.exists():
            return _path_report(original, resolved, label, attempted)

    if attempted:
        last = Path(attempted[-1]["path"])
        return _path_report(original, last, attempted[-1]["base"], attempted)
    resolved = current_dir / candidate
    attempted.append({"base": "cwd", "path": str(resolved)})
    return _path_report(original, resolved, "cwd", attempted)


def resolve_read_plan_source_path(
    read_plan: dict[str, Any],
    *,
    read_plan_path: str | Path | None = None,
    invocation_cwd: str | Path | None = None,
) -> dict[str, Any]:
    read_plan_dir = Path(read_plan_path).parent if read_plan_path is not None else None
    return resolve_path(
        read_plan.get("source_path"),
        base_dir=read_plan.get("source_base_dir"),
        read_plan_dir=read_plan_dir,
        cwd=invocation_cwd,
    )


def path_to_posix_string(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


def path_exists_report(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    target = Path(path)
    return {"path": str(target), "exists": target.exists(), "is_file": target.is_file(), "is_dir": target.is_dir()}


def _path_report(original: str, resolved: Path, base: str, attempted: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "original_path": original,
        "resolved_path": str(resolved),
        "base_dir_used": base,
        "exists": resolved.exists(),
        "attempted_paths": attempted,
        "status": "resolved" if resolved.exists() else "not_found",
    }

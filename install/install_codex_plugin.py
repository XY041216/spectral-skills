"""Install the Spectral Skills Codex plugin into a local Codex home.

This helper is intentionally conservative: it validates config.toml before and
after editing, writes a backup, and materializes the plugin image into Codex's
plugin cache so Desktop can load it even when the CLI install path is blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


PLUGIN_NAME = "spectral-skills"
DEFAULT_MARKETPLACE_NAME = "spectral-skills-local-marketplace"


def install_codex_plugin(
    *,
    repo_root: Path | None = None,
    codex_home: Path | None = None,
    skip_config: bool = False,
    skip_cache: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    home = (codex_home or _default_codex_home()).expanduser().resolve()
    marketplace_path = root / ".agents" / "plugins" / "marketplace.json"
    plugin_dir = root / "plugins" / PLUGIN_NAME
    manifest_path = plugin_dir / ".codex-plugin" / "plugin.json"
    result: dict[str, Any] = {
        "repo_root": str(root),
        "codex_home": str(home),
        "marketplace_path": str(marketplace_path),
        "plugin_dir": str(plugin_dir),
        "dry_run": dry_run,
        "config_updated": False,
        "cache_updated": False,
    }

    if tomllib is None:
        return _error("TOML_PARSER_UNAVAILABLE", "Python tomllib is unavailable. Use Python 3.11 or newer.", result)
    if not marketplace_path.exists():
        return _error("MARKETPLACE_NOT_FOUND", "Repository marketplace.json is missing.", result)
    if not manifest_path.exists():
        return _error("PLUGIN_MANIFEST_NOT_FOUND", "Built plugin manifest is missing.", result)

    marketplace = _load_json(marketplace_path)
    manifest = _load_json(manifest_path)
    marketplace_name = marketplace.get("name") or DEFAULT_MARKETPLACE_NAME
    plugin_name = manifest.get("name")
    plugin_version = manifest.get("version")
    result.update({"marketplace_name": marketplace_name, "plugin_name": plugin_name, "plugin_version": plugin_version})
    if plugin_name != PLUGIN_NAME or not plugin_version:
        return _error("PLUGIN_MANIFEST_INVALID", "Plugin manifest name/version is invalid.", result)
    _validate_marketplace_entry(marketplace, plugin_name, result)

    if not skip_config:
        config_result = _upsert_codex_config(home, marketplace_name, plugin_name, root, dry_run=dry_run)
        result["config"] = config_result
        result["config_updated"] = config_result.get("updated", False)

    if not skip_cache:
        cache_result = _materialize_cache(home, marketplace_name, plugin_name, plugin_version, plugin_dir, dry_run=dry_run)
        result["cache"] = cache_result
        result["cache_updated"] = cache_result.get("updated", False)

    return _ok(result)


def _default_codex_home() -> Path:
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".codex"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_marketplace_entry(marketplace: dict[str, Any], plugin_name: str, result: dict[str, Any]) -> None:
    for entry in marketplace.get("plugins", []):
        if entry.get("name") != plugin_name:
            continue
        source = entry.get("source", {})
        result["marketplace_entry"] = entry
        if source.get("source") != "local" or source.get("path") != f"./plugins/{plugin_name}":
            raise ValueError("Marketplace entry must point to ./plugins/spectral-skills as a local source.")
        return
    raise ValueError("Marketplace entry for spectral-skills was not found.")


def _upsert_codex_config(home: Path, marketplace_name: str, plugin_name: str, repo_root: Path, *, dry_run: bool) -> dict[str, Any]:
    config_path = home / "config.toml"
    plugin_ref = f"{plugin_name}@{marketplace_name}"
    result: dict[str, Any] = {"path": str(config_path), "updated": False, "backup_path": None}
    text = config_path.read_text(encoding="utf-8-sig") if config_path.exists() else ""
    _parse_toml_or_raise(text, config_path)
    lines = text.splitlines()
    original = list(lines)
    lines = _upsert_section(
        lines,
        f"marketplaces.{marketplace_name}",
        {
            "source_type": _toml_string("local"),
            "source": _toml_string(str(repo_root)),
        },
    )
    lines = _upsert_section(
        lines,
        f'plugins."{plugin_ref}"',
        {"enabled": "true"},
    )
    new_text = "\n".join(lines).rstrip() + "\n"
    _parse_toml_or_raise(new_text, config_path)
    if original == lines:
        return result
    result["updated"] = True
    if dry_run:
        result["backup_path"] = None
        return result
    home.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        backup = config_path.with_name(f"config.toml.bak-spectral-skills-{_timestamp()}")
        shutil.copy2(config_path, backup)
        result["backup_path"] = str(backup)
    config_path.write_text(new_text, encoding="utf-8", newline="\n")
    return result


def _parse_toml_or_raise(text: str, path: Path) -> None:
    try:
        tomllib.loads(text or "")
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML: {exc}") from exc


def _upsert_section(lines: list[str], table: str, values: dict[str, str]) -> list[str]:
    header = f"[{table}]"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        output = list(lines)
        if output and output[-1].strip():
            output.append("")
        output.append(header)
        output.extend(f"{key} = {value}" for key, value in values.items())
        return output

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break

    output = list(lines)
    present: set[str] = set()
    for idx in range(start + 1, end):
        stripped = output[idx].strip()
        for key, value in values.items():
            if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
                output[idx] = f"{key} = {value}"
                present.add(key)
    missing = [f"{key} = {value}" for key, value in values.items() if key not in present]
    if missing:
        output[end:end] = missing
    return output


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _materialize_cache(home: Path, marketplace_name: str, plugin_name: str, version: str, plugin_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    cache_dir = home / "plugins" / "cache" / marketplace_name / plugin_name / version
    result = {"path": str(cache_dir), "updated": False, "skills": []}
    if dry_run:
        result["updated"] = True
        return result
    _safe_replace_tree(plugin_dir, cache_dir)
    skills_dir = cache_dir / "skills"
    skills = sorted(path.name for path in skills_dir.iterdir() if path.is_dir()) if skills_dir.exists() else []
    result.update({"updated": True, "skills": skills})
    if not (cache_dir / ".codex-plugin" / "plugin.json").exists() or "spectral-workflow" not in skills:
        raise ValueError("Plugin cache was copied but required manifest or skills are missing.")
    return result


def _safe_replace_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Plugin source does not exist: {src}")
    cache_root = dst.parents[3]
    resolved_dst = dst.resolve()
    resolved_cache_root = cache_root.resolve()
    if resolved_cache_root not in resolved_dst.parents:
        raise ValueError(f"Refusing to replace path outside plugin cache: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "_check_*", "*.pyc", "*.pyo"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "tool": "install_codex_plugin", "result": result, "warnings": [], "errors": []}


def _error(code: str, message: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": False, "tool": "install_codex_plugin", "result": result, "warnings": [], "errors": [{"code": code, "message": message}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Spectral Skills into Codex config and plugin cache.")
    parser.add_argument("--repo-root", help="Repository root containing .agents/plugins/marketplace.json and plugins/spectral-skills.")
    parser.add_argument("--codex-home", help="Codex home. Defaults to $CODEX_HOME or ~/.codex.")
    parser.add_argument("--skip-config", action="store_true", help="Do not update config.toml.")
    parser.add_argument("--skip-cache", action="store_true", help="Do not materialize the plugin cache.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report intended changes without writing.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON envelope.")
    args = parser.parse_args(argv)
    try:
        response = install_codex_plugin(
            repo_root=Path(args.repo_root) if args.repo_root else None,
            codex_home=Path(args.codex_home) if args.codex_home else None,
            skip_config=args.skip_config,
            skip_cache=args.skip_cache,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        response = _error("CODEX_PLUGIN_INSTALL_FAILED", "Could not install Spectral Skills into Codex.", {"error": str(exc)})
    if args.json:
        sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    elif response.get("ok"):
        result = response["result"]
        sys.stdout.write(f"OK: installed {result.get('plugin_name')} {result.get('plugin_version')}\n")
        if result.get("config"):
            sys.stdout.write(f"Config: {result['config']['path']}\n")
        if result.get("cache"):
            sys.stdout.write(f"Cache: {result['cache']['path']}\n")
    else:
        sys.stdout.write(f"ERROR: {response['errors'][0]['message']}\n")
        if response.get("result", {}).get("error"):
            sys.stdout.write(f"Detail: {response['result']['error']}\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

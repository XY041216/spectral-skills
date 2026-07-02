"""Preflight-check Codex config.toml before plugin import or reload."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


def check_codex_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _default_config_path()
    result: dict[str, Any] = {
        "config_path": str(path),
        "exists": path.exists(),
        "parse_ok": False,
    }
    if tomllib is None:
        return _error(
            "TOML_PARSER_UNAVAILABLE",
            "Python tomllib is unavailable. Run this check with Python 3.11 or newer.",
            result,
        )
    if not path.exists():
        result["parse_ok"] = True
        return _ok(result, warnings=[{"code": "CONFIG_NOT_FOUND", "message": "No Codex config.toml exists yet."}])
    try:
        text = path.read_text(encoding="utf-8-sig")
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        line = getattr(exc, "lineno", None)
        column = getattr(exc, "colno", None)
        result.update(
            {
                "line": line,
                "column": column,
                "error": str(exc),
                "context": _line_context(text if "text" in locals() else "", line),
                "diagnosis": _diagnose_toml_error(str(exc)),
            }
        )
        return _error("CODEX_CONFIG_TOML_INVALID", "Codex config.toml is not valid TOML.", result)
    except Exception as exc:
        result["error"] = str(exc)
        return _error("CODEX_CONFIG_READ_FAILED", "Could not read Codex config.toml.", result)

    result["parse_ok"] = True
    return _ok(result)


def _default_config_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _line_context(text: str, line: int | None) -> list[dict[str, Any]]:
    if line is None or not text:
        return []
    lines = text.splitlines()
    start = max(1, line - 2)
    end = min(len(lines), line + 2)
    return [{"line": idx, "text": lines[idx - 1]} for idx in range(start, end + 1)]


def _diagnose_toml_error(message: str) -> dict[str, str]:
    lowered = message.lower()
    if "unclosed table" in lowered or "expected ']'" in lowered:
        return {
            "likely_cause": "A malformed TOML table header, often an old [projects.'...'] path missing the closing quote or ].",
            "safe_next_step": "Fix or remove the malformed table in config.toml, then rerun this preflight before importing plugins.",
        }
    if "invalid statement" in lowered or "expected newline" in lowered:
        return {
            "likely_cause": "A malformed key/value line or corrupted text in config.toml.",
            "safe_next_step": "Inspect the reported line, preserve valid marketplace/plugin entries, and rerun this preflight.",
        }
    return {
        "likely_cause": "Codex cannot parse config.toml as TOML.",
        "safe_next_step": "Inspect the reported line and validate with this script before reopening Codex.",
    }


def _ok(result: dict[str, Any], warnings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "check_codex_config",
        "result": result,
        "warnings": warnings or [],
        "errors": [],
    }


def _error(code: str, message: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": "check_codex_config",
        "result": result,
        "warnings": [],
        "errors": [{"code": code, "message": message}],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Codex config.toml before plugin import or reload.")
    parser.add_argument("--config", help="Path to config.toml. Defaults to $CODEX_HOME/config.toml or ~/.codex/config.toml.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON envelope.")
    args = parser.parse_args(argv)
    response = check_codex_config(Path(args.config).expanduser() if args.config else None)
    if args.json:
        sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    elif response.get("ok"):
        sys.stdout.write(f"OK: {response['result']['config_path']}\n")
    else:
        result = response["result"]
        sys.stdout.write(f"ERROR: {response['errors'][0]['message']}\n")
        sys.stdout.write(f"Path: {result.get('config_path')}\n")
        if result.get("line") is not None:
            sys.stdout.write(f"Line: {result.get('line')} Column: {result.get('column')}\n")
        if result.get("error"):
            sys.stdout.write(f"Parser: {result['error']}\n")
        diagnosis = result.get("diagnosis") or {}
        if diagnosis:
            sys.stdout.write(f"Likely cause: {diagnosis.get('likely_cause')}\n")
            sys.stdout.write(f"Next step: {diagnosis.get('safe_next_step')}\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

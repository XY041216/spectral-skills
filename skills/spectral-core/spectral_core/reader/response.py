"""Shared JSON response helpers for spectral-reader tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .version import CORE_VERSION, SCHEMA_VERSION


def message(code: str, text: str, *, severity: str = "warning", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": text, "severity": severity, "details": details}


def ok_response(
    tool: str,
    result: dict[str, Any] | None = None,
    *,
    backend: str = "core",
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": tool,
        "backend": backend,
        "schema_version": SCHEMA_VERSION,
        "core_version": CORE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result or {},
        "warnings": warnings or [],
        "errors": [],
    }


def error_response(
    tool: str,
    message_text: str,
    *,
    backend: str = "core",
    code: str = "ERROR",
    result: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "backend": backend,
        "schema_version": SCHEMA_VERSION,
        "core_version": CORE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result or {},
        "warnings": warnings or [],
        "errors": [{"code": code, "message": message_text, "severity": "error", "details": details or {}}],
    }

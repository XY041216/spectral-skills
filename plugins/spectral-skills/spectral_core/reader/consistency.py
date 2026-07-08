"""Consistency checks for the first-stage spectral-reader release surface."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .response import ok_response, error_response


USER_FACING_TOOLS = [
    "server_health",
    "read_spectral_dataset",
]

REQUIRED_TOOLS = USER_FACING_TOOLS

REQUIRED_SCHEMAS = [
    "spectral_data_contract.schema.json",
    "read_spectral_dataset_result.schema.json",
]

REQUIRED_STATIC_CORE = [
    "workflow.md",
    "internal-read-settings.md",
    "confirmation-gates.md",
    "execution-boundary.md",
    "output-contract.md",
    "handoff-rules.md",
]

REQUIRED_REFERENCES = [
    "reading-semantics-patterns.md",
    "csv-layout-cases.md",
    "column-role-cases.md",
    "band-axis-cases.md",
    "external-label-cases.md",
    "sample-file-folder-cases.md",
    "excel-multi-sheet-cases.md",
    "mat-npz-variable-selection-cases.md",
    "hdf5-netcdf-dataset-path-cases.md",
    "complex-table-layout-cases.md",
    "data-contract-examples.md",
    "failure-and-recovery-cases.md",
    "confirmation-dialogue-cases.md",
    "instrument-export-cases.md",
]

REQUIRED_CORE_MODULES = [
    "preview.py",
    "read_plan.py",
    "validator.py",
    "executor.py",
    "contract.py",
    "package_writer.py",
    "workflow.py",
    "consistency.py",
    "response.py",
]


def check_consistency(*, repo_root: str | Path | None = None, backend: str = "core") -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    recommendations: list[str] = []

    manifest_path = root / "skills" / "spectral-reader" / "manifest.yaml"
    manifest = _load_yaml(manifest_path, missing, mismatches)
    _check_paths(root, checked, missing)
    _check_json_schemas(root, checked, missing, mismatches)
    _check_manifest(root, manifest, checked, missing, mismatches)
    _check_mcp_server(root, manifest, checked, mismatches)
    _check_server_health(root, checked, mismatches, warnings)
    _check_shared_skill_absent(root, checked, mismatches)
    _check_old_flow_terms(root, checked, mismatches)
    _check_primary_entrypoint(manifest, checked, mismatches)
    _check_schema_sync(root, checked, missing, mismatches)
    _check_reader_contract_extensions(root, checked, mismatches)
    _check_skill_activation_boundary(root, manifest, checked, mismatches)

    status = "failed" if missing or any(item.get("severity") == "error" for item in mismatches) else "degraded" if mismatches or warnings else "passed"
    result = {
        "consistency_status": status,
        "checked_items": checked,
        "missing_items": missing,
        "mismatches": mismatches,
        "warnings": warnings,
        "recommendations": recommendations,
        "required_tools": USER_FACING_TOOLS,
        "primary_entrypoint": "read_spectral_dataset",
    }
    if status == "failed":
        return error_response("check_consistency", "spectral-reader consistency check failed.", backend=backend, code="CONSISTENCY_FAILED", result=result, warnings=warnings)
    return ok_response("check_consistency", result, backend=backend, warnings=warnings)


def _load_yaml(path: Path, missing: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        missing.append(_item("manifest", str(path), "manifest.yaml is missing."))
        return {}
    try:
        import yaml
    except Exception as exc:
        mismatches.append(_issue("YAML_UNAVAILABLE", "yaml package is unavailable.", severity="error", error=str(exc)))
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded or {}
    except Exception as exc:
        mismatches.append(_issue("MANIFEST_YAML_INVALID", "manifest.yaml could not be parsed.", severity="error", error=str(exc)))
        return {}


def _check_paths(root: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    skill_root = root / "skills" / "spectral-reader"
    for tool in USER_FACING_TOOLS:
        _require_file(skill_root / "scripts" / f"{tool}.py", checked, missing, f"skill_script:{tool}")
        _require_file(root / "scripts" / "reader" / f"{tool}.py", checked, missing, f"fallback_script:{tool}")
    _require_file(skill_root / "SKILL.md", checked, missing, "SKILL.md")
    _require_file(skill_root / "mcp-server" / "server.py", checked, missing, "mcp_server")
    for name in REQUIRED_STATIC_CORE:
        _require_file(skill_root / "static" / "core" / name, checked, missing, f"static_core:{name}")
    for name in REQUIRED_REFERENCES:
        _require_file(skill_root / "references" / name, checked, missing, f"reference:{name}")
    for name in REQUIRED_CORE_MODULES:
        _require_file(root / "spectral_core" / "reader" / name, checked, missing, f"core_module:{name}")
    allowed_scripts = {"read_spectral_dataset.py", "server_health.py", "check_consistency.py"}
    for script_dir in [skill_root / "scripts", root / "scripts" / "reader"]:
        extras = sorted(path.name for path in script_dir.glob("*.py") if path.name not in allowed_scripts)
        if extras:
            missing.append(_item("extra_public_reader_scripts", str(script_dir), f"Public reader scripts must only expose read_spectral_dataset, server_health, and check_consistency: {extras}"))


def _check_json_schemas(root: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    shared_schema_dir = _shared_schema_dir(root)
    schema_dirs = [shared_schema_dir, root / "skills" / "spectral-reader" / "schemas", root / "spectral_core" / "schemas"]
    for schema_dir in schema_dirs:
        if not schema_dir.exists():
            missing.append(_item("schema_dir", str(schema_dir), "Schema directory is missing."))
            continue
        for path in schema_dir.glob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
                checked.append(_ok(f"schema_json:{path.relative_to(root).as_posix()}", str(path)))
            except Exception as exc:
                mismatches.append(_issue("SCHEMA_JSON_INVALID", "Schema JSON could not be parsed.", severity="error", path=str(path), error=str(exc)))


def _check_manifest(root: Path, manifest: dict[str, Any], checked: list[dict[str, Any]], missing: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    skill_root = root / "skills" / "spectral-reader"
    for key in ["name", "version", "description", "skill_type", "always_load", "schemas", "scripts_fallback", "mcp", "primary_tool", "reader_entry", "boundary"]:
        if key not in manifest:
            mismatches.append(_issue("MANIFEST_KEY_MISSING", "manifest.yaml is missing a required key.", severity="error", key=key))
        else:
            checked.append(_ok(f"manifest_key:{key}", key))
    primary = (manifest.get("primary_tool") or {}).get("name")
    if primary != "read_spectral_dataset":
        mismatches.append(_issue("PRIMARY_TOOL_MISMATCH", "primary_tool must be read_spectral_dataset.", severity="error", observed=primary))
    tools = (manifest.get("scripts_fallback") or {}).get("tools") or {}
    for tool in USER_FACING_TOOLS:
        rel = tools.get(tool)
        if not rel:
            mismatches.append(_issue("MANIFEST_SCRIPT_TOOL_MISSING", "Required script tool is missing from manifest.", severity="error", tool=tool))
        else:
            _require_file(skill_root / rel, checked, missing, f"manifest_script:{tool}")
    mcp_tools = (manifest.get("mcp") or {}).get("tools") or []
    expected_mcp = [f"reader.{tool}" for tool in USER_FACING_TOOLS]
    if sorted(mcp_tools) != sorted(expected_mcp):
        mismatches.append(_issue("MANIFEST_MCP_TOOLS_MISMATCH", "manifest MCP tools must expose only the reader entrypoint and runtime health check.", severity="error", expected=expected_mcp, observed=mcp_tools))


def _check_mcp_server(root: Path, manifest: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    server = root / "skills" / "spectral-reader" / "mcp-server" / "server.py"
    text = server.read_text(encoding="utf-8") if server.exists() else ""
    for tool in USER_FACING_TOOLS:
        if f"def {tool}" not in text:
            mismatches.append(_issue("MCP_TOOL_FUNCTION_MISSING", "MCP server missing tool function.", severity="error", tool=tool))
        elif f"core_{tool}" in text or tool == "server_health":
            checked.append(_ok(f"mcp_tool:{tool}", tool))
        else:
            checked.append(_ok(f"mcp_tool:{tool}", tool))
    helper_names = ["preview_file", "propose_read_plan", "confirm_read_plan", "validate_read_plan", "apply_read_plan", "profile_spectral_data", "build_data_contract", "validate_data_contract", "write_standardized_package", "validate_package", "write_package", "validate_contract"]
    extras = [name for name in helper_names if f"mcp.tool()({name})" in text]
    if extras:
        mismatches.append(_issue("MCP_EXTRA_TOOL_DECLARED", "MCP server must not expose helper tools as the skill interface.", severity="error", extras=extras))


def _check_server_health(root: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    script = root / "skills" / "spectral-reader" / "scripts" / "server_health.py"
    try:
        completed = subprocess.run([sys.executable, str(script), "--json"], cwd=root, text=True, capture_output=True, timeout=30)
        payload = json.loads(completed.stdout)
        if completed.returncode != 0 or not payload.get("ok"):
            mismatches.append(_issue("SERVER_HEALTH_FAILED", "server_health returned non-ok.", severity="error", returncode=completed.returncode))
        else:
            checked.append(_ok("server_health", str(script)))
        result = payload.get("result") or {}
        for tool in USER_FACING_TOOLS:
            key = f"{tool}_available"
            if tool == "server_health":
                key = "server_health_available"
            if key in result and result[key] is False:
                mismatches.append(_issue("SERVER_HEALTH_TOOL_UNAVAILABLE", "server_health reports unavailable tool.", severity="error", tool=tool))
    except Exception as exc:
        mismatches.append(_issue("SERVER_HEALTH_EXCEPTION", "server_health could not be executed.", severity="error", error=str(exc)))


def _check_shared_skill_absent(root: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    dev_path = root / "skills" / "_shared" / "SKILL.md"
    plugin_path = root / "shared" / "SKILL.md"
    if dev_path.exists():
        mismatches.append(_issue("SHARED_SKILL_PRESENT", "_shared/SKILL.md must not exist as a user-facing skill.", severity="error", path=str(dev_path)))
    elif plugin_path.exists():
        mismatches.append(_issue("SHARED_SKILL_PRESENT", "shared/SKILL.md must not exist as a user-facing skill.", severity="error", path=str(plugin_path)))
    else:
        checked.append(_ok("shared_skill_absent", f"{dev_path};{plugin_path}"))


def _check_old_flow_terms(root: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    patterns = ["sniff_file", "parse_matrix", "infer_layout"]
    roots = [root / "spectral_core", root / "skills" / "spectral-reader", root / "scripts" / "reader"]
    hits = []
    for base in roots:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".json"}:
                if path.name in {"consistency.py", "check_consistency.py"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in patterns:
                    if re.search(pattern, text):
                        hits.append({"path": str(path), "pattern": pattern})
    if hits:
        mismatches.append(_issue("OLD_FLOW_TERM_FOUND", "Old parser/inference flow terms were found.", severity="error", hits=hits))
    else:
        checked.append(_ok("old_flow_terms_absent", ",".join(patterns)))


def _check_primary_entrypoint(manifest: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    reader_entry = manifest.get("reader_entry") or manifest.get("workflow") or {}
    if reader_entry.get("primary_entrypoint") != "read_spectral_dataset":
        mismatches.append(_issue("PRIMARY_ENTRYPOINT_MISMATCH", "reader_entry.primary_entrypoint must be read_spectral_dataset.", severity="error", observed=reader_entry.get("primary_entrypoint")))
    else:
        checked.append(_ok("primary_entrypoint", "read_spectral_dataset"))


def _check_schema_sync(root: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    reader_dir = root / "skills" / "spectral-reader" / "schemas"
    core_dir = root / "spectral_core" / "schemas"
    for name in REQUIRED_SCHEMAS:
        reader_path = reader_dir / name
        core_path = core_dir / name
        _require_file(reader_path, checked, missing, f"reader_schema:{name}")
        _require_file(core_path, checked, missing, f"core_schema:{name}")
        if reader_path.exists() and core_path.exists():
            if json.loads(reader_path.read_text(encoding="utf-8")) != json.loads(core_path.read_text(encoding="utf-8")):
                mismatches.append(_issue("SCHEMA_SYNC_MISMATCH", "Reader schema and core schema differ.", severity="error", schema=name))


def _check_reader_contract_extensions(root: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    contract_schema = root / "skills" / "spectral-reader" / "schemas" / "spectral_data_contract.schema.json"
    if contract_schema.exists():
        text = contract_schema.read_text(encoding="utf-8")
        if "confidence_scores" in text:
            mismatches.append(_issue("CONTRACT_CONFIDENCE_EXPOSED", "Data Contract schema must not expose confidence_scores.", severity="error"))
        else:
            checked.append(_ok("data_contract_no_confidence_scores", str(contract_schema)))
        for forbidden in ["read_plan", "preview_report", "validation_report", "profile_summary", "package_manifest"]:
            if forbidden in text:
                mismatches.append(_issue("CONTRACT_INTERNAL_FIELD_EXPOSED", "Data Contract schema must stay minimal and not expose internal reader artifacts.", severity="error", field=forbidden))
    knowledge_files = [
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "excel-multi-sheet.md", "excel_multi_sheet_fragment"),
        (root / "skills" / "spectral-reader" / "references" / "excel-multi-sheet-cases.md", "excel_multi_sheet_reference"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "mat-npz-variable-selection.md", "mat_npz_variable_selection_fragment"),
        (root / "skills" / "spectral-reader" / "references" / "mat-npz-variable-selection-cases.md", "mat_npz_variable_selection_reference"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "hdf5-netcdf-dataset-path.md", "hdf5_netcdf_dataset_path_fragment"),
        (root / "skills" / "spectral-reader" / "references" / "hdf5-netcdf-dataset-path-cases.md", "hdf5_netcdf_dataset_path_reference"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "complex-table-layout.md", "complex_table_layout_fragment"),
        (root / "skills" / "spectral-reader" / "references" / "complex-table-layout-cases.md", "complex_table_layout_reference"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "numeric-band-columns.md", "numeric_band_columns_fragment"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "folder-name-as-label.md", "folder_name_as_label_fragment"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "excel-layout-cases.md", "excel_layout_cases_fragment"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "missing-values.md", "missing_values_fragment"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "sample-id-rules.md", "sample_id_rules_fragment"),
        (root / "skills" / "spectral-reader" / "static" / "fragments" / "reader-check-boundary.md", "reader_check_boundary_fragment"),
        (root / "skills" / "spectral-reader" / "references" / "missing-value-cases.md", "missing_value_cases_reference"),
        (root / "skills" / "spectral-reader" / "references" / "sample-id-cases.md", "sample_id_cases_reference"),
    ]
    for path, key in knowledge_files:
        if path.exists():
            checked.append(_ok(key, str(path)))
        else:
            mismatches.append(_issue("READER_KNOWLEDGE_MISSING", "Reader knowledge file is missing.", severity="error", path=str(path)))


def _check_skill_activation_boundary(root: Path, manifest: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    skill_root = root / "skills" / "spectral-reader"
    skill_md = skill_root / "SKILL.md"
    if skill_md.exists():
        text = skill_md.read_text(encoding="utf-8", errors="ignore").lower()
        required = [
            "only for reading spectral datasets",
            "do not use this skill to develop",
            "modify",
            "test",
            "package",
            "spectral-reader skill itself",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        if missing:
            mismatches.append(_issue("SKILL_BOUNDARY_MISSING", "SKILL.md must exclude developing/modifying/testing/packaging spectral-reader itself.", severity="error", missing=missing))
        else:
            checked.append(_ok("skill_activation_boundary", str(skill_md)))

    manifest_text = json.dumps(manifest, ensure_ascii=False).lower()
    for phrase in ["develop spectral-reader", "modify spectral-reader", "build spectral-reader", "package spectral-reader", "skill development", "build skill", "create reader skill"]:
        if phrase in manifest_text:
            mismatches.append(_issue("MANIFEST_DEV_TRIGGER_PRESENT", "manifest.yaml must not use development tasks as positive spectral-reader triggers.", severity="error", phrase=phrase))
    boundary = manifest.get("activation_boundary") or manifest.get("usage_boundary") or manifest.get("boundary") or {}
    if not boundary:
        mismatches.append(_issue("MANIFEST_BOUNDARY_MISSING", "manifest.yaml must describe the reader activation boundary.", severity="error"))
    else:
        checked.append(_ok("manifest_activation_boundary", "activation_boundary"))

    public_roots = [skill_root / "static", skill_root / "references"]
    heading_pattern = re.compile(r"^##\s*Step\s+\d+", re.MULTILINE)
    forbidden_patterns = [
        ("STEP_HEADING", heading_pattern),
        ("READ_PLAN_MAPPING", re.compile(r"Read Plan Mapping", re.IGNORECASE)),
        ("DEVELOPMENT_TRIGGER", re.compile(r"develop spectral-reader|modify spectral-reader|build spectral-reader|package spectral-reader|skill development", re.IGNORECASE)),
        ("PUBLIC_HELPER_WORKFLOW", re.compile(r"\b(preview_file|propose_read_plan|confirm_read_plan|validate_read_plan|apply_read_plan|profile_spectral_data|build_data_contract|write_standardized_package)\b", re.IGNORECASE)),
    ]
    hits: list[dict[str, str]] = []
    for base in public_roots:
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for code, pattern in forbidden_patterns:
                if pattern.search(text):
                    hits.append({"path": str(path), "pattern": code})
    if hits:
        mismatches.append(_issue("PUBLIC_READER_KNOWLEDGE_DEV_FLOW_FOUND", "Public reader knowledge files must not expose development-flow headings or triggers.", severity="error", hits=hits))
    else:
        checked.append(_ok("public_reader_knowledge_boundary", "static;references"))

    band_cases = skill_root / "references" / "band-axis-cases.md"
    if band_cases.exists() and "Step 5 Read Plan Mapping" in band_cases.read_text(encoding="utf-8", errors="ignore"):
        mismatches.append(_issue("BAND_AXIS_STEP_MAPPING_PRESENT", "band-axis-cases.md must not contain Step 5 Read Plan Mapping.", severity="error", path=str(band_cases)))
    elif band_cases.exists():
        checked.append(_ok("band_axis_no_step_mapping", str(band_cases)))


def _shared_schema_dir(root: Path) -> Path:
    dev_dir = root / "skills" / "_shared" / "schemas"
    plugin_dir = root / "shared" / "schemas"
    if dev_dir.exists():
        return dev_dir
    return plugin_dir


def _require_file(path: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], key: str) -> None:
    if path.exists():
        checked.append(_ok(key, str(path)))
    else:
        missing.append(_item(key, str(path), "Required file is missing."))


def _ok(key: str, path: str) -> dict[str, Any]:
    return {"key": key, "status": "ok", "path": path}


def _item(key: str, path: str, message: str) -> dict[str, Any]:
    return {"key": key, "path": path, "message": message}


def _issue(code: str, message: str, *, severity: str = "warning", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "severity": severity, "details": details}

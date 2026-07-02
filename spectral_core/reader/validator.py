"""Read-plan and future contract validation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import load_json_file, resolve_read_plan_source_path, write_json_file
from .read_plan import load_json_document, unwrap_response
from .response import error_response, ok_response


TOP_LEVEL_REQUIRED = [
    "read_plan_id",
    "read_plan_version",
    "read_plan_status",
    "source_path",
    "input_kind",
    "file_type",
    "read_mode",
    "preview_evidence",
    "decision_evidence",
    "required_confirmations",
    "execution_intent",
]


def validate_read_plan(
    read_plan: dict[str, Any] | str | Path | None = None,
    *,
    preview_report: dict[str, Any] | str | Path | None = None,
    strict: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    """Validate read_plan structure, required fields, confirmations, and status."""

    if read_plan is None:
        return error_response("validate_read_plan", "No read_plan was provided.", backend=backend, code="READ_PLAN_MISSING")
    read_plan_path = Path(read_plan) if isinstance(read_plan, (str, Path)) else None
    try:
        plan = unwrap_response(load_json_document(read_plan))
    except Exception as exc:
        return error_response("validate_read_plan", f"Could not load read_plan: {exc}", backend=backend, code="READ_PLAN_LOAD_FAILED")

    valid_schema, schema_errors = _validate_schema(plan)
    missing = collect_missing_required_fields(plan)
    required_confirmations = collect_confirmation_requirements(plan)
    recommended = list(plan.get("recommended_confirmations") or [])
    blocking_errors = collect_blocking_errors(plan, missing)
    warnings = []
    warnings.extend(collect_created_by_warnings(plan))
    path_resolution = resolve_read_plan_source_path(plan, read_plan_path=read_plan_path)
    if plan.get("read_plan_status") == "confirmed" and not path_resolution.get("exists"):
        blocking_errors.append(_issue("SOURCE_PATH_NOT_FOUND", "confirmed read_plan source_path could not be resolved.", **path_resolution))
    elif not path_resolution.get("exists"):
        warnings.append(_issue("SOURCE_PATH_NOT_FOUND_WARNING", "source_path could not be resolved yet; provisional plans may still need repair.", **path_resolution))
    if strict and recommended:
        warnings.append({"code": "RECOMMENDED_CONFIRMATIONS_PRESENT", "message": "Recommended confirmations are present in strict mode.", "details": {"count": len(recommended)}})

    status = summarize_read_plan_readiness(plan, missing, required_confirmations, blocking_errors, valid_schema)
    downstream = {
        "apply_read_plan": status == "confirmed",
        "requires_user_confirmation": bool(required_confirmations),
        "reason": "confirmed read_plan may proceed to apply_read_plan" if status == "confirmed" else "read_plan requires repair or confirmation before apply_read_plan",
    }
    result = {
        "read_plan_status": status,
        "declared_read_plan_status": plan.get("read_plan_status"),
        "valid_schema": valid_schema,
        "schema_errors": schema_errors,
        "missing_required_fields": missing,
        "required_confirmations": required_confirmations,
        "recommended_confirmations": recommended,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "path_resolution": path_resolution,
        "downstream_readiness_hint": downstream,
    }
    if schema_errors:
        return error_response(
            "validate_read_plan",
            "read_plan schema validation failed.",
            backend=backend,
            code="READ_PLAN_SCHEMA_INVALID",
            result=result,
            details={"schema_errors": schema_errors},
        )
    return ok_response("validate_read_plan", result, backend=backend, warnings=[{"code": item["code"], "message": item["message"], "severity": "warning", "details": item.get("details", {})} for item in warnings])


def validate_read_plan_for_execution(
    read_plan: dict[str, Any] | str | Path | None = None,
    *,
    backend: str = "core",
) -> dict[str, Any]:
    """Validate the stricter apply_read_plan execution gate."""

    validation = validate_read_plan(read_plan, backend=backend)
    plan = unwrap_response(load_json_document(read_plan)) if read_plan is not None else {}
    status = (validation.get("result") or {}).get("read_plan_status")
    blocking_errors = list((validation.get("result") or {}).get("blocking_errors") or [])
    missing = list((validation.get("result") or {}).get("missing_required_fields") or [])
    required_confirmations = list((validation.get("result") or {}).get("required_confirmations") or [])

    execution_errors = []
    execution_errors.extend(validate_confirmed_status(plan, status, required_confirmations))
    execution_errors.extend(validate_supported_read_mode(plan))
    execution_errors.extend(validate_spectral_column_selection(plan))

    result = {
        "read_plan_status": status,
        "valid_schema": (validation.get("result") or {}).get("valid_schema", False),
        "schema_errors": (validation.get("result") or {}).get("schema_errors", []),
        "missing_required_fields": missing,
        "required_confirmations": required_confirmations,
        "blocking_errors": blocking_errors + execution_errors,
        "path_resolution": (validation.get("result") or {}).get("path_resolution", {}),
        "supported_apply_file_types": ["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "ods", "npy", "npz", "mat", "h5", "hdf5", "nc"],
        "supported_apply_modes": [
            "matrix_file.samples_as_rows",
            "matrix_file.samples_as_columns",
            "matrix_file.external_label_file",
            "sample_files_folder.one_file_per_sample",
        ],
        "unsupported_apply_modes": ["multi_sheet", "multi_variable", "json_table"],
    }
    if not validation.get("ok") or result["blocking_errors"] or missing:
        return error_response(
            "validate_read_plan_for_execution",
            "read_plan cannot be executed by apply_read_plan.",
            backend=backend,
            code="READ_PLAN_EXECUTION_BLOCKED",
            result=result,
        )
    return ok_response("validate_read_plan_for_execution", result, backend=backend)


def validate_confirmed_status(
    plan: dict[str, Any],
    computed_status: str | None,
    required_confirmations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    if plan.get("read_plan_status") != "confirmed":
        errors.append(_issue("READ_PLAN_NOT_CONFIRMED", "apply_read_plan requires declared read_plan_status=confirmed."))
    if computed_status != "confirmed":
        errors.append(_issue("READ_PLAN_VALIDATION_NOT_CONFIRMED", "validate_read_plan did not compute confirmed status.", computed_status=computed_status))
    if required_confirmations:
        errors.append(_issue("UNRESOLVED_CONFIRMATIONS", "apply_read_plan cannot run with unresolved required confirmations.", count=len(required_confirmations)))
    return errors


def validate_supported_read_mode(plan: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    suffix = str(plan.get("file_type") or Path(str(plan.get("source_path") or "")).suffix).lower().lstrip(".")
    if plan.get("read_mode") == "sample_files_folder":
        if plan.get("input_kind") != "folder":
            errors.append(_issue("FOLDER_INPUT_REQUIRED", "sample_files_folder read_plan requires input_kind=folder."))
        if suffix not in {"folder", ""}:
            errors.append(_issue("APPLY_FILE_TYPE_UNSUPPORTED", "sample_files_folder requires file_type=folder.", file_type=suffix))
        if plan.get("sample_orientation") != "one_file_per_sample":
            errors.append(_issue("APPLY_ORIENTATION_UNSUPPORTED", "sample_files_folder requires sample_orientation=one_file_per_sample.", sample_orientation=plan.get("sample_orientation")))
        if not (plan.get("sample_file_value_column") or (plan.get("sample_file_columns") or {}).get("value")):
            errors.append(_issue("SAMPLE_FILE_VALUE_COLUMN_MISSING", "sample_files_folder requires sample_file_value_column or sample_file_columns.value."))
        if not (plan.get("sample_file_band_axis_column") or (plan.get("sample_file_columns") or {}).get("band_axis")):
            errors.append(_issue("SAMPLE_FILE_BAND_AXIS_COLUMN_MISSING", "sample_files_folder requires sample_file_band_axis_column or sample_file_columns.band_axis."))
        if plan.get("task_hint") == "classification" and not (
            plan.get("folder_name_as_label") is True
            or plan.get("file_name_as_label") is True
            or _role_source(plan.get("label")) in {"folder_name", "file_name", "external_file"}
            or (plan.get("label_file") or {}).get("path")
        ):
            errors.append(_issue("LABEL_SOURCE_MISSING", "classification sample_files_folder read_plan requires folder_name_as_label, file_name_as_label, label.source, or label_file."))
        return errors
    if suffix not in {"csv", "tsv", "txt", "xlsx", "xls", "xlsm", "ods", "npy", "npz", "mat", "h5", "hdf5", "nc"}:
        errors.append(_issue("APPLY_FILE_TYPE_UNSUPPORTED", "apply_read_plan supports CSV, TSV, TXT, Excel, ODS, NPY, NPZ, MAT, HDF5, and NetCDF files.", file_type=suffix))
    if plan.get("read_mode") != "matrix_file":
        errors.append(_issue("APPLY_READ_MODE_UNSUPPORTED", "apply_read_plan supports read_mode=matrix_file for CSV/TSV/TXT execution.", read_mode=plan.get("read_mode")))
    if plan.get("sample_orientation") not in {"rows", "columns"}:
        errors.append(_issue("APPLY_ORIENTATION_UNSUPPORTED", "apply_read_plan supports sample_orientation rows or columns.", sample_orientation=plan.get("sample_orientation")))
    if plan.get("sample_orientation") == "columns" and (plan.get("samples_as_columns") or {}).get("enabled") is not True:
        errors.append(_issue("SAMPLES_AS_COLUMNS_NOT_DECLARED", "sample_orientation=columns requires samples_as_columns.enabled=true."))
    label_file = plan.get("label_file") or {}
    if label_file.get("path"):
        alignment = plan.get("alignment_plan") or {}
        row_order_confirmed = (alignment.get("method") == "row_order" or plan.get("label_alignment") == "row_order") and bool(plan.get("allow_row_order_labels") or plan.get("label_alignment") == "row_order")
        if not row_order_confirmed and not (alignment.get("join_key") or label_file.get("sample_id_column") or _role_column(plan.get("sample_id"))):
            errors.append(_issue("JOIN_KEY_MISSING", "External label file alignment requires alignment_plan.join_key or label_file.sample_id_column."))
        if not (label_file.get("label_column") or label_file.get("target_columns") or _role_column(plan.get("label")) or _role_column(plan.get("target"))):
            errors.append(_issue("LABEL_COLUMN_MISSING", "External label file alignment requires label_file.label_column or target_columns for supervised tasks."))
    return errors


def validate_spectral_column_selection(plan: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    if plan.get("read_mode") == "sample_files_folder":
        return errors
    suffix = str(plan.get("file_type") or Path(str(plan.get("source_path") or "")).suffix).lower().lstrip(".")
    if suffix in {"npy", "npz", "mat"}:
        variable_map = plan.get("variable_map") or {}
        container = plan.get("container") or {}
        if not (variable_map.get("X") or container.get("x_var")):
            errors.append(_issue("X_VAR_NOT_FOUND", "Container read_plan requires variable_map.X or container.x_var."))
        return errors
    if suffix in {"h5", "hdf5", "nc"}:
        dataset_map = plan.get("dataset_map") or {}
        container = plan.get("hierarchical_container") or {}
        if not (dataset_map.get("X") or container.get("x_path")):
            errors.append(_issue("X_DATASET_NOT_FOUND", "Hierarchical container read_plan requires dataset_map.X or hierarchical_container.x_path."))
        return errors
    if plan.get("sample_orientation") == "columns" and (plan.get("samples_as_columns") or {}).get("enabled") is True:
        if not (plan.get("samples_as_columns") or {}).get("band_axis_column") and (plan.get("samples_as_columns") or {}).get("band_axis_column") != 0:
            errors.append(_issue("BAND_AXIS_COLUMN_MISSING", "samples_as_columns requires samples_as_columns.band_axis_column."))
        return errors
    spectral = plan.get("spectral_columns") or {}
    if not (spectral.get("columns") or (spectral.get("start") is not None and spectral.get("end") is not None)):
        errors.append(_issue("SPECTRAL_COLUMNS_MISSING", "apply_read_plan requires explicit spectral_columns columns or start/end."))
    return errors


def validate_column_existence(required_columns: list[str], available_columns: list[str], *, role: str) -> list[dict[str, Any]]:
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        return [_issue("COLUMN_NOT_FOUND", "Declared read_plan columns are missing.", role=role, missing=missing, available=available_columns)]
    return []


def summarize_read_plan_readiness(
    plan: dict[str, Any],
    missing_required_fields: list[str],
    required_confirmations: list[dict[str, Any]],
    blocking_errors: list[dict[str, Any]],
    valid_schema: bool,
) -> str:
    if not valid_schema or missing_required_fields or blocking_errors:
        return "blocked"
    if required_confirmations or plan.get("read_plan_status") in {"draft", "provisional"}:
        return "provisional"
    return "confirmed"


def collect_confirmation_requirements(plan: dict[str, Any]) -> list[dict[str, Any]]:
    required = [item for item in plan.get("required_confirmations") or [] if isinstance(item, dict)]
    confirmed = set(plan.get("confirmed_items") or [])
    unresolved = []
    for item in required:
        identifier = item.get("id") or item.get("field") or item.get("question")
        if item.get("status") != "confirmed" and identifier not in confirmed:
            unresolved.append(item)
    for item in plan.get("unresolved_items") or []:
        if isinstance(item, str) and item not in {x.get("id") for x in unresolved}:
            unresolved.append({"id": item, "status": "unresolved", "question": f"Resolve confirmation item: {item}"})
    return unresolved


def collect_blocking_errors(plan: dict[str, Any], missing_required_fields: list[str] | None = None) -> list[dict[str, Any]]:
    errors = []
    if missing_required_fields:
        errors.append(_issue("MISSING_REQUIRED_FIELDS", "Required read_plan fields are missing.", fields=missing_required_fields))
    read_mode = plan.get("read_mode")
    if read_mode in {None, "unknown"}:
        errors.append(_issue("READ_MODE_UNKNOWN", "read_mode is unknown."))
    if read_mode == "matrix_file":
        if plan.get("sample_orientation") == "columns" and (plan.get("samples_as_columns") or {}).get("enabled") is True:
            pass
        elif not _has_spectral_columns(plan):
            errors.append(_issue("SPECTRAL_COLUMNS_MISSING", "matrix_file read_plan requires spectral_columns."))
        if plan.get("sample_orientation") in {None, "unknown"}:
            errors.append(_issue("SAMPLE_ORIENTATION_UNKNOWN", "matrix_file readPlan requires sample_orientation before execution."))
    if read_mode == "multi_sheet" and not plan.get("sheet_name"):
        errors.append(_issue("SHEET_NAME_MISSING", "multi_sheet read_plan requires sheet_name."))
    if read_mode == "multi_variable" and not (plan.get("variable_name") or plan.get("array_key")):
        errors.append(_issue("VARIABLE_NAME_MISSING", "multi_variable read_plan requires variable_name or array_key."))
    if read_mode == "sample_files_folder":
        if not plan.get("sample_file_pattern"):
            errors.append(_issue("SAMPLE_FILE_PATTERN_MISSING", "sample_files_folder read_plan requires sample_file_pattern."))
        if plan.get("file_name_as_sample_id") is not True:
            errors.append(_issue("SAMPLE_ID_POLICY_MISSING", "sample_files_folder read_plan requires sample ID policy."))
    task = plan.get("task_hint")
    if task == "classification" and not _role_column(plan.get("label")) and not _role_source(plan.get("label")) in {"folder_name", "file_name"} and plan.get("folder_name_as_label") is not True and plan.get("file_name_as_label") is not True and not (plan.get("label_file") or {}).get("path"):
        errors.append(_issue("LABEL_REQUIRED_FOR_CLASSIFICATION", "classification read_plan requires label column or label_file."))
    if task == "regression" and not _role_column(plan.get("target")) and not (plan.get("label_file") or {}).get("path"):
        errors.append(_issue("TARGET_REQUIRED_FOR_REGRESSION", "regression read_plan requires target column or label_file."))
    if (plan.get("alignment_plan") or {}).get("method") == "row_order" and not (plan.get("alignment_plan") or {}).get("join_key") and not (plan.get("allow_row_order_labels") or plan.get("label_alignment") == "row_order") and not collect_confirmation_requirements(plan):
        errors.append(_issue("ROW_ORDER_ALIGNMENT_REQUIRES_CONFIRMATION", "row_order alignment must be confirmed."))
    errors.extend(_issue("DECLARED_BLOCKED_REASON", reason) for reason in plan.get("blocked_reasons") or [])
    return errors


def collect_missing_required_fields(plan: dict[str, Any]) -> list[str]:
    return [field for field in TOP_LEVEL_REQUIRED if field not in plan or plan.get(field) is None]


def collect_created_by_warnings(plan: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    created_by = plan.get("created_by")
    if created_by and created_by not in {"agent", "propose_read_plan", "user", "test_fixture", "unknown", "tool", "script"}:
        warnings.append(_issue("CREATED_BY_DETAIL_STRING", "created_by is a descriptive string; prefer created_by_type plus created_by_detail.", created_by=created_by))
    return warnings


def collect_low_confidence_warnings(plan: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    scores = plan.get("confidence_scores") or {}
    for name, score in scores.items():
        if isinstance(score, (int, float)) and score < 0.75:
            warnings.append(_issue("LOW_CONFIDENCE_FIELD", "read_plan confidence is below 0.75.", field=name, score=score))
    return warnings


def validate_contract(contract: dict[str, Any] | None = None, *, backend: str = "core") -> dict[str, Any]:
    return validate_data_contract(contract, backend=backend)


def validate_data_contract(
    contract: dict[str, Any] | str | Path | None = None,
    *,
    output: str | Path | None = None,
    strict: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    """Validate a minimal Spectral Data Contract and its data refs."""

    if contract is None:
        return error_response("validate_data_contract", "No data contract was provided.", backend=backend, code="DATA_CONTRACT_MISSING")
    contract_base_dir = Path(contract).parent if isinstance(contract, (str, Path)) else Path(".")
    try:
        loaded = unwrap_response(load_json_document(contract))
    except Exception as exc:
        return error_response("validate_data_contract", f"Could not load data contract: {exc}", backend=backend, code="DATA_CONTRACT_LOAD_FAILED")

    schema_valid, schema_errors = validate_contract_schema(loaded)
    ref_report = validate_data_refs(loaded, base_dir=contract_base_dir)
    profile_report = validate_profile_consistency(loaded, ref_report)
    downstream_report = validate_downstream_readiness(loaded)
    blocking_errors = []
    warnings = []
    blocking_errors.extend(schema_errors)
    blocking_errors.extend(ref_report["blocking_errors"])
    blocking_errors.extend(profile_report["blocking_errors"])
    warnings.extend(profile_report["warnings"])
    warnings.extend(downstream_report["warnings"])
    recommended_status = summarize_contract_validation(loaded, blocking_errors, warnings)
    report = {
        "valid_schema": schema_valid,
        "valid_refs": not ref_report["blocking_errors"],
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "downstream_readiness": downstream_report["downstream_readiness"],
        "recommended_contract_status": recommended_status,
        "validation_summary": {
            "contract_status": loaded.get("contract_status"),
            "n_blocking_errors": len(blocking_errors),
            "n_warnings": len(warnings),
            "strict": strict,
        },
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(out, report, ensure_ascii=False)
        _write_validation_log(out.parent, report)
    if blocking_errors:
        return error_response("validate_data_contract", "Data Contract validation failed.", backend=backend, code="DATA_CONTRACT_INVALID", result=report)
    return ok_response("validate_data_contract", report, backend=backend, warnings=[_warning_to_envelope(item) for item in warnings])


def validate_contract_schema(contract: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    schema_path = Path(__file__).resolve().parents[2] / "skills" / "spectral-reader" / "schemas" / "spectral_data_contract.schema.json"
    if not schema_path.exists():
        return False, [_issue("DATA_CONTRACT_SCHEMA_NOT_FOUND", f"Schema not found: {schema_path}")]
    try:
        import jsonschema
    except Exception:
        missing = [field for field in ["status", "X", "band_axis", "n_samples", "n_features"] if field not in contract]
        return not missing, [_issue("DATA_CONTRACT_REQUIRED_FIELDS_MISSING", "Required fields are missing.", fields=missing)] if missing else []
    schema = load_json_file(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(contract), key=lambda err: list(err.path))
    return not errors, [_issue("DATA_CONTRACT_SCHEMA_ERROR", err.message, path=".".join(str(part) for part in err.path)) for err in errors]


def validate_data_refs(contract: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    data = _contract_ref_view(contract)
    base_dir = Path(str(base_dir or data.get("base_dir") or "."))
    required = ["X_ref", "band_axis_ref"]
    blocking = []
    resolved: dict[str, str] = {}
    for key in required + ["y_ref", "sample_ids_ref", "metadata_ref", "profile_ref"]:
        ref = data.get(key)
        if not ref:
            if key in required:
                blocking.append(_issue("DATA_REF_MISSING", f"{key} is missing.", ref_key=key))
            continue
        path = Path(ref)
        full = path if path.is_absolute() else base_dir / path
        resolved[key] = str(full)
        if not full.exists():
            blocking.append(_issue("DATA_REF_NOT_FOUND", f"{key} does not exist.", ref_key=key, path=str(full)))
    return {"blocking_errors": blocking, "resolved_refs": resolved}


def validate_profile_consistency(contract: dict[str, Any], ref_report: dict[str, Any]) -> dict[str, Any]:
    if ref_report["blocking_errors"]:
        return {"blocking_errors": [], "warnings": []}
    refs = {key: Path(value) for key, value in ref_report["resolved_refs"].items()}
    blocking = []
    warnings = []
    X_rows, X_cols = _csv_shape(refs["X_ref"], has_header=True)
    band_rows, _ = _csv_shape(refs["band_axis_ref"], has_header=True)
    if band_rows != X_cols:
        blocking.append(_issue("BAND_AXIS_LENGTH_MISMATCH", "band_axis row count does not match X feature count.", expected=X_cols, observed=band_rows))
    for key in ["y_ref", "sample_ids_ref", "metadata_ref"]:
        path = refs.get(key)
        if path:
            rows, _ = _csv_shape(path, has_header=True)
            if rows != X_rows:
                blocking.append(_issue("ROW_COUNT_MISMATCH", f"{key} row count does not match X.", ref_key=key, expected=X_rows, observed=rows))
    shape = contract.get("shape") or contract.get("profile") or {"n_samples": contract.get("n_samples"), "n_features": contract.get("n_features")}
    if shape.get("n_samples") is not None and shape.get("n_samples") != X_rows:
        blocking.append(_issue("PROFILE_SAMPLE_COUNT_MISMATCH", "contract n_samples does not match X.", expected=X_rows, observed=shape.get("n_samples")))
    if shape.get("n_features") is not None and shape.get("n_features") != X_cols:
        blocking.append(_issue("PROFILE_FEATURE_COUNT_MISMATCH", "contract n_features does not match X.", expected=X_cols, observed=shape.get("n_features")))
    return {"blocking_errors": blocking, "warnings": warnings}


def validate_downstream_readiness(contract: dict[str, Any]) -> dict[str, Any]:
    readiness = contract.get("downstream_readiness") or {"ready": contract.get("status") == "ready"}
    warnings = []
    if contract.get("contract_status") == "blocked" or contract.get("status") == "blocked":
        warnings.append(_issue("CONTRACT_BLOCKED", "Blocked contract cannot be handed to downstream supervised tasks."))
    return {"downstream_readiness": readiness, "warnings": warnings}


def summarize_contract_validation(contract: dict[str, Any], blocking_errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> str:
    if blocking_errors:
        return "blocked"
    if contract.get("status") == "ready":
        return "confirmed"
    if contract.get("contract_status") == "blocked":
        return "blocked"
    if contract.get("contract_status") == "provisional":
        return "provisional"
    return "confirmed"


def validate_package_structure(package_dir: str | Path | None = None, *, backend: str = "core") -> dict[str, Any]:
    """Validate the minimal standard reader output directory."""

    if package_dir is None:
        return error_response("validate_package", "No package_dir was provided.", backend=backend, code="PACKAGE_DIR_MISSING")
    root = Path(package_dir)
    if not root.exists():
        return error_response("validate_package", "package_dir does not exist.", backend=backend, code="PACKAGE_DIR_NOT_FOUND", result={"package_status": "blocked", "package_dir": str(root)})
    manifest_report = {"blocking_errors": [], "warnings": [], "manifest": None, "manifest_summary": {}}
    file_report = validate_package_files(root, None)
    contract_report = validate_package_against_contract(root)
    blocking_errors = []
    warnings = []
    blocking_errors.extend(manifest_report["blocking_errors"])
    blocking_errors.extend(file_report["blocking_errors"])
    blocking_errors.extend(contract_report["blocking_errors"])
    warnings.extend(manifest_report["warnings"])
    warnings.extend(file_report["warnings"])
    warnings.extend(contract_report["warnings"])
    status = summarize_package_validation(root, blocking_errors, warnings, manifest_report.get("manifest"))
    result = {
        "package_status": status,
        "package_dir": str(root),
        "valid_manifest": True,
        "valid_files": not file_report["blocking_errors"],
        "valid_against_contract": not contract_report["blocking_errors"],
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "manifest_summary": manifest_report.get("manifest_summary", {}),
        "file_summary": file_report.get("file_summary", {}),
        "contract_summary": contract_report.get("contract_summary", {}),
    }
    if blocking_errors:
        return error_response("validate_package", "Standardized reader package validation failed.", backend=backend, code="PACKAGE_INVALID", result=result)
    return ok_response("validate_package", result, backend=backend, warnings=[_warning_to_envelope(item) for item in warnings])


def validate_package_manifest(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir)
    manifest_path = root / "package_manifest.json"
    blocking = []
    warnings = []
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = load_json_file(manifest_path)
        except Exception as exc:
            blocking.append(_issue("PACKAGE_MANIFEST_UNREADABLE", "package_manifest.json could not be parsed.", error=str(exc)))
        warnings.append(_issue("PACKAGE_MANIFEST_DEPRECATED", "package_manifest.json is no longer part of spectral-reader standard output."))
    return {
        "manifest": manifest,
        "blocking_errors": blocking,
        "warnings": warnings,
        "manifest_summary": {
            "manifest_path": str(manifest_path),
            "package_status": (manifest or {}).get("package_status"),
            "file_count": len((manifest or {}).get("files") or []),
        },
    }


def validate_package_files(package_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(package_dir)
    blocking = []
    warnings = []
    core_groups = [["X.csv"], ["band_axis.csv"], ["data_contract.json"]]
    core_paths = ["/".join(group) for group in core_groups]
    for group in core_groups:
        if not any((root / rel).exists() for rel in group):
            blocking.append(_issue("PACKAGE_CORE_FILE_MISSING", "Core package file is missing.", path=group[0]))
    for rel in ["_internal", "package_manifest.json", "summary.json", "preview_report.json", "validation_report.json", "profile_summary.json", "logs"]:
        if (root / rel).exists():
            warnings.append(_issue("REDUNDANT_OUTPUT_PRESENT", "Redundant output should not be present in standard reader output.", path=rel))
    if manifest:
        for item in (manifest.get("files") or manifest.get("public_files") or []) + (manifest.get("internal_files") or []):
            rel = item.get("path")
            if item.get("exists") and rel and not (root / rel).exists():
                blocking.append(_issue("PACKAGE_MANIFEST_FILE_NOT_FOUND", "Manifest file entry does not exist.", path=rel, key=item.get("key")))
    return {
        "blocking_errors": blocking,
        "warnings": warnings,
        "file_summary": {"core_paths_checked": core_paths},
    }


def validate_package_against_contract(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir)
    blocking = []
    warnings = []
    contract_path = root / "data_contract.json"
    if not contract_path.exists():
        return {"blocking_errors": [], "warnings": [], "contract_summary": {"contract_present": False}}
    try:
        contract = load_json_file(contract_path)
    except Exception as exc:
        return {"blocking_errors": [_issue("PACKAGE_CONTRACT_UNREADABLE", "Packaged data_contract.json could not be parsed.", error=str(exc))], "warnings": [], "contract_summary": {"contract_present": True}}
    data = _contract_ref_view(contract)
    ref_map = {
        "X_ref": "X.csv",
        "band_axis_ref": "band_axis.csv",
        "y_ref": "y.csv",
        "sample_ids_ref": "sample_ids.csv",
        "metadata_ref": "metadata.csv",
    }
    for key, packaged_rel in ref_map.items():
        ref = data.get(key)
        if key in {"X_ref", "band_axis_ref"} and not (root / packaged_rel).exists():
            blocking.append(_issue("PACKAGE_CONTRACT_CORE_REF_MISSING", "Core contract ref is not present in package.", ref_key=key, expected_path=packaged_rel))
        elif ref and not (root / packaged_rel).exists():
            warnings.append(_issue("PACKAGE_CONTRACT_OPTIONAL_REF_MISSING", "Optional contract ref is not present in package.", ref_key=key, expected_path=packaged_rel))
    x_path = root / "X.csv"
    band_path = root / "band_axis.csv"
    if x_path.exists() and band_path.exists():
        x_rows, x_cols = _csv_shape(x_path, has_header=True)
        band_rows, _ = _csv_shape(band_path, has_header=True)
        if band_rows != x_cols:
            blocking.append(_issue("PACKAGE_BAND_AXIS_LENGTH_MISMATCH", "Packaged band_axis length does not match X feature count.", expected=x_cols, observed=band_rows))
        for rel in ["y.csv", "sample_ids.csv", "metadata.csv"]:
            path = root / rel
            if path.exists():
                rows, _ = _csv_shape(path, has_header=True)
                if rows != x_rows:
                    blocking.append(_issue("PACKAGE_ROW_COUNT_MISMATCH", "Packaged optional data rows do not match X.", path=rel, expected=x_rows, observed=rows))
    return {
        "blocking_errors": blocking,
        "warnings": warnings,
        "contract_summary": {"contract_present": True, "contract_status": contract.get("status") or contract.get("contract_status")},
    }


def summarize_package_validation(
    package_dir: str | Path,
    blocking_errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> str:
    if blocking_errors:
        return "blocked"
    return "ready"


def _csv_shape(path: Path, *, has_header: bool) -> tuple[int, int]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return 0, 0
    cols = len(rows[0])
    data_rows = rows[1:] if has_header else rows
    return len(data_rows), cols


def _warning_to_envelope(item: dict[str, Any]) -> dict[str, Any]:
    return {"code": item["code"], "message": item["message"], "severity": "warning", "details": item.get("details", {})}


def _write_validation_log(output_dir: Path, report: dict[str, Any]) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {"tool": "validate_data_contract", "recommended_contract_status": report.get("recommended_contract_status"), "blocking_errors": report.get("blocking_errors", [])}
    write_json_file(logs / "validation_log.json", payload, ensure_ascii=False)


def _validate_schema(plan: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "read_plan.schema.json"
    if not schema_path.exists():
        return False, [_issue("READ_PLAN_SCHEMA_NOT_FOUND", f"Schema not found: {schema_path}")]
    try:
        import jsonschema
    except Exception:
        missing = collect_missing_required_fields(plan)
        return not missing, [_issue("JSONSCHEMA_MISSING", "jsonschema is unavailable; used basic required-field validation.")] if missing else []
    schema = load_json_file(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(plan), key=lambda err: list(err.path))
    return not errors, [_schema_validation_issue(err) for err in errors]


def _schema_validation_issue(err: Any) -> dict[str, Any]:
    path = ".".join(str(part) for part in err.path)
    details: dict[str, Any] = {"path": path}
    if path == "band_unit":
        details.update(
            {
                "field": "band_unit",
                "received": err.instance,
                "allowed": ["nm", "cm-1", "index", "unknown"],
                "hint": "For a wavenumber axis, use --band-type wavenumber --band-unit cm-1.",
            }
        )
    return _issue("SCHEMA_VALIDATION_ERROR", err.message, **details)


def _has_spectral_columns(plan: dict[str, Any]) -> bool:
    suffix = str(plan.get("file_type") or Path(str(plan.get("source_path") or "")).suffix).lower().lstrip(".")
    if suffix in {"npy", "npz", "mat"}:
        variable_map = plan.get("variable_map") or {}
        container = plan.get("container") or {}
        return bool(variable_map.get("X") or container.get("x_var"))
    if suffix in {"h5", "hdf5", "nc"}:
        dataset_map = plan.get("dataset_map") or {}
        container = plan.get("hierarchical_container") or {}
        return bool(dataset_map.get("X") or container.get("x_path"))
    spectral = plan.get("spectral_columns") or {}
    return bool(spectral.get("columns") or spectral.get("start") is not None or spectral.get("end") is not None)


def _role_column(role: Any) -> bool:
    if not isinstance(role, dict):
        return False
    column = role.get("column")
    columns = role.get("columns")
    return column is not None or bool(columns)


def _role_source(role: Any) -> str | None:
    return str(role.get("source")) if isinstance(role, dict) and role.get("source") else None


def _contract_ref_view(contract: dict[str, Any]) -> dict[str, Any]:
    if isinstance(contract.get("files"), dict):
        files = dict(contract.get("files") or {})
        return {
            "base_dir": contract.get("base_dir"),
            "X_ref": files.get("X"),
            "y_ref": files.get("y"),
            "sample_ids_ref": files.get("sample_ids"),
            "band_axis_ref": files.get("band_axis"),
            "metadata_ref": files.get("metadata"),
        }
    if isinstance(contract.get("data"), dict):
        return dict(contract.get("data") or {})
    band_axis = contract.get("band_axis")
    if isinstance(band_axis, dict):
        band_axis_ref = band_axis.get("file")
    else:
        band_axis_ref = band_axis
    return {
        "base_dir": contract.get("base_dir"),
        "X_ref": contract.get("X"),
        "y_ref": contract.get("y"),
        "sample_ids_ref": contract.get("sample_ids"),
        "band_axis_ref": contract.get("band_axis_ref") or band_axis_ref,
        "metadata_ref": contract.get("metadata"),
    }


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}

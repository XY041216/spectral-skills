"""Minimal Spectral Data Contract builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import load_json_file
from .profiler import profile_spectral_data
from .read_plan import load_json_document, unwrap_response
from .response import error_response, ok_response


def build_spectral_data_contract(
    *,
    apply_dir: str | Path | None = None,
    read_plan: dict[str, Any] | str | Path | None = None,
    profile_summary: dict[str, Any] | str | Path | None = None,
    output: str | Path | None = None,
    status: str | None = None,
    confirmation_log: list[dict[str, Any]] | None = None,
    backend: str = "core",
) -> dict[str, Any]:
    if apply_dir is None:
        return error_response("build_data_contract", "--apply-dir is required.", backend=backend, code="APPLY_DIR_REQUIRED")
    try:
        root = Path(apply_dir)
        apply_result = load_json_file(root / "apply_read_plan_result.json")
        plan = _load_optional(read_plan)
        profile = _load_or_profile(root, profile_summary)
        contract = _build_contract(root, apply_result, plan, profile, status=status, confirmation_log=confirmation_log, backend=backend)
        if output:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            from .io_utils import write_json_file

            write_json_file(out, contract, ensure_ascii=False)
            contract["contract_ref"] = str(out)
        return ok_response("build_data_contract", contract, backend=backend)
    except Exception as exc:
        return error_response("build_data_contract", f"Could not build Data Contract: {exc}", backend=backend, code="BUILD_DATA_CONTRACT_FAILED")


def _load_optional(value: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return unwrap_response(load_json_document(value))


def _load_or_profile(root: Path, profile_summary: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if profile_summary is not None:
        profile = unwrap_response(load_json_document(profile_summary))
        if not profile.get("profile_ref") and not isinstance(profile_summary, dict):
            profile["profile_ref"] = str(Path(profile_summary).resolve())
        return profile
    response = profile_spectral_data(root)
    if not response.get("ok"):
        raise RuntimeError(response.get("errors"))
    return response["result"]


def _build_contract(
    root: Path,
    apply_result: dict[str, Any],
    read_plan: dict[str, Any] | None,
    profile: dict[str, Any],
    *,
    status: str | None,
    confirmation_log: list[dict[str, Any]] | None,
    backend: str,
) -> dict[str, Any]:
    data_refs = _normalize_refs(dict(apply_result.get("data_refs") or {}))
    computed_status = status or _status_from_inputs(apply_result, profile, read_plan)
    task_hint = apply_result.get("task_hint") or (read_plan or {}).get("task_hint")
    sample_orientation = profile.get("sample_orientation") or (apply_result.get("execution_summary") or {}).get("sample_orientation") or (read_plan or {}).get("sample_orientation")
    source_type = _source_type(read_plan, apply_result)
    label_status = "present_with_missing" if apply_result.get("label_has_missing") else "present" if profile.get("has_y") else "absent"
    metadata_status = "present_with_missing" if apply_result.get("metadata_has_missing") else "present" if profile.get("has_metadata") else "absent"
    X_ref = data_refs.get("X_ref", "X.csv")
    y_ref = data_refs.get("y_ref")
    sample_ids_ref = data_refs.get("sample_ids_ref", "sample_ids.csv")
    band_axis_ref = data_refs.get("band_axis_ref", "band_axis.csv")
    metadata_ref = data_refs.get("metadata_ref")
    n_samples = int(profile.get("n_samples") or apply_result.get("n_samples") or 0)
    n_features = int(profile.get("n_features") or apply_result.get("n_features") or 0)
    band_unit = profile.get("band_axis_unit") or (apply_result.get("band_axis_summary") or {}).get("unit")
    band_axis_type = _band_axis_type(read_plan, apply_result)
    contract = {
        "status": "ready" if computed_status == "confirmed" else computed_status,
        "reader_version": "0.1.0",
        "source_type": source_type,
        "source": _source_summary(read_plan),
        "files": {
            "X": X_ref,
            "y": y_ref,
            "sample_ids": sample_ids_ref,
            "band_axis": band_axis_ref,
            "metadata": metadata_ref,
        },
        "shape": {"n_samples": n_samples, "n_features": n_features},
        "X": X_ref,
        "y": y_ref,
        "sample_ids": sample_ids_ref,
        "band_axis_ref": band_axis_ref,
        "metadata": metadata_ref,
        "n_samples": n_samples,
        "n_features": n_features,
        "sample_orientation": sample_orientation,
        "label_status": label_status,
        "metadata_status": metadata_status,
        "missing_value_status": apply_result.get("missing_value_status") or "unknown",
        "sample_id_status": apply_result.get("sample_id_status") or "unknown",
        "sample_id_source": apply_result.get("sample_id_source") or "unknown",
        "task_hint": task_hint or "unknown",
        "band_unit": band_unit,
        "spectral_type": (read_plan or {}).get("spectral_type"),
        "band_axis": {
            "file": band_axis_ref,
            "unit": band_unit,
            "type": band_axis_type,
            "count": n_features,
        },
        "warnings": _contract_warnings(apply_result),
    }
    table_layout = _table_layout_summary(read_plan)
    if table_layout:
        contract["table_layout"] = table_layout
    target_columns = _target_columns(read_plan)
    if target_columns:
        contract["target_columns"] = target_columns
    band_source = ((read_plan or {}).get("band_axis_source") or {})
    if band_source.get("mode") == "external_file":
        contract["band_axis"]["source"] = band_source.get("file")
    alignment = apply_result.get("alignment_summary") or {}
    label_alignment = alignment.get("label_alignment") or alignment.get("alignment_method") or (read_plan or {}).get("label_alignment")
    if label_alignment == "row_order":
        contract["label_alignment"] = "row_order"
    legacy_confirmation_key = "read" + "_plan_confirmation"
    confirmation = (read_plan or {}).get("reader_confirmation") or (read_plan or {}).get(legacy_confirmation_key)
    if isinstance(confirmation, dict):
        # Keep only the compact confirmation summary in the public contract.
        # The key deliberately avoids ``read_plan`` because downstream
        # contracts must not expose or persist the internal read plan.
        contract["reader_confirmation"] = confirmation
    return contract


def _status_from_inputs(apply_result: dict[str, Any], profile: dict[str, Any], read_plan: dict[str, Any] | None) -> str:
    if apply_result.get("apply_status") != "applied":
        return "blocked"
    if not profile.get("n_samples") or not profile.get("n_features"):
        return "blocked"
    if not profile.get("has_y") and (read_plan or {}).get("task_hint") in {"classification", "regression"}:
        return "blocked"
    if (read_plan or {}).get("required_confirmations"):
        return "provisional"
    return "confirmed"


def _contract_warnings(apply_result: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = list(apply_result.get("warnings") or [])
    if apply_result.get("missing_value_status") == "present":
        warnings.append({
            "code": "X_MISSING_VALUES_PRESENT",
            "message": "X contains missing values; handle in spectral-qc.",
            "severity": "warning",
        })
    if apply_result.get("sample_id_status") == "partially_generated_after_confirmation":
        warnings.append({
            "code": "SAMPLE_IDS_PARTIALLY_GENERATED",
            "message": "Some sample IDs were generated after user confirmation.",
            "severity": "warning",
        })
    if apply_result.get("label_has_missing"):
        warnings.append({
            "code": "Y_MISSING_VALUES_PRESENT",
            "message": "Labels or targets contain missing values; handle in spectral-qc.",
            "severity": "warning",
        })
    if apply_result.get("metadata_has_missing"):
        warnings.append({
            "code": "METADATA_MISSING_VALUES_PRESENT",
            "message": "Metadata contains missing values.",
            "severity": "warning",
        })
    return warnings


def _downstream_readiness(status: str, profile: dict[str, Any], apply_result: dict[str, Any]) -> dict[str, Any]:
    has_y = bool(profile.get("has_y"))
    return {
        "spectral_splitter": {"ready": status == "confirmed", "supervised_ready": status == "confirmed" and has_y},
        "unsupervised": {"ready": status == "confirmed", "reason": "requires confirmed readable X and band_axis"},
        "blocked": status == "blocked",
    }


def _role_name(role: Any) -> str | None:
    if isinstance(role, dict):
        value = role.get("column") or role.get("name")
        return str(value) if value else None
    return None


def _normalize_refs(refs: dict[str, Any]) -> dict[str, str]:
    key_map = {
        "X_ref": "X_ref",
        "y_ref": "y_ref",
        "sample_ids_ref": "sample_ids_ref",
        "band_axis_ref": "band_axis_ref",
        "metadata_ref": "metadata_ref",
    }
    normalized: dict[str, str] = {}
    for key, out_key in key_map.items():
        value = refs.get(key)
        if not value:
            continue
        text = str(value)
        if text.startswith("data/"):
            text = text[len("data/") :]
        normalized[out_key] = text
    return normalized


def _source_type(read_plan: dict[str, Any] | None, apply_result: dict[str, Any]) -> str:
    if (read_plan or {}).get("mixed_folder"):
        return "folder"
    mode = (read_plan or {}).get("read_mode") or (apply_result.get("execution_summary") or {}).get("read_mode")
    if mode == "sample_files_folder":
        return "folder"
    file_type = (read_plan or {}).get("file_type")
    if file_type:
        suffix = str(file_type).lstrip(".")
        return "hdf5" if suffix in {"h5", "hdf5"} else "netcdf" if suffix == "nc" else suffix
    source = (read_plan or {}).get("source_path") or apply_result.get("original_source_path")
    if source:
        suffix = Path(str(source)).suffix.lower().lstrip(".")
        if suffix:
            return "hdf5" if suffix in {"h5", "hdf5"} else "netcdf" if suffix == "nc" else suffix
    return "unknown"


def _band_axis_type(read_plan: dict[str, Any] | None, apply_result: dict[str, Any]) -> str:
    band_source = ((read_plan or {}).get("band_axis_source") or {})
    if (read_plan or {}).get("band_type"):
        return str((read_plan or {}).get("band_type"))
    if band_source.get("mode") == "external_file":
        return str(band_source.get("type") or "external_file")
    source = ((read_plan or {}).get("band_axis") or {}).get("source")
    if source == "generated_index":
        return "generated_index"
    unit = (read_plan or {}).get("band_unit") or (apply_result.get("band_axis_summary") or {}).get("unit")
    if unit == "cm-1":
        return "wavenumber"
    if unit == "nm":
        return "wavelength"
    return "unknown"


def _table_layout_summary(read_plan: dict[str, Any] | None) -> dict[str, Any] | None:
    plan = read_plan or {}
    layout = plan.get("table_layout") or {}
    if not layout:
        return None
    header_rows = layout.get("header_rows") or []
    start = layout.get("spectral_start_column")
    end = layout.get("spectral_end_column")
    summary: dict[str, Any] = {"header_type": "multirow" if header_rows else "single"}
    if start is not None or end is not None:
        summary["spectral_region"] = {"start": start, "end": end}
    return summary


def _target_columns(read_plan: dict[str, Any] | None) -> list[str]:
    role = (read_plan or {}).get("target") or {}
    if isinstance(role, dict):
        columns = role.get("columns")
        if isinstance(columns, list):
            return [str(column) for column in columns]
        column = role.get("column")
        if column:
            return [str(column)]
    multi = (read_plan or {}).get("multi_target") or {}
    columns = multi.get("target_columns")
    return [str(column) for column in columns] if isinstance(columns, list) else []


def _source_summary(read_plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = read_plan or {}
    mixed_folder = plan.get("mixed_folder") or {}
    workbook = plan.get("workbook") or {}
    variable_map = plan.get("variable_map") or {}
    container = plan.get("container") or {}
    dataset_map = plan.get("dataset_map") or {}
    hierarchical = plan.get("hierarchical_container") or {}
    summary = {
        "input": str(plan.get("source_path")) if plan.get("source_path") is not None else None,
        "spectral_sheet": workbook.get("spectral_sheet") or plan.get("sheet_name"),
        "label_sheet": workbook.get("label_sheet") or (plan.get("label_file") or {}).get("sheet_name"),
        "X_variable": variable_map.get("X") or container.get("x_var"),
        "y_variable": variable_map.get("y") or container.get("y_var"),
        "sample_ids_variable": variable_map.get("sample_ids") or container.get("sample_ids_var"),
        "band_axis_variable": variable_map.get("band_axis") or container.get("band_axis_var"),
        "metadata_variable": variable_map.get("metadata") or container.get("metadata_var"),
        "X_path": dataset_map.get("X") or hierarchical.get("x_path"),
        "y_path": dataset_map.get("y") or hierarchical.get("y_path"),
        "sample_ids_path": dataset_map.get("sample_ids") or hierarchical.get("sample_ids_path"),
        "band_axis_path": dataset_map.get("band_axis") or hierarchical.get("band_axis_path"),
        "metadata_path": dataset_map.get("metadata") or hierarchical.get("metadata_path"),
    }
    if mixed_folder:
        summary.update({
            "input": mixed_folder.get("input"),
            "spectra_file": mixed_folder.get("spectra_file"),
            "label_file": mixed_folder.get("label_file"),
            "metadata_file": mixed_folder.get("metadata_file"),
            "band_axis_file": mixed_folder.get("band_axis_file"),
        })
    return summary

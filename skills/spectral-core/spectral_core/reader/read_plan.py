"""Read-plan loading and conservative draft helpers."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .io_utils import load_json_file, write_json_file
from .response import ok_response


def load_json_document(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a JSON object from a path or return a dict copy."""

    if isinstance(value, dict):
        return dict(value)
    return load_json_file(value)


def unwrap_response(value: dict[str, Any]) -> dict[str, Any]:
    """Return result payload when a tool envelope is supplied."""

    if isinstance(value, dict) and isinstance(value.get("result"), dict):
        return dict(value["result"])
    return dict(value or {})


def propose_read_plan(
    preview_result: dict[str, Any] | str | Path | None = None,
    *,
    backend: str = "core",
    mode: str = "conservative",
) -> dict[str, Any]:
    """Build a conservative provisional read_plan draft from preview evidence."""

    preview_doc = unwrap_response(load_json_document(preview_result)) if preview_result else {}
    file_previews = list(preview_doc.get("file_previews") or [])
    first_preview = file_previews[0] if file_previews else {}
    inventory = list(preview_doc.get("file_inventory") or [])
    first_inventory = inventory[0] if inventory else {}
    input_kind = preview_doc.get("input_kind") or "unknown"
    source_path = preview_doc.get("input_path") or first_preview.get("path") or first_inventory.get("path")
    suffix = first_preview.get("suffix") or first_inventory.get("suffix")
    delimiter = _first_value(first_preview.get("delimiter_candidates"), "delimiter")
    header_row = _first_value(first_preview.get("header_row_candidates"), "row_index")
    band_evidence = list(first_preview.get("band_like_column_evidence") or [])
    sample_evidence = list(first_preview.get("sample_id_like_column_evidence") or [])
    label_evidence = list(first_preview.get("label_like_column_evidence") or [])
    metadata_evidence = list(first_preview.get("metadata_like_column_evidence") or [])
    columns_evidence = first_preview.get("samples_as_columns_evidence") or {}
    multi_file_evidence = preview_doc.get("multi_file_evidence") or {}
    preamble = list(first_preview.get("leading_preamble_candidates") or [])

    selected_fragments = _select_fragments(preview_doc, first_preview)
    required_confirmations = []
    if preamble:
        required_confirmations.append(_confirmation("confirm_preamble_skip", "Confirm leading preamble lines should be skipped.", field="skiprows"))
    if band_evidence:
        required_confirmations.append(_confirmation("confirm_spectral_columns", "Confirm proposed spectral column boundaries.", field="spectral_columns"))
    if input_kind == "folder":
        required_confirmations.append(_confirmation("confirm_sample_files_folder", "Confirm one-file-per-sample folder interpretation.", field="read_mode"))
    if label_evidence:
        required_confirmations.append(_confirmation("confirm_label_column", "Confirm proposed label column.", field="label"))
    samples_as_columns = bool(columns_evidence.get("candidate"))
    if samples_as_columns:
        required_confirmations.extend([
            _confirmation("confirm_samples_as_columns", "Confirm samples are stored as columns.", field="sample_orientation"),
            _confirmation("confirm_band_axis_column", "Confirm the band axis column.", field="samples_as_columns.band_axis_column"),
            _confirmation("confirm_sample_id_columns", "Confirm sample IDs come from sample columns.", field="samples_as_columns.sample_id_source"),
            _confirmation("confirm_no_label_or_confirm_label_source", "Confirm whether labels are absent or provided externally.", field="label_file"),
        ])
    label_file_candidate = _first_label_file_candidate(multi_file_evidence)
    if label_file_candidate:
        required_confirmations.extend([
            _confirmation("confirm_spectral_file", "Confirm the spectral matrix file.", field="source_path"),
            _confirmation("confirm_label_file", "Confirm the external label file.", field="label_file.path"),
            _confirmation("confirm_join_key", "Confirm sample_id join key for alignment.", field="alignment_plan.join_key"),
            _confirmation("confirm_label_column", "Confirm external label column.", field="label_file.label_column"),
            _confirmation("confirm_alignment_policy", "Confirm external label alignment policy.", field="alignment_plan"),
        ])

    read_mode = "sample_files_folder" if input_kind == "folder" else "matrix_file" if suffix else "unknown"
    read_plan = {
        "read_plan_id": f"rp_{uuid.uuid4().hex[:12]}",
        "read_plan_version": "0.1.0",
        "read_plan_status": "provisional",
        "created_by": "propose_read_plan",
        "source_preview_ref": None,
        "source_path": source_path,
        "input_kind": input_kind,
        "file_type": suffix,
        "encoding": first_preview.get("encoding_hint"),
        "delimiter": delimiter,
        "skiprows": len(preamble) if preamble else 0,
        "header_row": header_row,
        "sheet_name": None,
        "variable_name": None,
        "json_path": None,
        "array_key": None,
        "folder_mode": "one_file_per_sample" if input_kind == "folder" else None,
        "read_mode": read_mode,
        "sample_orientation": "columns" if samples_as_columns else "unknown" if input_kind != "folder" else "one_file_per_sample",
        "sample_axis": None,
        "feature_axis": None,
        "table_region": {},
        "data_start_row": (header_row + 1) if isinstance(header_row, int) else None,
        "data_end_row": None,
        "data_start_column": None,
        "data_end_column": None,
        "sample_id": _role_from_evidence(sample_evidence, required=True),
        "label": _role_from_evidence(label_evidence, required=False),
        "target": {"source": None, "column": None, "required": False, "status": "not_detected", "evidence": []},
        "metadata": {"columns": [item.get("name") for item in metadata_evidence if item.get("name")], "source": "preview_evidence"},
        "spectral_columns": {"columns": [item.get("name") for item in band_evidence if item.get("name")], "source": "preview_evidence"},
        "band_axis": {"source": "column_headers" if band_evidence else None},
        "samples_as_columns": {
            "enabled": samples_as_columns,
            "band_axis_column": columns_evidence.get("band_axis_column_candidate") if samples_as_columns else None,
            "sample_id_source": "column_headers" if samples_as_columns else None,
            "sample_id_row": header_row if samples_as_columns else None,
            "label_row": None,
            "metadata_rows": [],
            "data_start_row": (header_row + 1) if samples_as_columns and isinstance(header_row, int) else None,
            "data_end_row": None,
            "sample_start_column": 1 if samples_as_columns else None,
            "sample_end_column": None,
            "transpose_required": samples_as_columns,
        },
        "band_unit": _band_unit(band_evidence),
        "spectral_type": None,
        "task_hint": "classification" if label_evidence else "unknown",
        "label_file": _draft_label_file(label_file_candidate),
        "band_axis_file": {},
        "metadata_file": {},
        "external_validation_file": {},
        "alignment_plan": _draft_alignment_plan(label_file_candidate),
        "sample_file_pattern": "*.*" if input_kind == "folder" else None,
        "sample_file_recursive": True if input_kind == "folder" else None,
        "sample_file_columns": {"band_axis": None, "value": None} if input_kind == "folder" else {},
        "file_name_as_sample_id": True if input_kind == "folder" else None,
        "folder_name_as_label": None,
        "per_file_band_axis_policy": "require_consistent_axis" if input_kind == "folder" else None,
        "preview_evidence": {
            "input_kind": input_kind,
            "preview_status": preview_doc.get("preview_status"),
            "header_row_candidates": first_preview.get("header_row_candidates") or [],
            "leading_preamble_candidates": preamble,
            "samples_as_columns_evidence": columns_evidence,
            "multi_file_evidence": multi_file_evidence,
        },
        "decision_evidence": _decision_evidence(first_preview),
        "selected_fragments": selected_fragments,
        "selected_references": ["references/read-plan-patterns.md"],
        "confidence_scores": {"overall": 0.55, "mode": mode},
        "required_confirmations": required_confirmations,
        "recommended_confirmations": [],
        "confirmed_items": [],
        "unresolved_items": [item["id"] for item in required_confirmations],
        "execution_intent": {"tool": "apply_read_plan", "write_package": False, "validation_level": "standard", "expected_backend": None},
        "expected_outputs": ["preview_evidence_only_until_apply_read_plan"],
        "blocked_reasons": [],
        "downstream_readiness_hint": {"apply_read_plan": False, "reason": "Provisional draft requires Agent review and user confirmation."},
    }
    return ok_response(
        "propose_read_plan",
        {
            "read_plan": read_plan,
            "selected_evidence": read_plan["decision_evidence"],
            "required_confirmations": required_confirmations,
            "confidence_summary": read_plan["confidence_scores"],
            "notes_for_agent": [
                "This is a conservative provisional draft, not a final decision.",
                "Agent must review fragments/references and ask required confirmations before apply_read_plan.",
                "No X, y, Data Contract, or package was generated.",
            ],
        },
        backend=backend,
    )


def confirm_read_plan(
    draft_read_plan: dict[str, Any] | str | Path | None = None,
    *,
    output: str | Path | None = None,
    confirm_items: list[str] | None = None,
    set_fields: dict[str, Any] | None = None,
    source_base_dir: str | None = None,
    created_by_type: str = "agent",
    created_by_detail: str | None = None,
    backend: str = "core",
) -> dict[str, Any]:
    """Apply confirmations and field updates to a draft read_plan."""

    if draft_read_plan is None:
        from .response import error_response

        return error_response("confirm_read_plan", "No draft read_plan was provided.", backend=backend, code="DRAFT_READ_PLAN_MISSING")
    try:
        plan = unwrap_response(load_json_document(draft_read_plan))
    except Exception as exc:
        from .response import error_response

        return error_response("confirm_read_plan", f"Could not load draft read_plan: {exc}", backend=backend, code="DRAFT_READ_PLAN_LOAD_FAILED")

    confirmations = list(confirm_items or [])
    updated = apply_confirmation_items(plan, confirmations)
    updated = set_read_plan_fields(updated, set_fields or {})
    if source_base_dir is not None:
        updated["source_base_dir"] = source_base_dir
    updated["created_by_type"] = created_by_type
    if created_by_detail:
        updated["created_by_detail"] = created_by_detail
    updated = normalize_confirmed_read_plan(updated)

    from .validator import validate_read_plan

    validation = validate_read_plan(updated, backend=backend)
    status = (validation.get("result") or {}).get("read_plan_status")
    updated["read_plan_status"] = "confirmed" if status == "confirmed" else status or updated.get("read_plan_status", "provisional")
    if output:
        write_json_file(output, updated, ensure_ascii=False, no_bom=True)
    result = {
        "read_plan": updated,
        "read_plan_status": updated.get("read_plan_status"),
        "confirmed_items": updated.get("confirmed_items") or [],
        "unresolved_items": updated.get("unresolved_items") or [],
        "validation": validation.get("result", {}),
        "output_ref": str(output) if output else None,
    }
    return ok_response("confirm_read_plan", result, backend=backend, warnings=validation.get("warnings", []))


def apply_confirmation_items(plan: dict[str, Any], confirm_items: list[str]) -> dict[str, Any]:
    updated = dict(plan)
    confirmed = set(updated.get("confirmed_items") or [])
    confirmed.update(confirm_items)
    required = []
    for item in updated.get("required_confirmations") or []:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id") or item.get("field") or item.get("question")
        if identifier in confirmed:
            patched = dict(item)
            patched["status"] = "confirmed"
            required.append(patched)
        else:
            required.append(item)
    unresolved = []
    for item in required:
        identifier = item.get("id") or item.get("field") or item.get("question")
        if item.get("status") != "confirmed" and identifier not in confirmed:
            unresolved.append(identifier)
    for item in updated.get("unresolved_items") or []:
        if item not in confirmed and item not in unresolved:
            unresolved.append(item)
    updated["confirmed_items"] = sorted(confirmed)
    updated["required_confirmations"] = [item for item in required if item.get("status") != "confirmed"]
    updated["unresolved_items"] = unresolved
    return updated


def set_read_plan_fields(plan: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    updated = dict(plan)
    for key, value in fields.items():
        _set_nested(updated, key.split("."), value)
    return updated


def normalize_confirmed_read_plan(plan: dict[str, Any]) -> dict[str, Any]:
    updated = dict(plan)
    unresolved = list(updated.get("unresolved_items") or [])
    required = list(updated.get("required_confirmations") or [])
    if not unresolved and not required and updated.get("read_plan_status") != "blocked":
        updated["read_plan_status"] = "confirmed"
        hint = dict(updated.get("downstream_readiness_hint") or {})
        hint["apply_read_plan"] = True
        hint["reason"] = "All required confirmations are resolved."
        updated["downstream_readiness_hint"] = hint
    elif updated.get("read_plan_status") != "blocked":
        updated["read_plan_status"] = "provisional"
    return updated


def _set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def _role_from_evidence(evidence: list[dict[str, Any]], *, required: bool) -> dict[str, Any]:
    first = evidence[0] if evidence else {}
    return {
        "source": "preview_evidence" if first else None,
        "column": first.get("name"),
        "required": required,
        "status": "candidate" if first else "not_detected",
        "evidence": evidence,
    }


def _first_value(items: Any, key: str) -> Any:
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0].get(key)
    return None


def _band_unit(band_evidence: list[dict[str, Any]]) -> str | None:
    units = {item.get("unit_hint") for item in band_evidence if item.get("unit_hint")}
    if "cm-1" in units:
        return "cm-1"
    if "nm" in units:
        return "nm"
    if units:
        return "unknown"
    return None


def _confirmation(identifier: str, question: str, *, field: str) -> dict[str, Any]:
    return {"id": identifier, "field": field, "question": question, "status": "unresolved"}


def _select_fragments(preview_doc: dict[str, Any], first_preview: dict[str, Any]) -> list[str]:
    fragments = []
    if first_preview.get("leading_preamble_candidates"):
        fragments.append("static/fragments/csv-with-preamble.md")
    if first_preview.get("metadata_like_column_evidence"):
        fragments.append("static/fragments/metadata-before-spectra.md")
    units = {item.get("unit_hint") for item in first_preview.get("band_like_column_evidence") or []}
    if "cm-1" in units:
        fragments.append("static/fragments/wavenumber-columns.md")
    if "nm" in units:
        fragments.append("static/fragments/wavelength-columns.md")
    if preview_doc.get("input_kind") == "folder":
        fragments.append("static/fragments/sample-files-folder.md")
    if (first_preview.get("samples_as_columns_evidence") or {}).get("candidate"):
        fragments.append("static/fragments/samples-as-columns.md")
    if (preview_doc.get("multi_file_evidence") or {}).get("possible_label_files"):
        fragments.append("static/fragments/external-label-file.md")
    return fragments


def _decision_evidence(first_preview: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for field in [
        "delimiter_candidates",
        "leading_preamble_candidates",
        "header_row_candidates",
        "band_like_column_evidence",
        "sample_id_like_column_evidence",
        "label_like_column_evidence",
        "metadata_like_column_evidence",
        "samples_as_columns_evidence",
    ]:
        values = first_preview.get(field) or []
        if values:
            evidence.append({"field": field, "values": values})
    return evidence


def _first_label_file_candidate(evidence: dict[str, Any]) -> dict[str, Any] | None:
    candidates = evidence.get("possible_label_files") or []
    return candidates[0] if candidates else None


def _draft_label_file(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "path": candidate.get("path"),
        "file_type": candidate.get("suffix"),
        "encoding": None,
        "delimiter": None,
        "skiprows": 0,
        "header_row": 0,
        "sample_id_column": "sample_id",
        "label_column": None,
        "target_columns": [],
        "metadata_columns": [],
        "required": True,
    }


def _draft_alignment_plan(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "join_key": "sample_id",
        "left_source": "spectra",
        "right_source": "label_file",
        "join_type": "left",
        "preserve_spectrum_order": True,
        "allow_unmatched_spectra": False,
        "allow_unmatched_labels": True,
        "duplicate_policy": "blocked",
        "missing_label_policy": "blocked",
    }

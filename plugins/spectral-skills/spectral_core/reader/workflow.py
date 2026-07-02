"""User-facing spectral-reader one-shot entry.

The reader is not a workflow archive. It builds transient internal read settings
when needed, then writes only downstream-ready standard data files.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .contract import build_spectral_data_contract
from .executor import apply_read_plan
from .io_utils import resolve_path
from .package_writer import write_standardized_package
from .response import error_response, ok_response


def read_spectral_dataset(
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    source_base_dir: str | None = None,
    spectra_file: str | Path | None = None,
    sheet: str | None = None,
    sheet_index: int | None = None,
    spectral_sheet: str | None = None,
    label_sheet: str | None = None,
    sample_orientation: str | None = None,
    sample_id_column: str | None = None,
    sample_id_column_index: int | None = None,
    label_column: str | None = None,
    target_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    spectral_columns: list[str] | None = None,
    band_axis_column: str | int | None = None,
    x_var: str | None = None,
    y_var: str | None = None,
    sample_ids_var: str | None = None,
    sample_ids_file: str | Path | None = None,
    band_axis_var: str | None = None,
    metadata_var: str | None = None,
    x_path: str | None = None,
    y_path: str | None = None,
    sample_ids_path: str | None = None,
    band_axis_path: str | None = None,
    metadata_path: str | None = None,
    label_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    join_key: str | None = None,
    folder_name_as_label: bool = False,
    file_name_as_label: bool = False,
    sample_file_pattern: str | None = None,
    sample_file_value_column: str | int | None = None,
    sample_file_band_column: str | int | None = None,
    label_alignment: str | None = None,
    allow_row_order_labels: bool = False,
    allow_generated_sample_ids_for_missing: bool = False,
    missing_sample_id_policy: str | None = None,
    missing_value_tokens: list[str] | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    skiprows: int | None = None,
    header_row: int | None = None,
    header_rows: list[int] | None = None,
    data_start_row: int | None = None,
    data_end_row: int | None = None,
    spectral_start_column: str | int | None = None,
    spectral_end_column: str | int | None = None,
    band_axis_file: str | Path | None = None,
    band_unit: str | None = None,
    band_type: str | None = None,
    spectral_type: str | None = None,
    task_type: str | None = None,
    max_auto_columns: int = 10000,
    max_spectral_columns: int = 20000,
    wide_table_mode: str = "auto",
    confirm_wide_table: bool = False,
    confirm_read_plan: bool = False,
    auto_folder: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    """Read a spectral dataset into standard downstream-ready files."""

    return _read_one_shot(
        input_path=input_path,
        output_dir=output_dir,
        source_base_dir=source_base_dir,
        spectra_file=spectra_file,
        sheet=sheet,
        sheet_index=sheet_index,
        spectral_sheet=spectral_sheet,
        label_sheet=label_sheet,
        sample_orientation=sample_orientation,
        sample_id_column=sample_id_column,
        sample_id_column_index=sample_id_column_index,
        label_column=label_column,
        target_columns=target_columns,
        metadata_columns=metadata_columns,
        spectral_columns=spectral_columns,
        band_axis_column=band_axis_column,
        x_var=x_var,
        y_var=y_var,
        sample_ids_var=sample_ids_var,
        sample_ids_file=sample_ids_file,
        band_axis_var=band_axis_var,
        metadata_var=metadata_var,
        x_path=x_path,
        y_path=y_path,
        sample_ids_path=sample_ids_path,
        band_axis_path=band_axis_path,
        metadata_path=metadata_path,
        label_file=label_file,
        metadata_file=metadata_file,
        join_key=join_key,
        folder_name_as_label=folder_name_as_label,
        file_name_as_label=file_name_as_label,
        sample_file_pattern=sample_file_pattern,
        sample_file_value_column=sample_file_value_column,
        sample_file_band_column=sample_file_band_column,
        label_alignment=label_alignment,
        allow_row_order_labels=allow_row_order_labels,
        allow_generated_sample_ids_for_missing=allow_generated_sample_ids_for_missing,
        missing_sample_id_policy=missing_sample_id_policy,
        missing_value_tokens=missing_value_tokens,
        delimiter=delimiter,
        encoding=encoding,
        skiprows=skiprows,
        header_row=header_row,
        header_rows=header_rows,
        data_start_row=data_start_row,
        data_end_row=data_end_row,
        spectral_start_column=spectral_start_column,
        spectral_end_column=spectral_end_column,
        band_axis_file=band_axis_file,
        band_unit=band_unit,
        band_type=band_type,
        spectral_type=spectral_type,
        task_type=task_type,
        max_auto_columns=max_auto_columns,
        max_spectral_columns=max_spectral_columns,
        wide_table_mode=wide_table_mode,
        confirm_wide_table=confirm_wide_table,
        confirm_read_plan=confirm_read_plan,
        auto_folder=auto_folder,
        overwrite=overwrite,
        strict=strict,
        backend=backend,
    )


def _blocked_from_tool(stage: str, response: dict[str, Any], out: Path, backend: str) -> dict[str, Any]:
    errors = response.get("errors") or []
    message = errors[0].get("message") if errors else f"{stage} failed."
    schema_details = _schema_error_details(response)
    if schema_details:
        first_schema_error = (schema_details.get("schema_errors") or [{}])[0]
        field = (first_schema_error.get("details") or {}).get("field") or (first_schema_error.get("details") or {}).get("path")
        message = f"read_plan schema validation failed for {field}." if field else "read_plan schema validation failed."
    return error_response(
        "read_spectral_dataset",
        str(message),
        backend=backend,
        code="READ_PLAN_SCHEMA_INVALID" if schema_details else errors[0].get("code", "READER_BLOCKED") if errors else "READER_BLOCKED",
        result={
            "status": "blocked",
            "output_dir": str(out),
            "blocked_stage": stage,
            "blocking_reason": str(message),
            "next_step_hint": "Fix the reader input or parameters and run again.",
        },
        warnings=response.get("warnings", []),
        details=schema_details or errors[0].get("details", {}) if errors else {},
    )


def _schema_error_details(response: dict[str, Any]) -> dict[str, Any] | None:
    errors = response.get("errors") or []
    if not errors:
        return None
    validation = ((errors[0].get("details") or {}).get("validation") or {})
    schema_errors = validation.get("schema_errors") or []
    if not schema_errors:
        return None
    first = schema_errors[0]
    details = first.get("details") or {}
    hint = details.get("hint")
    return {
        "schema_errors": schema_errors,
        "field": details.get("field") or details.get("path"),
        "received": details.get("received"),
        "allowed": details.get("allowed"),
        "hint": hint,
    }


def _confirmation_from_apply_response(response: dict[str, Any]) -> dict[str, Any] | None:
    errors = response.get("errors") or []
    if not errors:
        return None
    error = errors[0]
    if error.get("code") != "SAMPLE_ID_MISSING":
        return None
    details = error.get("details") or {}
    return {
        "status": "needs_confirmation",
        "reason": "sample_id contains missing values; confirm generated replacement sample IDs before reading.",
        "required_fields": ["missing_sample_id_policy"],
        "suggested_arguments": details.get("suggested_arguments") or {"--missing-sample-id-policy": ["generate", "blocked"], "--confirm-generate-missing-sample-ids": True},
    }


def _public_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hidden = {"LOW_CONFIDENCE_FIELD", "OPTIONAL_PACKAGE_ARTIFACT_MISSING", "SMALL_SAMPLE_WARNING_PRESENT"}
    return [item for item in warnings if item.get("code") not in hidden]


def _read_one_shot(
    *,
    input_path: str | Path | None,
    output_dir: str | Path | None,
    source_base_dir: str | None,
    spectra_file: str | Path | None,
    sheet: str | None,
    sheet_index: int | None,
    spectral_sheet: str | None,
    label_sheet: str | None,
    sample_orientation: str | None,
    sample_id_column: str | None,
    sample_id_column_index: int | None,
    label_column: str | None,
    target_columns: list[str] | None,
    metadata_columns: list[str] | None,
    spectral_columns: list[str] | None,
    band_axis_column: str | int | None,
    x_var: str | None,
    y_var: str | None,
    sample_ids_var: str | None,
    sample_ids_file: str | Path | None,
    band_axis_var: str | None,
    metadata_var: str | None,
    x_path: str | None,
    y_path: str | None,
    sample_ids_path: str | None,
    band_axis_path: str | None,
    metadata_path: str | None,
    label_file: str | Path | None,
    metadata_file: str | Path | None,
    join_key: str | None,
    folder_name_as_label: bool,
    file_name_as_label: bool,
    sample_file_pattern: str | None,
    sample_file_value_column: str | int | None,
    sample_file_band_column: str | int | None,
    label_alignment: str | None,
    allow_row_order_labels: bool,
    allow_generated_sample_ids_for_missing: bool,
    missing_sample_id_policy: str | None,
    missing_value_tokens: list[str] | None,
    delimiter: str | None,
    encoding: str | None,
    skiprows: int | None,
    header_row: int | None,
    header_rows: list[int] | None,
    data_start_row: int | None,
    data_end_row: int | None,
    spectral_start_column: str | int | None,
    spectral_end_column: str | int | None,
    band_axis_file: str | Path | None,
    band_unit: str | None,
    band_type: str | None,
    spectral_type: str | None,
    task_type: str | None,
    max_auto_columns: int,
    max_spectral_columns: int,
    wide_table_mode: str,
    confirm_wide_table: bool,
    confirm_read_plan: bool,
    auto_folder: bool,
    overwrite: bool,
    strict: bool,
    backend: str,
) -> dict[str, Any]:
    if input_path is None:
        return error_response("read_spectral_dataset", "--input is required.", backend=backend, code="INPUT_REQUIRED", result={"status": "blocked"})
    if output_dir is None:
        return error_response("read_spectral_dataset", "--output-dir is required.", backend=backend, code="OUTPUT_DIR_REQUIRED", result={"status": "blocked"})

    from .preview import preview_file

    path_resolution = resolve_path(input_path, base_dir=source_base_dir, cwd=Path.cwd())
    preview_input_path = path_resolution.get("resolved_path") or str(input_path)
    preview = preview_file(
        str(preview_input_path),
        backend=backend,
        options={"max_auto_columns": max_auto_columns},
    )
    if not preview.get("ok"):
        errors = preview.get("errors") or []
        first_error = errors[0] if errors else {}
        return error_response(
            "read_spectral_dataset",
            first_error.get("message") or "Input preview failed.",
            backend=backend,
            code=first_error.get("code") or "PREVIEW_FAILED",
            result={"status": "blocked", "reason": first_error.get("message") or "input preview failed"},
            warnings=preview.get("warnings", []),
            details=first_error.get("details", {}),
        )
    if path_resolution.get("exists"):
        input_path = Path(str(preview_input_path))

    plan_result = _build_transient_read_plan(
        input_path=input_path,
        preview_result=preview.get("result") or {},
        source_base_dir=source_base_dir,
        spectra_file=spectra_file,
        sheet=sheet,
        sheet_index=sheet_index,
        spectral_sheet=spectral_sheet,
        label_sheet=label_sheet,
        sample_orientation=sample_orientation,
        sample_id_column=sample_id_column,
        sample_id_column_index=sample_id_column_index,
        label_column=label_column,
        target_columns=target_columns,
        metadata_columns=metadata_columns,
        spectral_columns=spectral_columns,
        band_axis_column=band_axis_column,
        x_var=x_var,
        y_var=y_var,
        sample_ids_var=sample_ids_var,
        sample_ids_file=sample_ids_file,
        band_axis_var=band_axis_var,
        metadata_var=metadata_var,
        x_path=x_path,
        y_path=y_path,
        sample_ids_path=sample_ids_path,
        band_axis_path=band_axis_path,
        metadata_path=metadata_path,
        label_file=label_file,
        metadata_file=metadata_file,
        join_key=join_key,
        folder_name_as_label=folder_name_as_label,
        file_name_as_label=file_name_as_label,
        sample_file_pattern=sample_file_pattern,
        sample_file_value_column=sample_file_value_column,
        sample_file_band_column=sample_file_band_column,
        label_alignment=label_alignment,
        allow_row_order_labels=allow_row_order_labels,
        allow_generated_sample_ids_for_missing=allow_generated_sample_ids_for_missing,
        missing_sample_id_policy=missing_sample_id_policy,
        missing_value_tokens=missing_value_tokens,
        delimiter=delimiter,
        encoding=encoding,
        skiprows=skiprows,
        header_row=header_row,
        header_rows=header_rows,
        data_start_row=data_start_row,
        data_end_row=data_end_row,
        spectral_start_column=spectral_start_column,
        spectral_end_column=spectral_end_column,
        band_axis_file=band_axis_file,
        band_unit=band_unit,
        band_type=band_type,
        spectral_type=spectral_type,
        task_type=task_type,
        max_auto_columns=max_auto_columns,
        max_spectral_columns=max_spectral_columns,
        wide_table_mode=wide_table_mode,
        confirm_wide_table=confirm_wide_table,
        confirm_read_plan=confirm_read_plan,
        auto_folder=auto_folder,
    )
    if plan_result["status"] != "ready":
        if plan_result["status"] == "blocked":
            return error_response(
                "read_spectral_dataset",
                plan_result["reason"],
                backend=backend,
                code=plan_result.get("code") or "READ_SEMANTICS_BLOCKED",
                result={
                    "status": "blocked",
                    "reason": plan_result["reason"],
                    "required_fields": plan_result.get("required_fields", []),
                    "suggested_arguments": plan_result.get("suggested_arguments", {}),
                },
                warnings=preview.get("warnings", []),
            )
        return ok_response(
            "read_spectral_dataset",
            {
                "status": "needs_confirmation",
                "reason": plan_result["reason"],
                "required_fields": plan_result["required_fields"],
                "suggested_arguments": plan_result["suggested_arguments"],
            },
            backend=backend,
            warnings=preview.get("warnings", []),
        )

    return _execute_confirmed_plan(
        plan_result["read_plan"],
        output_dir,
        overwrite=overwrite,
        strict=strict,
        backend=backend,
        warnings=preview.get("warnings", []),
    )


def _execute_confirmed_plan(
    read_plan: dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool,
    strict: bool,
    backend: str,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        return error_response(
            "read_spectral_dataset",
            "output_dir already exists and is not empty. Use --overwrite to rebuild.",
            backend=backend,
            code="OUTPUT_DIR_EXISTS",
            result={"status": "blocked", "output_dir": str(out)},
        )
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="spectral_reader_") as temp_name:
            temp_root = Path(temp_name)
            apply_root = temp_root / "apply"
            contract_root = temp_root / "contract"
            applied = apply_read_plan(read_plan, output_dir=apply_root, overwrite=True, strict=strict, backend=backend)
            if not applied.get("ok"):
                confirmation = _confirmation_from_apply_response(applied)
                if confirmation:
                    return ok_response("read_spectral_dataset", confirmation, backend=backend, warnings=_public_warnings(warnings or []))
                return _blocked_from_tool("apply_read_plan", applied, out, backend)
            contract = build_spectral_data_contract(
                apply_dir=apply_root,
                read_plan=read_plan,
                output=contract_root / "data_contract.json",
                backend=backend,
            )
            if not contract.get("ok"):
                return _blocked_from_tool("build_data_contract", contract, out, backend)
            written = write_standardized_package(
                apply_dir=apply_root,
                contract_dir=contract_root,
                output_dir=out,
                overwrite=True,
                strict=strict,
                backend=backend,
            )
            if not written.get("ok"):
                return _blocked_from_tool("write_standardized_package", written, out, backend)
            result = _compact_ready_result(dict(written.get("result") or {}))
            return ok_response(
                "read_spectral_dataset",
                result,
                backend=backend,
                warnings=_public_warnings((warnings or []) + (applied.get("warnings") or []) + (contract.get("warnings") or []) + (written.get("warnings") or [])),
            )
    except Exception as exc:
        return error_response(
            "read_spectral_dataset",
            f"Reader failed: {exc}",
            backend=backend,
            code="READER_FAILED",
            result={"status": "blocked", "output_dir": str(out), "reason": str(exc)},
        )


def _build_transient_read_plan(
    *,
    input_path: str | Path,
    preview_result: dict[str, Any],
    source_base_dir: str | None,
    spectra_file: str | Path | None,
    sheet: str | None,
    sheet_index: int | None,
    spectral_sheet: str | None,
    label_sheet: str | None,
    sample_orientation: str | None,
    sample_id_column: str | None,
    sample_id_column_index: int | None,
    label_column: str | None,
    target_columns: list[str] | None,
    metadata_columns: list[str] | None,
    spectral_columns: list[str] | None,
    band_axis_column: str | int | None,
    x_var: str | None,
    y_var: str | None,
    sample_ids_var: str | None,
    sample_ids_file: str | Path | None,
    band_axis_var: str | None,
    metadata_var: str | None,
    x_path: str | None,
    y_path: str | None,
    sample_ids_path: str | None,
    band_axis_path: str | None,
    metadata_path: str | None,
    label_file: str | Path | None,
    metadata_file: str | Path | None,
    join_key: str | None,
    folder_name_as_label: bool,
    file_name_as_label: bool,
    sample_file_pattern: str | None,
    sample_file_value_column: str | int | None,
    sample_file_band_column: str | int | None,
    label_alignment: str | None,
    allow_row_order_labels: bool,
    allow_generated_sample_ids_for_missing: bool,
    missing_sample_id_policy: str | None,
    missing_value_tokens: list[str] | None,
    delimiter: str | None,
    encoding: str | None,
    skiprows: int | None,
    header_row: int | None,
    header_rows: list[int] | None,
    data_start_row: int | None,
    data_end_row: int | None,
    spectral_start_column: str | int | None,
    spectral_end_column: str | int | None,
    band_axis_file: str | Path | None,
    band_unit: str | None,
    band_type: str | None,
    spectral_type: str | None,
    task_type: str | None,
    max_auto_columns: int,
    max_spectral_columns: int,
    wide_table_mode: str,
    confirm_wide_table: bool,
    confirm_read_plan: bool,
    auto_folder: bool,
) -> dict[str, Any]:
    file_previews = list(preview_result.get("file_previews") or [])
    first = file_previews[0] if file_previews else {}
    input_kind = preview_result.get("input_kind") or ("folder" if Path(input_path).is_dir() else "file")
    original_input_path = str(input_path)
    mixed_folder: dict[str, Any] = {}
    sample_files_requested = bool(
        folder_name_as_label
        or file_name_as_label
        or sample_file_pattern
        or sample_file_value_column is not None
        or sample_file_band_column is not None
    )
    if input_kind == "folder" and not sample_files_requested:
        folder_selection = _select_mixed_folder_files(
            input_path,
            preview_result,
            spectra_file=spectra_file,
            label_file=label_file,
            metadata_file=metadata_file,
            band_axis_file=band_axis_file,
        )
        if folder_selection.get("status") == "needs_confirmation":
            return folder_selection
        if folder_selection.get("status") == "ready":
            from .preview import preview_file

            mixed_folder = folder_selection
            input_path = folder_selection["spectra_file"]
            input_kind = "file"
            label_file = folder_selection.get("label_file") or label_file
            metadata_file = folder_selection.get("metadata_file") or metadata_file
            band_axis_file = folder_selection.get("band_axis_file") or band_axis_file
            selected_preview = preview_file(str(input_path), backend="core")
            if not selected_preview.get("ok"):
                return {
                    "status": "blocked",
                    "code": "SPECTRA_FILE_PREVIEW_FAILED",
                    "reason": "Selected spectra_file could not be previewed.",
                    "required_fields": [],
                    "suggested_arguments": {"--spectra-file": folder_selection.get("spectra_file")},
                }
            file_previews = list((selected_preview.get("result") or {}).get("file_previews") or [])
            first = file_previews[0] if file_previews else {}
            preview_result = selected_preview.get("result") or preview_result
    suffix = first.get("suffix") or Path(str(input_path)).suffix
    suffix = str(suffix).lower()
    workbook_suffixes = {".xlsx", ".xls", ".xlsm", ".ods"}
    container_suffixes = {".npy", ".npz", ".mat"}
    hierarchical_suffixes = {".h5", ".hdf5", ".nc"}
    is_workbook = suffix in workbook_suffixes
    is_container = suffix in container_suffixes
    is_hierarchical = suffix in hierarchical_suffixes
    workbook = first.get("workbook") or {}
    sheet_names = list(workbook.get("sheet_names") or [])
    selected_sheet = spectral_sheet or sheet
    if is_workbook and selected_sheet is not None and sheet_names and selected_sheet not in sheet_names:
        return {
            "status": "blocked",
            "code": "SHEET_NOT_FOUND",
            "reason": f"Workbook sheet was not found: {selected_sheet}",
            "required_fields": [],
            "suggested_arguments": {"--spectral-sheet": sheet_names, "--sheet-index": "zero-based sheet index"},
        }
    if is_workbook and selected_sheet is None and sheet_index is not None and 0 <= sheet_index < len(sheet_names):
        selected_sheet = sheet_names[sheet_index]
    if is_workbook and selected_sheet is None:
        if len(sheet_names) == 1:
            selected_sheet = sheet_names[0]
        elif len(sheet_names) > 1:
            return {
                "status": "needs_confirmation",
                "reason": "Workbook contains multiple candidate sheets; choose the spectral sheet.",
                "required_fields": ["spectral_sheet"],
                "suggested_arguments": {"--spectral-sheet": sheet_names, "--sheet-index": "zero-based sheet index"},
            }
    inferred_delimiter = delimiter if delimiter is not None else _first_value(first.get("delimiter_candidates"), "delimiter")
    inferred_header = header_row if header_row is not None else _first_value(first.get("header_row_candidates"), "row_index")
    if header_rows:
        inferred_header = min(header_rows)
    preamble = list(first.get("leading_preamble_candidates") or [])
    sample_id_ref: str | int | None = sample_id_column if sample_id_column is not None else sample_id_column_index
    explicit_table_columns = bool(sample_id_ref is not None or label_column or target_columns or metadata_columns or spectral_columns or spectral_start_column is not None or spectral_end_column is not None)
    if header_row is None and not header_rows and skiprows is None and not preamble and explicit_table_columns:
        inferred_header = 0
    inferred_skiprows = skiprows if skiprows is not None else 0 if delimiter is not None else len(preamble)
    orientation = sample_orientation
    if orientation is None:
        orientation = "rows"
    if input_kind == "folder":
        orientation = "one_file_per_sample"

    missing: list[str] = []
    suggested: dict[str, Any] = {}
    selected_sheet_preview = _sheet_preview(first, selected_sheet) if is_workbook else first
    evidence_source = selected_sheet_preview or first
    wide_table = _infer_wide_table_numeric_block(evidence_source) if str(wide_table_mode or "auto").lower() != "off" else None
    if wide_table and int(wide_table["n_spectral_features"]) > int(max_spectral_columns) and not confirm_wide_table:
        return {
            "status": "needs_confirmation",
            "reason": "A very wide spectral table was detected; confirm before reading all spectral columns.",
            "required_fields": ["confirm_wide_table"],
            "suggested_arguments": {
                "--confirm-wide-table": True,
                "--max-spectral-columns": max_spectral_columns,
                "--spectral-start-column": wide_table["spectral_start_column"],
                "--spectral-end-column": wide_table["spectral_end_column"],
            },
        }

    container_selection: dict[str, Any] = {}
    hierarchical_selection: dict[str, Any] = {}
    spectral_decision_source = "not_applicable"
    if label_column and target_columns:
        return {
            "status": "blocked",
            "code": "LABEL_TARGET_CONFLICT",
            "reason": "label_column and target_columns cannot both be used in one read without explicit conflict resolution.",
            "required_fields": [],
            "suggested_arguments": {"--label-column": label_column, "--target-columns": target_columns},
        }

    bundle_numeric_header = _bundle_numeric_header_columns(
        evidence_source,
        has_external_band_axis=band_axis_file is not None,
        has_external_sample_ids=sample_ids_file is not None,
        has_external_labels=label_file is not None,
        has_external_metadata=metadata_file is not None,
        orientation=orientation,
    )
    if bundle_numeric_header and header_row is None and not header_rows and skiprows is None:
        inferred_header = 0

    if is_container:
        container_plan = _container_selection(first, suffix, x_var=x_var, y_var=y_var, sample_ids_var=sample_ids_var, band_axis_var=band_axis_var, metadata_var=metadata_var)
        if container_plan["status"] != "ready":
            return container_plan
        container_selection = container_plan
        spectral = []
    elif is_hierarchical:
        hierarchical_plan = _hierarchical_selection(first, suffix, x_path=x_path, y_path=y_path, sample_ids_path=sample_ids_path, band_axis_path=band_axis_path, metadata_path=metadata_path)
        if hierarchical_plan["status"] != "ready":
            return hierarchical_plan
        hierarchical_selection = hierarchical_plan
        spectral = []
    elif input_kind != "folder" and orientation == "rows":
        spectral = spectral_columns or []
        spectral_decision_source = "user_specified" if spectral_columns or spectral_start_column is not None or spectral_end_column is not None else "auto_inferred"
        if not spectral and spectral_start_column is not None and spectral_end_column is not None:
            spectral = []
        elif not spectral and wide_table:
            spectral_start_column = wide_table["spectral_start_column"]
            spectral_end_column = wide_table["spectral_end_column"]
            spectral_decision_source = "auto_inferred_wide_table_numeric_header"
        elif not spectral and bundle_numeric_header:
            spectral = bundle_numeric_header
        elif not spectral:
            spectral = _names(evidence_source.get("band_like_column_evidence"))
        if not spectral and label_file:
            spectral = _numeric_header_columns(evidence_source)
        if not spectral and band_axis_file:
            excluded = {str(value) for value in [sample_id_ref, label_column] if value is not None}
            excluded.update(str(value) for value in (target_columns or []))
            excluded.update(str(value) for value in (metadata_columns or []))
            spectral = [name for name in _names(evidence_source.get("numeric_column_summary")) if name not in excluded]
        if not spectral and not (spectral_start_column is not None and spectral_end_column is not None):
            missing.append("spectral_columns")
            suggested["--spectral-columns"] = "comma-separated spectral column names or use --spectral-start-column and --spectral-end-column"
    else:
        spectral = spectral_columns or []
        spectral_decision_source = "user_specified" if spectral_columns else "not_applicable"

    sample_id = sample_id_ref if sample_id_ref is not None else _first_name(evidence_source.get("sample_id_like_column_evidence"))
    if sample_id is None and wide_table and wide_table.get("sample_id_column_index") is not None:
        sample_id = int(wide_table["sample_id_column_index"])
    labels = target_columns or ([label_column] if label_column else [])
    if is_container and container_selection.get("y_var"):
        labels = [container_selection["y_var"]]
    if is_hierarchical and hierarchical_selection.get("y_path"):
        labels = [hierarchical_selection["y_path"]]
    if not labels and not label_file and input_kind != "folder":
        inferred_label = _first_name(evidence_source.get("label_like_column_evidence"))
        if inferred_label is None and wide_table and wide_table.get("label_column"):
            inferred_label = str(wide_table["label_column"])
        labels = [inferred_label] if inferred_label else []
    metadata = metadata_columns if metadata_columns is not None else _names(evidence_source.get("metadata_like_column_evidence"))
    spectral_metadata = [] if label_sheet else metadata

    if input_kind == "folder":
        sample_file_columns = {
            "band_axis": sample_file_band_column if sample_file_band_column is not None else band_axis_column or "wavelength",
            "value": sample_file_value_column if sample_file_value_column is not None else spectral_columns[0] if spectral_columns else "intensity",
        }
        if band_axis_column is None and sample_file_band_column is None:
            suggested["--band-axis-column"] = "band column in each sample file, default wavelength"
        if not spectral_columns and sample_file_value_column is None:
            suggested["--spectral-columns"] = "value column in each sample file, default intensity"
    elif orientation == "columns":
        samples_evidence = evidence_source.get("samples_as_columns_evidence") or {}
        axis_column = band_axis_column if band_axis_column is not None else samples_evidence.get("band_axis_column_candidate")
        if axis_column is None:
            missing.append("band_axis_column")
            suggested["--band-axis-column"] = "column containing the band axis"
        sample_start = spectral_columns[0] if spectral_columns else 1
    else:
        sample_file_columns = {}
        axis_column = None
        sample_start = None

    if missing:
        return {
            "status": "needs_confirmation",
            "reason": "The input can be previewed, but key read semantics are missing.",
            "required_fields": missing,
            "suggested_arguments": suggested,
        }

    task_hint = task_type or ("classification" if labels or label_file or folder_name_as_label or file_name_as_label else "unsupervised")
    label_role = {"source": "external_file" if label_file else "read_plan" if labels else None, "column": labels[0] if labels and not label_file else None, "required": bool(labels or label_file), "status": "confirmed" if labels or label_file else "not_applicable", "evidence": []}
    target_role = {"source": None, "column": None, "required": False, "status": "not_applicable", "evidence": []}
    if target_columns:
        task_hint = task_type or ("multi_target_regression" if len(target_columns) > 1 else "regression")
        target_role = {"source": "external_file" if label_file else "read_plan", "column": target_columns[0], "columns": target_columns, "required": True, "status": "confirmed", "evidence": []}
        label_role = {"source": None, "column": None, "required": False, "status": "not_applicable", "evidence": []}

    plan: dict[str, Any] = {
        "read_plan_id": "transient_one_shot",
        "read_plan_version": "0.1.0",
        "read_plan_status": "confirmed",
        "created_by": "tool",
        "created_by_type": "tool",
        "source_preview_ref": None,
        "source_path": str(input_path),
        "source_base_dir": source_base_dir,
        "input_kind": "folder" if mixed_folder else input_kind,
        "file_type": "folder" if input_kind == "folder" else suffix,
        "encoding": encoding or first.get("encoding_hint") or "utf-8-sig",
        "delimiter": delimiter if input_kind == "folder" else inferred_delimiter,
        "skiprows": inferred_skiprows,
        "header_row": inferred_header if inferred_header is not None else 0,
        "header_rows": header_rows or [],
        "data_start_row": data_start_row,
        "data_end_row": data_end_row,
        "read_mode": "sample_files_folder" if input_kind == "folder" else "matrix_file",
        "sample_orientation": orientation,
        "sample_id": {"source": "read_plan" if sample_id is not None else "generated", "column": sample_id, "required": False, "status": "confirmed", "evidence": []},
        "missing_sample_id_policy": missing_sample_id_policy or ("generate" if allow_generated_sample_ids_for_missing else None),
        "missing_value_tokens": missing_value_tokens or [],
        "label": label_role,
        "target": target_role,
        "metadata": {"columns": spectral_metadata, "source": "read_plan"},
        "spectral_columns": {"columns": spectral, "start": spectral_start_column, "end": spectral_end_column, "source": "read_plan"},
        "band_axis": {"source": "band_axis_column" if orientation == "columns" else "generated_index" if (is_container or is_hierarchical) else "column_headers", "required": False},
        "band_unit": band_unit or (wide_table or {}).get("band_unit") or _band_unit(evidence_source.get("band_like_column_evidence")),
        "band_type": band_type or (wide_table or {}).get("band_type"),
        "spectral_type": spectral_type,
        "task_hint": task_hint,
        "preview_evidence": {"wide_table_numeric_block": wide_table} if wide_table else {},
        "decision_evidence": [],
        "selected_fragments": [],
        "selected_references": [],
        "required_confirmations": [],
        "recommended_confirmations": [],
        "confirmed_items": [],
        "unresolved_items": [],
        "execution_intent": {"tool": "apply_read_plan", "write_package": False, "validation_level": "minimal", "expected_backend": None},
        "expected_outputs": ["X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"],
        "blocked_reasons": [],
        "downstream_readiness_hint": {"apply_read_plan": True},
    }
    plan["table_layout"] = {
        "header_rows": header_rows or [],
        "header_row": inferred_header if inferred_header is not None else 0,
        "skiprows": inferred_skiprows,
        "data_start_row": data_start_row,
        "data_end_row": data_end_row,
        "spectral_start_column": spectral_start_column,
        "spectral_end_column": spectral_end_column,
        "spectral_columns": spectral,
        "spectral_columns_decision_source": spectral_decision_source,
        "metadata_columns": spectral_metadata,
        "label_column": label_column,
        "target_columns": target_columns or [],
        "sample_id_column": sample_id,
        "sample_id_column_index": sample_id_column_index,
        "wide_table_mode": wide_table_mode,
        "max_auto_columns": max_auto_columns,
        "max_spectral_columns": max_spectral_columns,
    }
    if mixed_folder:
        plan["mixed_folder"] = {
            "input": original_input_path,
            "spectra_file": str(mixed_folder.get("spectra_file")),
            "label_file": str(label_file) if label_file else None,
            "metadata_file": str(metadata_file) if metadata_file else None,
            "band_axis_file": str(band_axis_file) if band_axis_file else None,
        }
        plan["source_path"] = str(input_path)
        plan["input_kind"] = "file"
        plan["file_type"] = suffix
    if band_axis_file:
        plan["band_axis_file"] = {
            "path": str(band_axis_file),
            "file_type": Path(str(band_axis_file)).suffix.lower(),
            "column": band_axis_column,
            "required": True,
        }
        plan["band_axis"] = {"source": "external_file", "column": band_axis_column, "required": True}
        plan["band_axis_source"] = {"mode": "external_file", "file": str(band_axis_file), "column": band_axis_column, "unit": plan.get("band_unit"), "type": band_type or "external_file"}
    else:
        plan["band_axis_source"] = {"mode": plan["band_axis"].get("source"), "file": None, "column": band_axis_column, "unit": plan.get("band_unit"), "type": band_type or plan["band_axis"].get("source")}
    if sample_ids_file:
        plan["sample_ids_file"] = {
            "path": str(sample_ids_file),
            "file_type": Path(str(sample_ids_file)).suffix.lower(),
            "column": sample_id_ref,
            "required": True,
        }
        plan["sample_id"] = {"source": "external_file", "column": sample_id_ref, "required": True, "status": "confirmed", "evidence": []}
    if target_columns:
        plan["multi_target"] = {"enabled": len(target_columns) > 1, "target_columns": target_columns}
    if is_workbook:
        engine = _workbook_engine(suffix)
        plan["workbook"] = {
            "path": str(input_path),
            "file_type": suffix.lstrip("."),
            "engine": engine,
            "spectral_sheet": selected_sheet,
            "label_sheet": label_sheet,
            "selected_sheet": selected_sheet,
            "sheet_index": sheet_index,
        }
        plan["sheet_name"] = selected_sheet
        plan["sheet"] = {
            "name": selected_sheet,
            "index": sheet_index,
            "skiprows": inferred_skiprows,
            "header_row": inferred_header if inferred_header is not None else 0,
            "sample_orientation": orientation,
            "sample_id_column": sample_id,
            "label_column": label_column,
            "target_columns": target_columns or [],
            "metadata_columns": spectral_metadata,
            "spectral_columns": spectral,
            "band_axis_column": band_axis_column,
        }
        if label_sheet:
            plan["label_file"] = {
                "path": str(input_path),
                "file_type": suffix,
                "sheet_name": label_sheet,
                "sheet_index": None,
                "sample_id_column": join_key or sample_id or "sample_id",
                "label_column": label_column,
                "target_columns": target_columns or [],
                "metadata_columns": metadata_columns or [],
                "required": True,
            }
            plan["alignment_plan"] = {
                "join_key": join_key or sample_id or "sample_id",
                "left_source": "spectra",
                "right_source": "label_sheet",
                "join_type": "left",
                "preserve_spectrum_order": True,
                "allow_unmatched_spectra": False,
                "allow_unmatched_labels": True,
                "duplicate_policy": "blocked",
                "missing_label_policy": "blocked",
            }
            plan["label"] = {"source": "external_file", "column": None, "required": True, "status": "confirmed", "evidence": []}
            plan["task_hint"] = task_type or ("regression" if target_columns else "classification")
    if is_container:
        var_map = {
            "X": container_selection.get("x_var"),
            "y": container_selection.get("y_var"),
            "sample_ids": container_selection.get("sample_ids_var"),
            "band_axis": container_selection.get("band_axis_var"),
            "metadata": container_selection.get("metadata_var"),
        }
        plan["container"] = {
            "path": str(input_path),
            "file_type": suffix.lstrip("."),
            "x_var": var_map["X"],
            "y_var": var_map["y"],
            "sample_ids_var": var_map["sample_ids"],
            "band_axis_var": var_map["band_axis"],
            "metadata_var": var_map["metadata"],
            "sample_orientation": orientation,
            "allow_generated_sample_ids": True,
            "allow_generated_band_axis": True,
        }
        plan["variable_map"] = var_map
        plan["spectral_columns"] = {"columns": [], "source": "container_variable"}
        plan["sample_id"] = {"source": "container_variable" if var_map["sample_ids"] else "generated", "column": var_map["sample_ids"], "required": False, "status": "confirmed", "evidence": []}
        plan["band_axis"] = {"source": "container_variable" if var_map["band_axis"] else "generated_index", "variable": var_map["band_axis"], "required": False}
        if var_map["y"]:
            plan["label"] = {"source": "container_variable", "column": var_map["y"], "required": True, "status": "confirmed", "evidence": []}
            plan["task_hint"] = task_type or "classification"
        if var_map["metadata"]:
            plan["metadata"] = {"columns": [], "source": "container_variable", "variable": var_map["metadata"]}
    if is_hierarchical:
        dataset_map = {
            "X": hierarchical_selection.get("x_path"),
            "y": hierarchical_selection.get("y_path"),
            "sample_ids": hierarchical_selection.get("sample_ids_path"),
            "band_axis": hierarchical_selection.get("band_axis_path"),
            "metadata": hierarchical_selection.get("metadata_path"),
        }
        source_type = "netcdf" if suffix == ".nc" else "hdf5"
        plan["hierarchical_container"] = {
            "path": str(input_path),
            "file_type": source_type,
            "x_path": dataset_map["X"],
            "y_path": dataset_map["y"],
            "sample_ids_path": dataset_map["sample_ids"],
            "band_axis_path": dataset_map["band_axis"],
            "metadata_path": dataset_map["metadata"],
            "sample_orientation": orientation,
            "allow_generated_sample_ids": True,
            "allow_generated_band_axis": True,
        }
        plan["dataset_map"] = dataset_map
        plan["spectral_columns"] = {"columns": [], "source": "hierarchical_dataset"}
        plan["sample_id"] = {"source": "dataset_path" if dataset_map["sample_ids"] else "generated", "column": dataset_map["sample_ids"], "required": False, "status": "confirmed", "evidence": []}
        plan["band_axis"] = {"source": "dataset_path" if dataset_map["band_axis"] else "generated_index", "path": dataset_map["band_axis"], "required": False}
        if dataset_map["y"]:
            plan["label"] = {"source": "dataset_path", "column": dataset_map["y"], "required": True, "status": "confirmed", "evidence": []}
            plan["task_hint"] = task_type or "classification"
        if dataset_map["metadata"]:
            plan["metadata"] = {"columns": [], "source": "dataset_path", "path": dataset_map["metadata"]}
    if input_kind == "folder":
        plan.update({
            "sample_file_pattern": sample_file_pattern or "*.*",
            "sample_file_recursive": True,
            "sample_file_columns": sample_file_columns,
            "sample_file_value_column": sample_file_columns["value"],
            "sample_file_band_axis_column": sample_file_columns["band_axis"],
            "file_name_as_sample_id": True,
            "folder_name_as_label": bool(folder_name_as_label),
            "file_name_as_label": bool(file_name_as_label),
            "file_label_rule": "prefix_before_underscore" if file_name_as_label else None,
            "band_axis": {"source": "per_file_column", "column": sample_file_columns["band_axis"], "required": False},
            "metadata": {"columns": ["source_file", "relative_path", "source_folder"], "source": "file_inventory"},
        })
        if folder_name_as_label:
            plan["label"] = {"source": "folder_name", "column": None, "required": True, "status": "confirmed", "evidence": []}
            plan["task_hint"] = task_type or "classification"
        if file_name_as_label:
            plan["label"] = {"source": "file_name", "column": None, "required": True, "status": "confirmed", "evidence": []}
            plan["task_hint"] = task_type or "classification"
    if orientation == "columns":
        plan["samples_as_columns"] = {
            "enabled": True,
            "band_axis_column": axis_column,
            "sample_id_source": "column_headers",
            "sample_id_row": plan["header_row"],
            "metadata_rows": [],
            "data_start_row": int(plan["header_row"]) + 1,
            "sample_start_column": sample_start,
            "transpose_required": True,
        }
        plan["band_axis"] = {"source": "band_axis_column", "required": False}
    if label_file and not label_sheet:
        label_suffix = Path(str(label_file)).suffix.lower()
        row_order_requested = label_alignment == "row_order" or allow_row_order_labels
        label_join_key = None if row_order_requested else join_key or sample_id or "sample_id"
        plan["label_file"] = {
            "path": str(label_file),
            "file_type": label_suffix,
            "encoding": encoding or "utf-8-sig",
            "delimiter": delimiter or ("\t" if label_suffix in {".tsv", ".txt"} else ","),
            "skiprows": 0,
            "header_row": 0,
            "sample_id_column": label_join_key,
            "label_column": label_column,
            "target_columns": target_columns or [],
            "metadata_columns": [],
            "required": True,
        }
        plan["alignment_plan"] = {
            "method": "row_order" if row_order_requested else label_alignment or "sample_id",
            "join_key": label_join_key,
            "left_source": "spectra",
            "right_source": "label_file",
            "join_type": "left",
            "preserve_spectrum_order": True,
            "allow_unmatched_spectra": False,
            "allow_unmatched_labels": True,
            "duplicate_policy": "blocked",
            "missing_label_policy": "blocked",
        }
        plan["label_alignment"] = "row_order" if row_order_requested else label_alignment or "sample_id"
        plan["allow_row_order_labels"] = bool(allow_row_order_labels)
    elif not label_sheet:
        plan["label_file"] = {}
        plan["alignment_plan"] = {}
    if metadata_file:
        metadata_suffix = Path(str(metadata_file)).suffix.lower()
        plan["metadata_file"] = {
            "path": str(metadata_file),
            "file_type": metadata_suffix,
            "encoding": encoding or "utf-8-sig",
            "delimiter": delimiter or ("\t" if metadata_suffix in {".tsv", ".txt"} else ","),
            "skiprows": 0,
            "header_row": 0,
            "sample_id_column": join_key or sample_id or "sample_id",
            "metadata_columns": metadata_columns or [],
            "required": True,
        }
    _attach_reader_confirmation(
        plan,
        confirm_read_plan=confirm_read_plan,
        sample_orientation=sample_orientation,
        sample_id_column=sample_id_column,
        sample_id_column_index=sample_id_column_index,
        label_column=label_column,
        target_columns=target_columns,
        spectral_columns=spectral_columns,
        spectral_start_column=spectral_start_column,
        spectral_end_column=spectral_end_column,
        x_var=x_var,
        x_path=x_path,
        band_unit=band_unit,
        band_type=band_type,
        spectral_type=spectral_type,
        task_type=task_type,
    )
    return {"status": "ready", "read_plan": plan}


def _attach_reader_confirmation(
    plan: dict[str, Any],
    *,
    confirm_read_plan: bool,
    sample_orientation: str | None,
    sample_id_column: str | None,
    sample_id_column_index: int | None,
    label_column: str | None,
    target_columns: list[str] | None,
    spectral_columns: list[str] | None,
    spectral_start_column: str | int | None,
    spectral_end_column: str | int | None,
    x_var: str | None,
    x_path: str | None,
    band_unit: str | None,
    band_type: str | None,
    spectral_type: str | None,
    task_type: str | None,
) -> None:
    explicit_sample_id = sample_id_column is not None or sample_id_column_index is not None
    explicit_label = label_column is not None or bool(target_columns) or (task_type in {"unsupervised", "exploratory"})
    explicit_spectral = bool(spectral_columns) or (spectral_start_column is not None and spectral_end_column is not None) or bool(x_var) or bool(x_path)
    explicit_band_axis = bool(band_unit) and bool(band_type or spectral_type)
    explicit_core_plan = bool(sample_orientation) and explicit_sample_id and explicit_label and explicit_spectral and explicit_band_axis and bool(task_type)
    source = "confirm_read_plan_argument" if confirm_read_plan else "explicit_cli_arguments" if explicit_core_plan else "implicit_reader_plan"
    if confirm_read_plan or explicit_core_plan:
        plan["read_plan_status"] = "confirmed"
        plan["required_confirmations"] = []
        plan["unresolved_items"] = []
        confirmed_items = set(plan.get("confirmed_items") or [])
        confirmed_items.update(
            [
                "read_plan",
                "sample_orientation",
                "sample_id_column",
                "label_or_target_column",
                "spectral_columns",
                "band_axis_semantics",
                "task_type",
            ]
        )
        plan["confirmed_items"] = sorted(confirmed_items)
        execution_intent = dict(plan.get("execution_intent") or {})
        execution_intent["apply_read_plan"] = True
        execution_intent["write_package"] = True
        plan["execution_intent"] = execution_intent
    plan["reader_confirmation"] = {
        "status": "confirmed" if confirm_read_plan or explicit_core_plan or plan.get("read_plan_status") == "confirmed" else "provisional",
        "source": source,
        "sample_orientation": sample_orientation or plan.get("sample_orientation"),
        "sample_id_column": sample_id_column if sample_id_column is not None else sample_id_column_index,
        "label_column": label_column,
        "target_columns": target_columns or [],
        "spectral_columns": _spectral_confirmation_summary(spectral_columns, spectral_start_column, spectral_end_column, x_var, x_path),
        "band_unit": band_unit or plan.get("band_unit"),
        "band_type": band_type or plan.get("band_type"),
        "spectral_type": spectral_type or plan.get("spectral_type"),
        "task_type": task_type or plan.get("task_hint"),
    }


def _spectral_confirmation_summary(
    spectral_columns: list[str] | None,
    spectral_start_column: str | int | None,
    spectral_end_column: str | int | None,
    x_var: str | None,
    x_path: str | None,
) -> str | list[str] | None:
    if spectral_columns:
        return spectral_columns
    if spectral_start_column is not None and spectral_end_column is not None:
        return f"{spectral_start_column}..{spectral_end_column}"
    if x_var:
        return x_var
    if x_path:
        return x_path
    return None


def _compact_ready_result(result: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "status",
        "output_dir",
        "X",
        "y",
        "sample_ids",
        "band_axis",
        "metadata",
        "data_contract",
        "n_samples",
        "n_features",
        "label_status",
        "task_hint",
        "band_unit",
        "source_type",
        "warnings",
    }
    compact = {key: result.get(key) for key in keep if key in result}
    compact["status"] = compact.get("status") or "ready"
    compact["warnings"] = compact.get("warnings") or []
    return compact


def _select_mixed_folder_files(
    input_path: str | Path,
    preview_result: dict[str, Any],
    *,
    spectra_file: str | Path | None,
    label_file: str | Path | None,
    metadata_file: str | Path | None,
    band_axis_file: str | Path | None,
) -> dict[str, Any]:
    base = Path(input_path)
    roles = preview_result.get("folder_role_candidates") or {}
    spectra_candidates = _candidate_paths(roles.get("spectra"))
    label_candidates = _candidate_paths(roles.get("label"))
    metadata_candidates = _candidate_paths(roles.get("metadata"))
    band_axis_candidates = _candidate_paths(roles.get("band_axis"))

    selected_spectra = _resolve_folder_role_path(base, spectra_file) if spectra_file else _unique_or_none(spectra_candidates)
    selected_label = _resolve_folder_role_path(base, label_file) if label_file else _unique_or_none(label_candidates)
    selected_metadata = _resolve_folder_role_path(base, metadata_file) if metadata_file else _unique_or_none(metadata_candidates)
    selected_band_axis = _resolve_folder_role_path(base, band_axis_file) if band_axis_file else _unique_or_none(band_axis_candidates)

    if spectra_file is None and len(spectra_candidates) > 1:
        return {
            "status": "needs_confirmation",
            "reason": "Folder contains multiple candidate spectra files; choose one.",
            "required_fields": ["spectra_file"],
            "suggested_arguments": {"--spectra-file": [str(path) for path in spectra_candidates]},
        }
    if spectra_file is None and selected_spectra is None:
        return {"status": "not_mixed_folder"}
    if label_file is None and len(label_candidates) > 1:
        return {
            "status": "needs_confirmation",
            "reason": "Folder contains multiple candidate label files; choose one.",
            "required_fields": ["label_file"],
            "suggested_arguments": {"--label-file": [str(path) for path in label_candidates], "--spectra-file": str(selected_spectra) if selected_spectra else [str(path) for path in spectra_candidates]},
        }
    return {
        "status": "ready",
        "spectra_file": selected_spectra,
        "label_file": selected_label,
        "metadata_file": selected_metadata,
        "band_axis_file": selected_band_axis,
    }


def _candidate_paths(items: Any) -> list[Path]:
    paths: list[Path] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("path"):
            path = Path(str(item["path"]))
            if path not in paths:
                paths.append(path)
    return paths


def _unique_or_none(paths: list[Path]) -> Path | None:
    return paths[0] if len(paths) == 1 else None


def _resolve_folder_role_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item.get("name")) for item in items if isinstance(item, dict) and item.get("name")]


def _first_name(items: Any) -> str | None:
    values = _names(items)
    return values[0] if values else None


def _first_value(items: Any, key: str) -> Any:
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0].get(key)
    return None


def _infer_wide_table_numeric_block(evidence: dict[str, Any], *, min_spectral_columns: int = 50) -> dict[str, Any] | None:
    preview = evidence.get("column_preview") or []
    header = [str(item.get("name", "") if isinstance(item, dict) else item).strip() for item in preview]
    if len(header) < min_spectral_columns + 1:
        return None
    runs: list[tuple[int, int, list[float]]] = []
    start: int | None = None
    values: list[float] = []
    for idx, name in enumerate(header):
        value = _float_header(name)
        if value is None:
            if start is not None and len(values) >= min_spectral_columns:
                runs.append((start, idx - 1, values))
            start = None
            values = []
            continue
        if start is None:
            start = idx
            values = [value]
        else:
            values.append(value)
    if start is not None and len(values) >= min_spectral_columns:
        runs.append((start, len(header) - 1, values))
    monotonic_runs = [
        (run_start, run_end, run_values)
        for run_start, run_end, run_values in runs
        if _is_monotonic(run_values)
    ]
    if not monotonic_runs:
        return None
    run_start, run_end, run_values = max(monotonic_runs, key=lambda item: len(item[2]))
    label_column = _wide_table_label_column(header, run_end)
    sample_id_index = 0 if run_start == 1 and _is_empty_or_sample_id_header(header[0]) else None
    max_value = max(run_values)
    min_value = min(run_values)
    decreasing = all(a > b for a, b in zip(run_values, run_values[1:]))
    band_type = "wavenumber" if decreasing and 100 <= min_value <= max_value <= 10000 else "wavelength" if 100 <= min_value <= max_value <= 20000 else None
    band_unit = "cm-1" if band_type == "wavenumber" else "nm" if band_type == "wavelength" else None
    return {
        "status": "detected",
        "reason": "contiguous_monotonic_numeric_header_block",
        "spectral_start_column": header[run_start],
        "spectral_end_column": header[run_end],
        "spectral_start_index": run_start,
        "spectral_end_index": run_end,
        "n_spectral_features": len(run_values),
        "monotonic": "decreasing" if decreasing else "increasing",
        "sample_id_column_index": sample_id_index,
        "label_column": label_column,
        "band_type": band_type,
        "band_unit": band_unit,
    }


def _float_header(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return all(a < b for a, b in zip(values, values[1:])) or all(a > b for a, b in zip(values, values[1:]))


def _is_empty_or_sample_id_header(value: str) -> bool:
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized == "" or normalized in {"sample id", "sampleid", "sample", "id", "index", "unnamed: 0"}


def _wide_table_label_column(header: list[str], spectral_end_index: int) -> str | None:
    for name in header[spectral_end_index + 1 :]:
        normalized = str(name).strip().lower().replace("_", " ")
        if normalized in {"label", "class", "target", "category", "group", "type"}:
            return name
    return None


def _numeric_header_columns(evidence: dict[str, Any]) -> list[str]:
    rows = evidence.get("row_preview") or []
    if len(rows) < 2:
        return []
    header = [str(value).strip() for value in rows[0]]
    if len(header) < 2 or any(not value for value in header):
        return []
    try:
        [float(value) for value in header]
    except ValueError:
        return []
    return header


def _bundle_numeric_header_columns(
    evidence: dict[str, Any],
    *,
    has_external_band_axis: bool,
    has_external_sample_ids: bool,
    has_external_labels: bool,
    has_external_metadata: bool,
    orientation: str | None,
) -> list[str]:
    if orientation != "rows":
        return []
    if not (has_external_band_axis and (has_external_sample_ids or has_external_labels or has_external_metadata)):
        return []
    raw_lines = evidence.get("raw_head_lines") or []
    if len(raw_lines) < 2:
        return []
    delimiter = _first_value(evidence.get("delimiter_candidates"), "delimiter") or ","
    header = [str(value).strip() for value in raw_lines[0].split(delimiter)]
    if len(header) < 2 or any(not value for value in header):
        return []
    try:
        [float(value) for value in header]
    except ValueError:
        return []
    second_row = [str(value).strip() for value in raw_lines[1].split(delimiter)]
    if len(second_row) != len(header):
        return []
    numeric_cells = 0
    observed_cells = 0
    for value in second_row:
        if value == "":
            continue
        observed_cells += 1
        try:
            float(value)
            numeric_cells += 1
        except ValueError:
            if value.lower() in {"na", "n/a", "nan", "null", "none", "missing", "--", "-", "."}:
                numeric_cells += 1
    if observed_cells == 0 or numeric_cells != observed_cells:
        return []
    return header


def _band_unit(items: Any) -> str | None:
    units = {item.get("unit_hint") for item in items or [] if isinstance(item, dict) and item.get("unit_hint")}
    if "cm-1" in units:
        return "cm-1"
    if "nm" in units:
        return "nm"
    return None


def _container_selection(
    preview: dict[str, Any],
    suffix: str,
    *,
    x_var: str | None,
    y_var: str | None,
    sample_ids_var: str | None,
    band_axis_var: str | None,
    metadata_var: str | None,
) -> dict[str, Any]:
    if suffix == ".npy":
        array = preview.get("array") or {}
        if int(array.get("ndim") or len(array.get("shape") or [])) != 2 or not array.get("numeric", False):
            return {"status": "blocked", "code": "NPY_ARRAY_NOT_2D", "reason": "NPY input must contain one 2D numeric array.", "required_fields": [], "suggested_arguments": {}}
        return {"status": "ready", "x_var": x_var or "__npy_array__", "y_var": y_var, "sample_ids_var": sample_ids_var, "band_axis_var": band_axis_var, "metadata_var": metadata_var}

    candidates = preview.get("container_candidates") or {}
    array_inventory = preview.get("array_inventory") or preview.get("variable_inventory") or []
    names = [str(item.get("name")) for item in array_inventory if item.get("name")]
    if x_var:
        if names and x_var not in names:
            return {"status": "blocked", "code": "X_VAR_NOT_FOUND", "reason": f"X variable was not found: {x_var}", "required_fields": [], "suggested_arguments": {"--x-var": names}}
        selected_x = x_var
    else:
        x_candidates = list(candidates.get("X") or [])
        if len(x_candidates) == 1:
            selected_x = x_candidates[0]
        elif len(x_candidates) > 1:
            return {
                "status": "needs_confirmation",
                "reason": "Container has multiple 2D numeric X candidates; choose one.",
                "required_fields": ["x_var"],
                "suggested_arguments": {"--x-var": x_candidates},
            }
        else:
            return {"status": "blocked", "code": "X_VAR_NOT_FOUND", "reason": "No 2D numeric X variable was found.", "required_fields": ["x_var"], "suggested_arguments": {"--x-var": names}}

    return {
        "status": "ready",
        "x_var": selected_x,
        "y_var": y_var or _candidate_by_name(candidates.get("y") or [], ["y", "label", "class"]),
        "sample_ids_var": sample_ids_var or _candidate_by_name(candidates.get("sample_ids") or [], ["sample_ids", "sample_id", "ids"]),
        "band_axis_var": band_axis_var or _candidate_by_name(candidates.get("band_axis") or [], ["band_axis", "wavelength", "wavenumber", "bands"]),
        "metadata_var": metadata_var,
    }


def _hierarchical_selection(
    preview: dict[str, Any],
    suffix: str,
    *,
    x_path: str | None,
    y_path: str | None,
    sample_ids_path: str | None,
    band_axis_path: str | None,
    metadata_path: str | None,
) -> dict[str, Any]:
    candidates = preview.get("dataset_candidates") or {}
    inventory = preview.get("dataset_inventory") or preview.get("variable_inventory") or []
    names = [str(item.get("name") or item.get("path")) for item in inventory if item.get("name") or item.get("path")]
    if x_path:
        if names and x_path not in names:
            code = "VARIABLE_PATH_NOT_FOUND" if suffix == ".nc" else "DATASET_PATH_NOT_FOUND"
            return {"status": "blocked", "code": code, "reason": f"X dataset path was not found: {x_path}", "required_fields": [], "suggested_arguments": {"--x-path": names}}
        selected_x = x_path
    else:
        x_candidates = list(candidates.get("X") or [])
        if len(x_candidates) == 1:
            selected_x = x_candidates[0]
        elif len(x_candidates) > 1:
            return {
                "status": "needs_confirmation",
                "reason": "Hierarchical container has multiple 2D numeric X candidates; choose one.",
                "required_fields": ["x_path"],
                "suggested_arguments": {"--x-path": x_candidates},
            }
        else:
            return {"status": "blocked", "code": "X_DATASET_NOT_FOUND", "reason": "No 2D numeric X dataset was found.", "required_fields": ["x_path"], "suggested_arguments": {"--x-path": names}}

    return {
        "status": "ready",
        "x_path": selected_x,
        "y_path": y_path or _candidate_by_name(candidates.get("y") or [], ["y", "label", "class"]),
        "sample_ids_path": sample_ids_path or _candidate_by_name(candidates.get("sample_ids") or [], ["sample_ids", "sample_id", "ids"]),
        "band_axis_path": band_axis_path or _candidate_by_name(candidates.get("band_axis") or [], ["band_axis", "wavelength", "wavenumber", "bands"]),
        "metadata_path": metadata_path,
    }


def _candidate_by_name(candidates: list[str], preferred: list[str]) -> str | None:
    lowered = {str(item).lower(): str(item) for item in candidates}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    return str(candidates[0]) if len(candidates) == 1 else None


def _sheet_preview(file_preview: dict[str, Any], sheet_name: str | None) -> dict[str, Any]:
    if not sheet_name:
        return {}
    for item in file_preview.get("sheet_previews") or []:
        if item.get("sheet_name") == sheet_name:
            return item
    return {}


def _workbook_engine(suffix: str) -> str:
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    if suffix == ".ods":
        return "odf"
    return "unknown"

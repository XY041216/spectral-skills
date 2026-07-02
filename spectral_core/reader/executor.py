"""Confirmed read_plan executor for CSV/TSV/TXT reader tables."""

from __future__ import annotations

import csv
import fnmatch
import importlib.util
import math
import re
from pathlib import Path
from typing import Any

from .io_utils import resolve_read_plan_source_path
from .package_writer import write_apply_outputs
from .post_read_audit import PostReadAuditBlocked, audit_post_read
from .read_plan import load_json_document, unwrap_response
from .response import error_response, ok_response
from .validator import validate_read_plan_for_execution


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt"}
WORKBOOK_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".ods"}
CONTAINER_SUFFIXES = {".npy", ".npz", ".mat"}
HIERARCHICAL_SUFFIXES = {".h5", ".hdf5", ".nc"}


def apply_read_plan(
    read_plan: dict[str, Any] | str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    """Execute a confirmed read_plan without inferring or repairing semantics."""

    if read_plan is None:
        return _blocked("No read_plan was provided.", "READ_PLAN_MISSING", backend=backend)
    read_plan_path = Path(read_plan) if isinstance(read_plan, (str, Path)) else None
    try:
        plan = unwrap_response(load_json_document(read_plan))
    except Exception as exc:
        return _blocked(f"Could not load read_plan: {exc}", "READ_PLAN_LOAD_FAILED", backend=backend)

    execution_check = validate_read_plan_for_execution(plan, backend=backend)
    if not execution_check.get("ok"):
        result = _base_result(plan, output_dir, "blocked")
        result.update(execution_check.get("result") or {})
        return error_response(
            "apply_read_plan",
            "read_plan is not executable.",
            backend=backend,
            code="READ_PLAN_NOT_EXECUTABLE",
            result=result,
            details={"validation": execution_check.get("result", {})},
        )

    path_resolution = resolve_read_plan_source_path(plan, read_plan_path=read_plan_path, invocation_cwd=Path.cwd())
    resolved_value = path_resolution.get("resolved_path")
    source_path = Path(str(resolved_value)) if resolved_value else Path(str(plan.get("source_path") or ""))
    if not source_path.exists():
        return _blocked("source_path does not exist.", "SOURCE_PATH_NOT_FOUND", backend=backend, plan=plan, output_dir=output_dir, **path_resolution, cwd=str(Path.cwd()), read_plan_dir=str(read_plan_path.parent) if read_plan_path else None, source_base_dir=plan.get("source_base_dir"))

    is_workbook = source_path.suffix.lower() in WORKBOOK_SUFFIXES
    is_container = source_path.suffix.lower() in CONTAINER_SUFFIXES
    is_hierarchical = source_path.suffix.lower() in HIERARCHICAL_SUFFIXES
    if plan.get("read_mode") != "sample_files_folder":
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES | WORKBOOK_SUFFIXES | CONTAINER_SUFFIXES | HIERARCHICAL_SUFFIXES:
            return _blocked("Only CSV, TSV, TXT, Excel, ODS, NPY, NPZ, MAT, HDF5, and NetCDF table/container files are supported.", "UNSUPPORTED_FILE_TYPE", backend=backend, plan=plan, output_dir=output_dir, suffix=suffix, **path_resolution)

    if output_dir is None:
        return _blocked("--output-dir is required for apply_read_plan.", "OUTPUT_DIR_REQUIRED", backend=backend, plan=plan)
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    if output_root.exists() and any(output_root.iterdir()) and not overwrite and not dry_run:
        return _blocked("output_dir already exists and is not empty. Use --overwrite to replace apply outputs.", "OUTPUT_DIR_EXISTS", backend=backend, plan=plan, output_dir=str(output_root))

    try:
        if plan.get("read_mode") == "sample_files_folder":
            extracted = _extract_sample_files_folder(source_path, plan)
        elif is_container:
            extracted = _extract_container(source_path, plan)
        elif is_hierarchical:
            extracted = _extract_hierarchical_container(source_path, plan)
        elif plan.get("sample_orientation") == "columns" and (plan.get("samples_as_columns") or {}).get("enabled") is True:
            table = _read_workbook_raw_table(source_path, plan) if is_workbook else _read_raw_table(source_path, plan)
            extracted = _extract_samples_as_columns(table, plan)
        else:
            table = _read_workbook_text_table(source_path, plan) if is_workbook else _read_text_table(source_path, plan)
            extracted = _extract_columns(table, plan)
        if (plan.get("sample_ids_file") or {}).get("path"):
            external_sample_ids, external_ids_filled = _load_external_sample_ids(plan)
            extracted["sample_ids"] = external_sample_ids
            extracted["sample_id_status"] = "partially_generated_after_confirmation" if external_ids_filled else "original"
        if (plan.get("label_file") or {}).get("path"):
            if is_container and _generated_sample_ids_used(plan) and not (plan.get("container") or {}).get("sample_ids_var"):
                raise ApplyBlockedError("EXTERNAL_LABEL_REQUIRES_SAMPLE_IDS", "External label alignment for container data requires sample_ids_var.")
            if is_hierarchical and _generated_sample_ids_used(plan) and not (plan.get("hierarchical_container") or {}).get("sample_ids_path"):
                raise ApplyBlockedError("EXTERNAL_LABEL_REQUIRES_SAMPLE_IDS", "External label alignment for hierarchical container data requires sample_ids_path.")
            extracted = _apply_external_label_file(extracted, plan, read_plan_path=read_plan_path, source_path=source_path if is_workbook else None)
        if (plan.get("metadata_file") or {}).get("path"):
            extracted = _apply_external_metadata_file(extracted, plan, read_plan_path=read_plan_path)
        _validate_shapes_and_values(extracted, plan)
        band_axis, band_warnings = _build_band_axis(plan, extracted["spectral_columns"])
        if extracted.get("band_axis") is None:
            extracted["band_axis"] = band_axis
        elif len(extracted["band_axis"]) != len(extracted["spectral_columns"]):
            raise ApplyBlockedError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=len(extracted["spectral_columns"]), observed=len(extracted["band_axis"]))
        audit_warnings = audit_post_read(source_path, plan, extracted, extracted["band_axis"])
    except ApplyBlockedError as exc:
        return _blocked(exc.message, exc.code, backend=backend, plan=plan, output_dir=str(output_root), **exc.details)
    except PostReadAuditBlocked as exc:
        return _blocked(exc.message, exc.code, backend=backend, plan=plan, output_dir=str(output_root), **exc.details)
    except Exception as exc:
        return _blocked(f"Unexpected apply_read_plan failure: {exc}", "APPLY_READ_PLAN_FAILED", backend=backend, plan=plan, output_dir=str(output_root))

    warnings = list(extracted.get("warnings") or []) + band_warnings + audit_warnings
    result = _build_result(plan, output_root, extracted, "dry_run" if dry_run else "applied", warnings, strict, path_resolution)
    if dry_run:
        result["data_refs"] = {}
        result["execution_summary"]["dry_run"] = True
        return ok_response("apply_read_plan", result, backend=backend, warnings=warnings)

    try:
        refs = write_apply_outputs(
            output_dir=output_root,
            X=extracted["X"],
            spectral_columns=extracted["spectral_columns"],
            y=extracted["y"],
            y_name=extracted["y_name"],
            sample_ids=extracted["sample_ids"],
            band_axis=extracted["band_axis"],
            band_unit=plan.get("band_unit"),
            metadata_rows=extracted["metadata_rows"],
            apply_result={**result, "data_refs": {}},
            warnings=warnings,
        )
    except Exception as exc:
        return _blocked(f"Could not write apply outputs: {exc}", "OUTPUT_WRITE_FAILED", backend=backend, plan=plan, output_dir=str(output_root))
    result["data_refs"] = refs
    result["execution_summary"]["dry_run"] = False
    return ok_response("apply_read_plan", result, backend=backend, warnings=warnings)


def _read_text_table(path: Path, plan: dict[str, Any]) -> list[dict[str, str]]:
    delimiter = _delimiter_from_plan(plan, path)
    encoding = plan.get("encoding") or "utf-8-sig"
    skiprows = int(plan.get("skiprows") or 0)
    header_row = plan.get("header_row")
    if header_row is not None and int(header_row) < skiprows:
        raise ApplyBlockedError("HEADER_BEFORE_SKIPROWS", "header_row cannot be before skipped rows.", header_row=header_row, skiprows=skiprows)
    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            lines = handle.readlines()
    except UnicodeDecodeError as exc:
        raise ApplyBlockedError("DECODE_FAILED", f"Could not decode source file with encoding {encoding}.", error=str(exc)) from exc
    except PermissionError as exc:
        raise ApplyBlockedError("PERMISSION_DENIED", "Permission denied while reading source file.", path=str(path)) from exc
    if not lines:
        raise ApplyBlockedError("EMPTY_FILE", "Source table is empty.")

    header_rows = _header_rows_from_plan(plan)
    if header_rows:
        return _read_text_table_multi_header(lines, delimiter, header_rows, plan)

    start = int(header_row) if header_row is not None else skiprows
    candidate_lines = lines[start:]
    if not candidate_lines:
        raise ApplyBlockedError("NO_TABLE_AFTER_SKIPROWS", "No table rows remain after skiprows.", skiprows=skiprows)
    reader = _dict_reader(candidate_lines, delimiter)
    fieldnames = list(reader[0].keys()) if isinstance(reader, list) and reader else getattr(reader, "fieldnames", None)
    if not fieldnames:
        raise ApplyBlockedError("HEADER_MISSING", "Could not read table header from read_plan parameters.")
    rows = [dict(row) for row in reader]
    rows = _slice_data_rows(rows, plan)
    if not rows:
        raise ApplyBlockedError("NO_DATA_ROWS", "Table contains no data rows.")
    return rows


def _read_raw_table(path: Path, plan: dict[str, Any]) -> list[list[str]]:
    delimiter = _delimiter_from_plan(plan, path)
    encoding = plan.get("encoding") or "utf-8-sig"
    skiprows = int(plan.get("skiprows") or 0)
    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
    except UnicodeDecodeError as exc:
        raise ApplyBlockedError("DECODE_FAILED", f"Could not decode source file with encoding {encoding}.", error=str(exc)) from exc
    except PermissionError as exc:
        raise ApplyBlockedError("PERMISSION_DENIED", "Permission denied while reading source file.", path=str(path)) from exc
    rows = rows[skiprows:]
    if not rows:
        raise ApplyBlockedError("EMPTY_FILE", "Source table is empty.")
    return rows


def _read_workbook_text_table(path: Path, plan: dict[str, Any]) -> list[dict[str, str]]:
    rows = _read_workbook_raw_table(path, plan)
    header_rows = _header_rows_from_plan(plan)
    if header_rows:
        return _rows_from_multi_header(rows, header_rows, plan)
    header_row = int(plan.get("header_row") if plan.get("header_row") is not None else 0)
    if header_row < 0 or header_row >= len(rows):
        raise ApplyBlockedError("HEADER_ROW_INVALID", "header_row is outside the workbook sheet.", header_row=header_row)
    header = [str(value).strip() for value in rows[header_row]]
    data_rows = rows[header_row + 1 :]
    if not header or not any(header):
        raise ApplyBlockedError("HEADER_MISSING", "Workbook sheet header row is empty.")
    data_rows = _slice_raw_data_rows(data_rows, plan, base_row=header_row + 1)
    if not data_rows:
        raise ApplyBlockedError("NO_DATA_ROWS", "Workbook sheet contains no data rows.")
    return [{header[idx]: row[idx] if idx < len(row) else "" for idx in range(len(header))} for row in data_rows]


def _read_workbook_raw_table(path: Path, plan: dict[str, Any]) -> list[list[str]]:
    engine = resolve_table_engine(path)
    sheet = resolve_sheet(path, sheet_name=_plan_sheet_name(plan), sheet_index=_plan_sheet_index(plan), engine=engine)
    skiprows = int(plan.get("skiprows") or 0)
    try:
        import pandas as pd

        df = pd.read_excel(path, sheet_name=sheet, engine=engine, header=None, skiprows=skiprows)
    except ImportError as exc:
        raise ApplyBlockedError("WORKBOOK_ENGINE_MISSING", "pandas is required for workbook reading.", package="pandas") from exc
    except ValueError as exc:
        raise ApplyBlockedError("SHEET_NOT_FOUND", "Requested workbook sheet was not found.", sheet=sheet, path=str(path), error=str(exc)) from exc
    except Exception as exc:
        code = "ODS_ENGINE_MISSING" if path.suffix.lower() == ".ods" and importlib.util.find_spec("odf") is None else "EXCEL_READ_FAILED"
        raise ApplyBlockedError(code, f"Workbook read failed: {exc}", sheet=sheet, path=str(path)) from exc
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ApplyBlockedError("EMPTY_SHEET", "Workbook sheet is empty.", sheet=sheet, path=str(path))
    return [[_cell_to_string(value) for value in row] for row in df.values.tolist()]


def _header_rows_from_plan(plan: dict[str, Any]) -> list[int]:
    rows = plan.get("header_rows") or (plan.get("table_layout") or {}).get("header_rows") or []
    return [int(row) for row in rows if row is not None]


def _read_text_table_multi_header(lines: list[str], delimiter: str, header_rows: list[int], plan: dict[str, Any]) -> list[dict[str, str]]:
    if not header_rows:
        raise ApplyBlockedError("HEADER_ROWS_INVALID", "header_rows must not be empty.")
    if min(header_rows) < 0 or max(header_rows) >= len(lines):
        raise ApplyBlockedError("HEADER_ROWS_INVALID", "header_rows are outside the source table.", header_rows=header_rows)
    raw_rows = [_split_delimited_line(line, delimiter) for line in lines]
    return _rows_from_multi_header(raw_rows, header_rows, plan)


def _rows_from_multi_header(raw_rows: list[list[str]], header_rows: list[int], plan: dict[str, Any]) -> list[dict[str, str]]:
    if not header_rows:
        raise ApplyBlockedError("HEADER_ROWS_INVALID", "header_rows must not be empty.")
    if min(header_rows) < 0 or max(header_rows) >= len(raw_rows):
        raise ApplyBlockedError("HEADER_ROWS_INVALID", "header_rows are outside the source table.", header_rows=header_rows)
    header_parts = [raw_rows[row] for row in header_rows]
    width = max((len(row) for row in header_parts), default=0)
    if width == 0:
        raise ApplyBlockedError("HEADER_ROWS_INVALID", "header_rows are empty.")
    columns = flatten_multirow_columns(header_parts, original_names=True)
    data_start = plan.get("data_start_row")
    if data_start is None:
        data_start = max(header_rows) + 1
    data_end = plan.get("data_end_row")
    data_rows = raw_rows[int(data_start) : int(data_end) + 1 if data_end is not None else None]
    if not data_rows:
        raise ApplyBlockedError("NO_DATA_ROWS", "No data rows remain after multi-row header.")
    return [{columns[idx]: row[idx] if idx < len(row) else "" for idx in range(len(columns))} for row in data_rows]


def flatten_multirow_columns(header_parts: list[list[Any]], *, original_names: bool = False) -> list[str]:
    width = max((len(row) for row in header_parts), default=0)
    columns: list[str] = []
    seen: dict[str, int] = {}
    for idx in range(width):
        parts = [str(row[idx]).strip() for row in header_parts if idx < len(row) and str(row[idx]).strip()]
        if not parts:
            name = f"column_{idx + 1}"
        elif original_names and parts[-1]:
            name = parts[-1]
        else:
            name = "_".join(parts)
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        columns.append(name)
    return columns


def _slice_data_rows(rows: list[dict[str, str]], plan: dict[str, Any]) -> list[dict[str, str]]:
    header_row = int(plan.get("header_row") if plan.get("header_row") is not None else 0)
    base_row = header_row + 1
    start = plan.get("data_start_row")
    end = plan.get("data_end_row")
    rel_start = max(0, int(start) - base_row) if start is not None else 0
    rel_end = int(end) - base_row + 1 if end is not None else None
    return rows[rel_start:rel_end]


def _slice_raw_data_rows(rows: list[list[str]], plan: dict[str, Any], *, base_row: int) -> list[list[str]]:
    start = plan.get("data_start_row")
    end = plan.get("data_end_row")
    rel_start = max(0, int(start) - base_row) if start is not None else 0
    rel_end = int(end) - base_row + 1 if end is not None else None
    return rows[rel_start:rel_end]


def resolve_table_engine(path: Path) -> str:
    suffix = path.suffix.lower()
    if importlib.util.find_spec("pandas") is None:
        raise ApplyBlockedError("WORKBOOK_ENGINE_MISSING", "pandas is required for workbook reading.", package="pandas")
    if suffix in {".xlsx", ".xlsm"}:
        if importlib.util.find_spec("openpyxl") is None:
            raise ApplyBlockedError("EXCEL_ENGINE_MISSING", "openpyxl is required for .xlsx/.xlsm reading.", package="openpyxl")
        return "openpyxl"
    if suffix == ".xls":
        if importlib.util.find_spec("xlrd") is None:
            raise ApplyBlockedError("EXCEL_ENGINE_MISSING", "xlrd is required for .xls reading.", package="xlrd")
        return "xlrd"
    if suffix == ".ods":
        if importlib.util.find_spec("odf") is None:
            raise ApplyBlockedError("ODS_ENGINE_MISSING", "odfpy is required for .ods reading.", package="odfpy")
        return "odf"
    raise ApplyBlockedError("UNSUPPORTED_FILE_TYPE", "Unsupported workbook file type.", suffix=suffix)


def list_workbook_sheets(path: Path, *, engine: str | None = None) -> list[str]:
    try:
        import pandas as pd

        return list(pd.ExcelFile(path, engine=engine or resolve_table_engine(path)).sheet_names)
    except ApplyBlockedError:
        raise
    except Exception as exc:
        raise ApplyBlockedError("EXCEL_READ_FAILED", f"Could not list workbook sheets: {exc}", path=str(path)) from exc


def resolve_sheet(path: Path, *, sheet_name: str | None = None, sheet_index: int | None = None, engine: str | None = None) -> str | int:
    sheets = list_workbook_sheets(path, engine=engine)
    if sheet_name:
        if sheet_name not in sheets:
            raise ApplyBlockedError("SHEET_NOT_FOUND", "Requested workbook sheet was not found.", sheet=sheet_name, available=sheets)
        return sheet_name
    if sheet_index is not None:
        if sheet_index < 0 or sheet_index >= len(sheets):
            raise ApplyBlockedError("SHEET_NOT_FOUND", "Requested workbook sheet index was not found.", sheet_index=sheet_index, available=sheets)
        return sheet_index
    if len(sheets) == 1:
        return sheets[0]
    raise ApplyBlockedError("SPECTRAL_SHEET_NOT_CONFIRMED", "Workbook contains multiple sheets; specify spectral_sheet.", available=sheets)


def _extract_container(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    arrays = _load_container_arrays(path)
    variables = plan.get("variable_map") or {}
    container = plan.get("container") or {}
    x_var = variables.get("X") or container.get("x_var")
    if not x_var:
        x_var = _single_x_candidate(arrays)
    if x_var not in arrays:
        raise ApplyBlockedError("X_VAR_NOT_FOUND", "Declared X variable does not exist.", variable=x_var, available=sorted(arrays))

    X = _array_to_matrix(arrays[x_var], orientation=plan.get("sample_orientation") or container.get("sample_orientation") or "rows")
    n_samples = len(X)
    n_features = len(X[0]) if X else 0

    sample_ids_var = variables.get("sample_ids") or container.get("sample_ids_var")
    if sample_ids_var:
        if sample_ids_var not in arrays:
            raise ApplyBlockedError("SAMPLE_IDS_VAR_NOT_FOUND", "Declared sample_ids variable does not exist.", variable=sample_ids_var, available=sorted(arrays))
        sample_ids = _vector_to_values(arrays[sample_ids_var], "sample_ids")
    else:
        if container.get("allow_generated_sample_ids") is False:
            raise ApplyBlockedError("SAMPLE_IDS_VAR_NOT_FOUND", "sample_ids variable is required for this read_plan.")
        sample_ids = _generated_sample_ids(n_samples)
    if len(sample_ids) != n_samples:
        raise ApplyBlockedError("SAMPLE_IDS_LENGTH_MISMATCH", "sample_ids length must equal X sample count.", expected=n_samples, observed=len(sample_ids))

    band_axis_var = variables.get("band_axis") or container.get("band_axis_var")
    if band_axis_var:
        if band_axis_var not in arrays:
            raise ApplyBlockedError("BAND_AXIS_VAR_NOT_FOUND", "Declared band_axis variable does not exist.", variable=band_axis_var, available=sorted(arrays))
        band_axis = _vector_to_values(arrays[band_axis_var], "band_axis")
    else:
        if container.get("allow_generated_band_axis") is False:
            raise ApplyBlockedError("BAND_AXIS_VAR_NOT_FOUND", "band_axis variable is required for this read_plan.")
        band_axis = [f"feature_{idx:03d}" for idx in range(1, n_features + 1)]
    if len(band_axis) != n_features:
        raise ApplyBlockedError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=n_features, observed=len(band_axis))

    y_var = variables.get("y") or container.get("y_var")
    y = None
    y_name = None
    if y_var:
        if y_var not in arrays:
            raise ApplyBlockedError("Y_VAR_NOT_FOUND", "Declared y variable does not exist.", variable=y_var, available=sorted(arrays))
        y = _vector_to_values(arrays[y_var], "y")
        if len(y) != n_samples:
            raise ApplyBlockedError("Y_LENGTH_MISMATCH", "y length must equal X sample count.", expected=n_samples, observed=len(y))
        y_name = str(y_var)

    metadata_var = variables.get("metadata") or container.get("metadata_var")
    metadata_rows = None
    if metadata_var:
        if metadata_var not in arrays:
            raise ApplyBlockedError("METADATA_VAR_NOT_FOUND", "Declared metadata variable does not exist.", variable=metadata_var, available=sorted(arrays))
        metadata_rows = _metadata_to_rows(arrays[metadata_var], n_samples)

    spectral_columns = [str(value) for value in band_axis]
    return {
        "X": X,
        "spectral_columns": spectral_columns,
        "sample_ids": sample_ids,
        "y": y,
        "y_name": y_name,
        "metadata_rows": metadata_rows,
        "band_axis": band_axis,
        "alignment_summary": _empty_alignment_summary("embedded" if y is not None else "none", n_samples),
        "label_source": "embedded" if y is not None else "none",
        "external_label_used": False,
        "warnings": [],
    }


def _load_container_arrays(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if importlib.util.find_spec("numpy") is None:
        raise ApplyBlockedError("CONTAINER_READ_FAILED", "numpy is required for NPY/NPZ/MAT container reading.", package="numpy")
    import numpy as np

    try:
        if suffix == ".npy":
            return {"__npy_array__": np.load(path, allow_pickle=False)}
        if suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                return {name: data[name] for name in data.files}
        if suffix == ".mat":
            if _is_mat_v73(path):
                raise ApplyBlockedError("MAT_V73_NOT_SUPPORTED", "MAT v7.3 is HDF5-based and is not supported in this reader step.")
            if importlib.util.find_spec("scipy") is None:
                raise ApplyBlockedError("SCIPY_MISSING", "scipy is required for MAT reading.", package="scipy")
            import scipy.io

            data = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
            return {name: value for name, value in data.items() if not name.startswith("__")}
    except ApplyBlockedError:
        raise
    except NotImplementedError as exc:
        raise ApplyBlockedError("MAT_V73_NOT_SUPPORTED", "MAT v7.3 is HDF5-based and is not supported in this reader step.") from exc
    except Exception as exc:
        raise ApplyBlockedError("CONTAINER_READ_FAILED", f"Container read failed: {exc}", path=str(path)) from exc
    raise ApplyBlockedError("UNSUPPORTED_FILE_TYPE", "Unsupported container file type.", suffix=suffix)


def _extract_hierarchical_container(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    arrays = _load_hierarchical_arrays(path)
    variables = plan.get("dataset_map") or {}
    container = plan.get("hierarchical_container") or {}
    x_path = variables.get("X") or container.get("x_path")
    if not x_path:
        x_path = _single_x_candidate(arrays)
    if x_path not in arrays:
        code = "VARIABLE_PATH_NOT_FOUND" if path.suffix.lower() == ".nc" else "DATASET_PATH_NOT_FOUND"
        raise ApplyBlockedError(code, "Declared X dataset path does not exist.", path=x_path, available=sorted(arrays))

    X = _array_to_matrix(arrays[x_path], orientation=plan.get("sample_orientation") or container.get("sample_orientation") or "rows")
    n_samples = len(X)
    n_features = len(X[0]) if X else 0

    sample_ids_path = variables.get("sample_ids") or container.get("sample_ids_path")
    if sample_ids_path:
        if sample_ids_path not in arrays:
            code = "VARIABLE_PATH_NOT_FOUND" if path.suffix.lower() == ".nc" else "DATASET_PATH_NOT_FOUND"
            raise ApplyBlockedError(code, "Declared sample_ids dataset path does not exist.", path=sample_ids_path, available=sorted(arrays))
        sample_ids = _vector_to_values(arrays[sample_ids_path], "sample_ids")
    else:
        if container.get("allow_generated_sample_ids") is False:
            raise ApplyBlockedError("DATASET_PATH_NOT_FOUND", "sample_ids path is required for this read_plan.")
        sample_ids = _generated_sample_ids(n_samples)
    if len(sample_ids) != n_samples:
        raise ApplyBlockedError("SAMPLE_IDS_LENGTH_MISMATCH", "sample_ids length must equal X sample count.", expected=n_samples, observed=len(sample_ids))

    band_axis_path = variables.get("band_axis") or container.get("band_axis_path")
    if band_axis_path:
        if band_axis_path not in arrays:
            code = "VARIABLE_PATH_NOT_FOUND" if path.suffix.lower() == ".nc" else "DATASET_PATH_NOT_FOUND"
            raise ApplyBlockedError(code, "Declared band_axis dataset path does not exist.", path=band_axis_path, available=sorted(arrays))
        band_axis = _vector_to_values(arrays[band_axis_path], "band_axis")
    else:
        if container.get("allow_generated_band_axis") is False:
            raise ApplyBlockedError("DATASET_PATH_NOT_FOUND", "band_axis path is required for this read_plan.")
        band_axis = [f"feature_{idx:03d}" for idx in range(1, n_features + 1)]
    if len(band_axis) != n_features:
        raise ApplyBlockedError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=n_features, observed=len(band_axis))

    y_path = variables.get("y") or container.get("y_path")
    y = None
    y_name = None
    if y_path:
        if y_path not in arrays:
            code = "VARIABLE_PATH_NOT_FOUND" if path.suffix.lower() == ".nc" else "DATASET_PATH_NOT_FOUND"
            raise ApplyBlockedError(code, "Declared y dataset path does not exist.", path=y_path, available=sorted(arrays))
        y = _vector_to_values(arrays[y_path], "y")
        if len(y) != n_samples:
            raise ApplyBlockedError("Y_LENGTH_MISMATCH", "y length must equal X sample count.", expected=n_samples, observed=len(y))
        y_name = str(y_path)

    metadata_path = variables.get("metadata") or container.get("metadata_path")
    metadata_rows = None
    if metadata_path:
        if metadata_path not in arrays:
            code = "VARIABLE_PATH_NOT_FOUND" if path.suffix.lower() == ".nc" else "DATASET_PATH_NOT_FOUND"
            raise ApplyBlockedError(code, "Declared metadata dataset path does not exist.", path=metadata_path, available=sorted(arrays))
        metadata_rows = _metadata_to_rows(arrays[metadata_path], n_samples)

    spectral_columns = [str(value) for value in band_axis]
    return {
        "X": X,
        "spectral_columns": spectral_columns,
        "sample_ids": sample_ids,
        "y": y,
        "y_name": y_name,
        "metadata_rows": metadata_rows,
        "band_axis": band_axis,
        "alignment_summary": _empty_alignment_summary("embedded" if y is not None else "none", n_samples),
        "label_source": "embedded" if y is not None else "none",
        "external_label_used": False,
        "warnings": [],
    }


def _load_hierarchical_arrays(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5_arrays(path)
    if suffix == ".nc":
        return _load_netcdf_arrays(path)
    raise ApplyBlockedError("UNSUPPORTED_FILE_TYPE", "Unsupported hierarchical container file type.", suffix=suffix)


def _load_hdf5_arrays(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("h5py") is None:
        raise ApplyBlockedError("H5PY_MISSING", "h5py is required for HDF5 reading.", package="h5py")
    try:
        import h5py

        arrays: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    arrays["/" + name.strip("/")] = obj[()]

            handle.visititems(visit)
        return arrays
    except ApplyBlockedError:
        raise
    except Exception as exc:
        raise ApplyBlockedError("HDF5_READ_FAILED", f"HDF5 read failed: {exc}", path=str(path)) from exc


def _load_netcdf_arrays(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("netCDF4") is None:
        raise ApplyBlockedError("NETCDF4_MISSING", "netCDF4 is required for NetCDF reading.", package="netCDF4")
    try:
        import netCDF4

        arrays: dict[str, Any] = {}
        with netCDF4.Dataset(path, "r") as handle:
            _collect_netcdf_arrays(handle, "", arrays)
        return arrays
    except ApplyBlockedError:
        raise
    except Exception as exc:
        raise ApplyBlockedError("NETCDF_READ_FAILED", f"NetCDF read failed: {exc}", path=str(path)) from exc


def _collect_netcdf_arrays(group: Any, prefix: str, arrays: dict[str, Any]) -> None:
    group_path = prefix or "/"
    for name, var in group.variables.items():
        var_path = f"{group_path.rstrip('/')}/{name}" if group_path != "/" else name
        arrays[var_path] = var[:]
    for name, child in group.groups.items():
        child_prefix = f"{group_path.rstrip('/')}/{name}" if group_path != "/" else name
        _collect_netcdf_arrays(child, child_prefix, arrays)


def _single_x_candidate(arrays: dict[str, Any]) -> str:
    candidates = []
    for name, value in arrays.items():
        try:
            arr = _as_numpy_array(value)
            if arr.ndim == 2 and _is_numeric_array(arr):
                candidates.append(name)
        except ApplyBlockedError:
            continue
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ApplyBlockedError("MULTIPLE_X_CANDIDATES", "Multiple 2D numeric arrays could be X; specify x_var.", candidates=candidates)
    raise ApplyBlockedError("X_VAR_NOT_FOUND", "No 2D numeric X variable was found.", available=sorted(arrays))


def _array_to_matrix(value: Any, *, orientation: str) -> list[list[float]]:
    arr = _as_numpy_array(value)
    if arr.ndim != 2:
        raise ApplyBlockedError("X_VAR_NOT_2D", "X variable must be a 2D numeric matrix.", shape=list(arr.shape))
    if not _is_numeric_array(arr):
        raise ApplyBlockedError("X_VAR_NOT_NUMERIC", "X variable must be numeric.", dtype=str(arr.dtype))
    if orientation == "columns":
        arr = arr.T
    elif orientation not in {"rows", None}:
        raise ApplyBlockedError("APPLY_ORIENTATION_UNSUPPORTED", "Container data supports sample_orientation rows or columns.", sample_orientation=orientation)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ApplyBlockedError("EMPTY_X", "X matrix has zero rows or zero columns.")
    return [[float(value) for value in row] for row in arr.tolist()]


def _vector_to_values(value: Any, role: str) -> list[Any]:
    arr = _as_numpy_array(value)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ApplyBlockedError(f"{role.upper()}_VAR_NOT_VECTOR", f"{role} variable must be one-dimensional.", shape=list(arr.shape))
    return [_scalar_to_python(item) for item in arr.tolist()]


def _metadata_to_rows(value: Any, n_samples: int) -> list[dict[str, Any]]:
    arr = _as_numpy_array(value)
    if arr.ndim == 2 and 1 in arr.shape and arr.size == n_samples:
        arr = arr.reshape(-1)
    if arr.ndim == 1:
        if len(arr) != n_samples:
            raise ApplyBlockedError("METADATA_ROWS_MISMATCH", "metadata length must equal X sample count.", expected=n_samples, observed=len(arr))
        return [{"metadata": _scalar_to_python(item)} for item in arr.tolist()]
    if arr.ndim == 2:
        if arr.shape[0] != n_samples:
            raise ApplyBlockedError("METADATA_ROWS_MISMATCH", "metadata row count must equal X sample count.", expected=n_samples, observed=int(arr.shape[0]))
        return [
            {f"metadata_{idx + 1}": _scalar_to_python(item) for idx, item in enumerate(row)}
            for row in arr.tolist()
        ]
    raise ApplyBlockedError("COMPLEX_MAT_VARIABLE_UNSUPPORTED", "metadata variable must be a 1D or 2D array.", shape=list(arr.shape))


def _as_numpy_array(value: Any) -> Any:
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype == object:
        try:
            arr = np.vectorize(_mat_object_scalar_to_python, otypes=[object])(arr)
        except Exception as exc:
            raise ApplyBlockedError("COMPLEX_MAT_VARIABLE_UNSUPPORTED", "Complex object/cell variables are not supported.", dtype=str(arr.dtype)) from exc
    return arr


def _is_numeric_array(arr: Any) -> bool:
    import numpy as np

    return bool(np.issubdtype(arr.dtype, np.number)) and not np.issubdtype(arr.dtype, np.complexfloating)


def _scalar_to_python(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _mat_object_scalar_to_python(value: Any) -> Any:
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.size == 1:
            return _scalar_to_python(arr.reshape(-1)[0])
        if arr.ndim == 1 and arr.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in arr.tolist())
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool, bytes)):
        return _scalar_to_python(value)
    raise TypeError("unsupported MAT object value")


def _is_mat_v73(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(256)
        return b"MATLAB 7.3 MAT-file" in head or head.startswith(b"\x89HDF")
    except OSError:
        return False


def _generated_sample_ids_used(plan: dict[str, Any]) -> bool:
    return ((plan.get("sample_id") or {}).get("source") == "generated")


def _extract_columns(rows: list[dict[str, str]], plan: dict[str, Any]) -> dict[str, Any]:
    columns = list(rows[0].keys())
    sample_column = _resolve_role_column(columns, _role_column(plan.get("sample_id")), "sample_id")
    label_column = None if _role_source(plan.get("label")) == "external_file" else _resolve_role_column(columns, _role_column(plan.get("label")), "label")
    target_columns = [] if _role_source(plan.get("target")) == "external_file" else _resolve_role_columns(columns, _role_columns(plan.get("target")), "target")
    if label_column and target_columns:
        raise ApplyBlockedError("LABEL_TARGET_CONFLICT", "label_column and target_columns cannot both be used in one read without explicit conflict resolution.")
    metadata_spec = plan.get("metadata") or {}
    metadata_columns = _resolve_role_columns(columns, list(metadata_spec.get("columns") or []), "metadata", required=metadata_spec.get("required") is True)
    metadata_warnings: list[dict[str, Any]] = []
    spectral_columns = _resolve_spectral_columns(plan, columns)
    _require_columns(spectral_columns, columns, "spectral_columns")
    missing_metadata = [column for column in metadata_columns if column not in columns]
    if missing_metadata and metadata_spec.get("required") is True:
        _require_columns(metadata_columns, columns, "metadata")
    if missing_metadata and metadata_spec.get("required") is not True:
        metadata_warnings.append({
            "code": "OPTIONAL_METADATA_COLUMNS_MISSING",
            "message": "Optional metadata columns are missing and were omitted.",
            "severity": "warning",
            "details": {"missing": missing_metadata},
        })
        metadata_columns = [column for column in metadata_columns if column in columns]
    if sample_column is not None:
        _require_columns([sample_column], columns, "sample_id")
    if label_column:
        _require_columns([label_column], columns, "label")
    if target_columns:
        _require_columns(target_columns, columns, "target")

    non_spectral_columns = {value for value in [sample_column, label_column] if value is not None}
    non_spectral_columns.update(metadata_columns)
    non_spectral_columns.update(target_columns)
    spectral_columns = [column for column in spectral_columns if column not in non_spectral_columns]
    if not spectral_columns:
        raise ApplyBlockedError("SPECTRAL_BLOCK_EMPTY", "No spectral columns remain after excluding sample_id, label, target, and metadata columns.")

    X: list[list[float | None]] = []
    for row_idx, row in enumerate(rows):
        values: list[float] = []
        for column in spectral_columns:
            raw = row.get(column)
            try:
                values.append(_to_float_value(raw, plan))
            except (TypeError, ValueError) as exc:
                code = "SPECTRAL_BLOCK_NON_NUMERIC" if _uses_spectral_range(plan) else "X_NON_NUMERIC"
                raise ApplyBlockedError(code, "X contains a value that cannot be converted to float.", row_index=row_idx, column=column, value=raw) from exc
        X.append(values)

    sample_id_status = "original" if sample_column is not None else "generated"
    sample_id_source = _sample_id_source(sample_column, columns, plan)
    sample_ids = [row.get(sample_column) for row in rows] if sample_column is not None else _generated_sample_ids(len(rows))
    sample_ids, filled_missing_ids = _fill_missing_sample_ids(sample_ids, plan)
    if filled_missing_ids:
        sample_id_status = "partially_generated_after_confirmation"
        sample_id_source = "partially_generated_after_confirmation"
    if target_columns:
        y_name: str | list[str] | None = target_columns
        y = [[row.get(column) for column in target_columns] for row in rows]
    else:
        y_name = label_column
        y = [row.get(y_name) for row in rows] if y_name else None
    metadata_rows = [{column: row.get(column) for column in metadata_columns} for row in rows] if metadata_columns else None

    return {
        "X": X,
        "spectral_columns": spectral_columns,
        "sample_ids": sample_ids,
        "y": y,
        "y_name": y_name,
        "metadata_rows": metadata_rows,
        "alignment_summary": _empty_alignment_summary("embedded" if y is not None else "none", len(X)),
        "label_source": "embedded" if y is not None else "none",
        "external_label_used": False,
        "warnings": metadata_warnings,
        "missing_value_status": _missing_value_status(X),
        "sample_id_status": sample_id_status,
        "sample_id_source": sample_id_source,
        "label_has_missing": _sequence_has_missing(y) if y is not None else False,
        "metadata_has_missing": _metadata_has_missing(metadata_rows),
    }


def _extract_samples_as_columns(rows: list[list[str]], plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan.get("samples_as_columns") or {}
    header_row = int(plan.get("header_row") if plan.get("header_row") is not None else 0)
    if header_row < 0 or header_row >= len(rows):
        raise ApplyBlockedError("HEADER_ROW_INVALID", "header_row is outside the sampled table.", header_row=header_row)
    header = rows[header_row]
    data_start = int(spec.get("data_start_row") if spec.get("data_start_row") is not None else plan.get("data_start_row") if plan.get("data_start_row") is not None else header_row + 1)
    data_end = spec.get("data_end_row")
    data_rows = rows[data_start : int(data_end) + 1 if data_end is not None else None]
    if not data_rows:
        raise ApplyBlockedError("NO_DATA_ROWS", "No spectral rows remain for samples-as-columns extraction.")

    band_axis_column = spec.get("band_axis_column")
    band_idx = _column_index(header, band_axis_column, default=0)
    sample_start = spec.get("sample_start_column", plan.get("data_start_column", 1))
    sample_end = spec.get("sample_end_column", plan.get("data_end_column"))
    sample_indices = _column_range(header, sample_start, sample_end, exclude={band_idx})
    if not sample_indices:
        raise ApplyBlockedError("SAMPLE_COLUMNS_MISSING", "samples_as_columns requires at least one sample column.")

    sample_ids, filled_missing_ids = _fill_missing_sample_ids([str(header[idx]).strip() for idx in sample_indices], plan)

    band_axis: list[Any] = []
    columns_as_values: list[list[float | None]] = [[] for _ in sample_indices]
    for row_idx, row in enumerate(data_rows):
        if band_idx >= len(row):
            raise ApplyBlockedError("BAND_AXIS_COLUMN_MISSING", "band_axis_column is missing from a data row.", row_index=row_idx)
        band_axis.append(_parse_axis_value(row[band_idx]))
        for out_idx, src_idx in enumerate(sample_indices):
            raw = row[src_idx] if src_idx < len(row) else ""
            try:
                columns_as_values[out_idx].append(_to_float_value(raw, plan))
            except (TypeError, ValueError) as exc:
                raise ApplyBlockedError("X_NON_NUMERIC", "X contains a value that cannot be converted to float.", row_index=row_idx, column=header[src_idx], value=raw) from exc

    spectral_columns = [str(value) for value in band_axis]
    return {
        "X": columns_as_values,
        "spectral_columns": spectral_columns,
        "sample_ids": sample_ids,
        "y": None,
        "y_name": None,
        "metadata_rows": None,
        "band_axis": band_axis,
        "alignment_summary": _empty_alignment_summary("none", len(sample_ids)),
        "label_source": "none",
        "external_label_used": False,
        "warnings": [],
        "missing_value_status": _missing_value_status(columns_as_values),
        "sample_id_status": "partially_generated_after_confirmation" if filled_missing_ids else "original",
        "label_has_missing": False,
        "metadata_has_missing": False,
    }


def _extract_sample_files_folder(source_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise ApplyBlockedError("SOURCE_FOLDER_NOT_FOUND", "sample_files_folder source_path must resolve to a folder.", path=str(source_dir))
    files = _sample_files(source_dir, plan)
    if not files:
        raise ApplyBlockedError("SAMPLE_FILE_PATTERN_NO_MATCH", "No sample files matched sample_file_pattern.", source_dir=str(source_dir), pattern=plan.get("sample_file_pattern"))

    band_column = plan.get("sample_file_band_axis_column") or (plan.get("sample_file_columns") or {}).get("band_axis")
    value_column = plan.get("sample_file_value_column") or (plan.get("sample_file_columns") or {}).get("value")
    if band_column is None or value_column is None:
        raise ApplyBlockedError("SAMPLE_FILE_COLUMNS_MISSING", "sample_files_folder requires band and value columns.")

    reference_axis: list[Any] | None = None
    X: list[list[float | None]] = []
    sample_ids: list[str] = []
    y: list[str] | None = [] if _folder_label_enabled(plan) or _file_label_enabled(plan) else None
    metadata_rows: list[dict[str, Any]] = []
    for file_path in files:
        rows = _read_sample_file_table(file_path, plan)
        if not rows:
            raise ApplyBlockedError("SAMPLE_FILE_EMPTY", "Sample file contains no data rows.", path=str(file_path))
        columns = list(rows[0].keys())
        band_key = _sample_file_column(columns, band_column, "band_axis")
        value_key = _sample_file_column(columns, value_column, "value")
        band_values: list[Any] = []
        intensities: list[float | None] = []
        for row_index, row in enumerate(rows):
            band_values.append(_parse_axis_value(row.get(band_key)))
            try:
                intensities.append(_to_float_value(row.get(value_key), plan))
            except (TypeError, ValueError) as exc:
                raise ApplyBlockedError("X_NON_NUMERIC", "Sample file intensity column contains a non-numeric value.", path=str(file_path), row_index=row_index, column=value_key, value=row.get(value_key)) from exc
        if reference_axis is None:
            reference_axis = band_values
        elif [str(value) for value in band_values] != [str(value) for value in reference_axis]:
            raise ApplyBlockedError("PER_FILE_BAND_AXIS_MISMATCH", "All sample files must share the same band_axis.", path=str(file_path))
        X.append(intensities)
        sample_ids.append(_sample_id_from_file(file_path, plan))
        if y is not None:
            y.append(_label_from_file(file_path, source_dir, plan))
        metadata_rows.append(
            {
                "source_file": file_path.name,
                "relative_path": file_path.relative_to(source_dir).as_posix(),
                "source_folder": file_path.parent.relative_to(source_dir).as_posix() if file_path.parent != source_dir else ".",
            }
        )

    label_source = "folder_name" if _folder_label_enabled(plan) else "file_name" if _file_label_enabled(plan) else "none"
    return {
        "X": X,
        "spectral_columns": [str(value) for value in (reference_axis or [])],
        "sample_ids": sample_ids,
        "y": y,
        "y_name": "label" if y is not None else None,
        "metadata_rows": metadata_rows,
        "band_axis": reference_axis or [],
        "alignment_summary": _empty_alignment_summary(label_source, len(X)),
        "label_source": label_source,
        "external_label_used": False,
        "warnings": [],
        "missing_value_status": _missing_value_status(X),
        "sample_id_status": "generated",
        "label_has_missing": False,
        "metadata_has_missing": _metadata_has_missing(metadata_rows),
    }


def _apply_external_label_file(extracted: dict[str, Any], plan: dict[str, Any], *, read_plan_path: Path | None, source_path: Path | None = None) -> dict[str, Any]:
    label_spec = plan.get("label_file") or {}
    alignment = plan.get("alignment_plan") or {}
    if label_spec.get("sheet_name") or label_spec.get("sheet_index") is not None:
        label_path = source_path or Path(str(label_spec.get("path") or plan.get("source_path")))
        rows = _read_workbook_label_table(label_path, label_spec, plan)
    else:
        path_report = _resolve_auxiliary_path(label_spec.get("path"), plan, read_plan_path)
        label_path_value = path_report.get("resolved_path")
        if not label_path_value or not path_report.get("exists"):
            raise ApplyBlockedError("LABEL_FILE_NOT_FOUND", "label_file.path could not be resolved.", **path_report)
        label_path = Path(str(label_path_value))
        rows = _read_auxiliary_table(label_path, label_spec, plan)
    if not rows:
        raise ApplyBlockedError("LABEL_FILE_EMPTY", "External label file contains no data rows.", path=str(label_path))

    method = str(alignment.get("method") or plan.get("label_alignment") or "sample_id")
    row_order_requested = method == "row_order" or bool(plan.get("allow_row_order_labels"))
    label_column = label_spec.get("label_column") or _role_column(plan.get("label")) or _role_column(plan.get("target"))
    target_columns = list(label_spec.get("target_columns") or [])
    if not label_column and target_columns:
        label_column = target_columns[0]
    metadata_columns = list(label_spec.get("metadata_columns") or [])
    sample_ids = extracted.get("sample_ids") or []

    if row_order_requested:
        if len(rows) != len(sample_ids):
            raise ApplyBlockedError("ROW_ORDER_LENGTH_MISMATCH", "Row-order label alignment requires label rows to match X rows.", expected=len(sample_ids), observed=len(rows))
        if not label_column:
            raise ApplyBlockedError("LABEL_COLUMN_NOT_FOUND", "Row-order label alignment requires label_column or target_columns.")
        if metadata_columns:
            _require_columns([str(column) for column in metadata_columns], list(rows[0].keys()), "label_file.metadata_columns")
        _require_columns([str(label_column)], list(rows[0].keys()), "label_file.label_column")
        if not metadata_columns:
            excluded = {str(label_column)}
            metadata_columns = [column for column in rows[0].keys() if column not in excluded]
        y = [row.get(str(label_column)) for row in rows]
        existing_metadata = extracted.get("metadata_rows")
        metadata_rows: list[dict[str, Any]] | None = list(existing_metadata) if existing_metadata else ([{} for _ in sample_ids] if metadata_columns else None)
        if metadata_columns:
            if metadata_rows is None:
                metadata_rows = [{} for _ in sample_ids]
            for idx, row in enumerate(rows):
                metadata_rows[idx].update({str(column): row.get(str(column)) for column in metadata_columns})
        updated = dict(extracted)
        updated["y"] = y
        updated["y_name"] = str(label_column)
        updated["metadata_rows"] = metadata_rows
        updated["label_source"] = "external"
        updated["external_label_used"] = True
        updated["alignment_summary"] = {
            "alignment_status": "aligned",
            "alignment_method": "row_order",
            "label_alignment": "row_order",
            "label_source": "external",
            "external_label_used": True,
            "label_file_ref": str(label_spec.get("path")),
            "resolved_label_file": str(label_path),
            "preserve_spectrum_order": True,
            "aligned_sample_count": len(sample_ids),
            "spectra_sample_count": len(sample_ids),
            "label_row_count": len(rows),
            "missing_label_count": 0,
            "duplicate_label_key_count": 0,
            "unmatched_label_count": 0,
        }
        return updated

    join_key = alignment.get("join_key") or label_spec.get("sample_id_column") or _role_column(plan.get("sample_id")) or "sample_id"
    if not metadata_columns:
        excluded = {str(join_key)}
        if label_column:
            excluded.add(str(label_column))
        metadata_columns = [column for column in rows[0].keys() if column not in excluded]
    if str(join_key) not in rows[0]:
        raise ApplyBlockedError("ROW_ORDER_ALIGNMENT_NOT_ALLOWED", "External labels without a join key require explicit --label-alignment row_order or --allow-row-order-labels.", join_key=join_key, available=list(rows[0].keys()))
    _require_columns([str(join_key)], list(rows[0].keys()), "label_file.join_key")
    if label_column:
        _require_columns([str(label_column)], list(rows[0].keys()), "label_file.label_column")
    if metadata_columns:
        _require_columns([str(column) for column in metadata_columns], list(rows[0].keys()), "label_file.metadata_columns")

    duplicate_count = _count_duplicate_keys(rows, str(join_key))
    duplicate_policy = alignment.get("duplicate_policy") or "blocked"
    if duplicate_count and duplicate_policy == "blocked":
        raise ApplyBlockedError("DUPLICATE_LABEL_KEYS", "External label file has duplicate join keys.", join_key=join_key, duplicate_count=duplicate_count)

    label_by_key = {str(row.get(str(join_key))): row for row in rows if row.get(str(join_key)) not in {None, ""}}
    if not sample_ids:
        raise ApplyBlockedError("SAMPLE_IDS_REQUIRED_FOR_ALIGNMENT", "External label alignment requires sample_ids from the spectral file.")

    y: list[Any] | None = [] if label_column else None
    existing_metadata = extracted.get("metadata_rows")
    metadata_rows: list[dict[str, Any]] | None = list(existing_metadata) if existing_metadata else ([{} for _ in sample_ids] if metadata_columns else None)
    missing: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        key = str(sample_id)
        label_row = label_by_key.get(key)
        if label_row is None:
            missing.append(key)
            if y is not None:
                y.append("")
            continue
        if y is not None:
            y.append(label_row.get(str(label_column)))
        if metadata_columns:
            if metadata_rows is None:
                metadata_rows = [{} for _ in sample_ids]
            metadata_rows[idx].update({column: label_row.get(column) for column in metadata_columns})

    missing_policy = alignment.get("missing_label_policy") or ("blocked" if plan.get("task_hint") in {"classification", "regression"} else "warning")
    allow_unmatched = bool(alignment.get("allow_unmatched_spectra", False))
    if missing and missing_policy == "blocked" and not allow_unmatched:
        raise ApplyBlockedError("MISSING_REQUIRED_LABELS", "External label file is missing required spectra sample IDs.", missing_sample_ids=missing, missing_count=len(missing))

    extra_label_count = len([key for key in label_by_key if key not in {str(value) for value in sample_ids}])
    updated = dict(extracted)
    updated["y"] = y
    updated["y_name"] = str(label_column) if label_column else None
    updated["metadata_rows"] = metadata_rows
    updated["label_source"] = "external" if label_column else "none"
    updated["external_label_used"] = True
    updated["alignment_summary"] = {
        "alignment_status": "aligned" if not missing else "aligned_with_warnings",
        "label_source": "external" if label_column else "none",
        "external_label_used": True,
        "label_file_ref": str(label_spec.get("path")),
        "resolved_label_file": str(label_path),
        "join_key": str(join_key),
        "alignment_method": method,
        "label_alignment": method,
        "join_type": alignment.get("join_type") or "left",
        "preserve_spectrum_order": alignment.get("preserve_spectrum_order", True),
        "aligned_sample_count": len(sample_ids) - len(missing),
        "spectra_sample_count": len(sample_ids),
        "label_row_count": len(rows),
        "missing_label_count": len(missing),
        "duplicate_label_key_count": duplicate_count,
        "unmatched_label_count": extra_label_count,
    }
    if missing:
        updated.setdefault("warnings", []).append({"code": "MISSING_LABELS_ALLOWED", "message": "Some spectra did not have external labels.", "severity": "warning", "details": {"missing_count": len(missing)}})
    return updated


def _apply_external_metadata_file(extracted: dict[str, Any], plan: dict[str, Any], *, read_plan_path: Path | None) -> dict[str, Any]:
    metadata_spec = plan.get("metadata_file") or {}
    path_report = _resolve_auxiliary_path(metadata_spec.get("path"), plan, read_plan_path)
    metadata_path_value = path_report.get("resolved_path")
    if not metadata_path_value or not path_report.get("exists"):
        raise ApplyBlockedError("METADATA_FILE_NOT_FOUND", "metadata_file.path could not be resolved.", **path_report)
    metadata_path = Path(str(metadata_path_value))
    rows = _read_auxiliary_table(metadata_path, metadata_spec, plan)
    if not rows:
        raise ApplyBlockedError("METADATA_FILE_EMPTY", "External metadata file contains no data rows.", path=str(metadata_path))
    sample_ids = extracted.get("sample_ids")
    if not sample_ids:
        raise ApplyBlockedError("SAMPLE_IDS_REQUIRED_FOR_ALIGNMENT", "External metadata alignment requires sample_ids from the spectral file.")
    join_key = metadata_spec.get("sample_id_column") or (plan.get("alignment_plan") or {}).get("join_key") or _role_column(plan.get("sample_id")) or "sample_id"
    _require_columns([str(join_key)], list(rows[0].keys()), "metadata_file.join_key")
    metadata_columns = list(metadata_spec.get("metadata_columns") or [])
    if not metadata_columns:
        metadata_columns = [column for column in rows[0].keys() if column != str(join_key)]
    _require_columns([str(column) for column in metadata_columns], list(rows[0].keys()), "metadata_file.metadata_columns")
    duplicate_count = _count_duplicate_keys(rows, str(join_key))
    if duplicate_count:
        raise ApplyBlockedError("DUPLICATE_METADATA_KEYS", "External metadata file has duplicate join keys.", join_key=join_key, duplicate_count=duplicate_count)
    metadata_by_key = {str(row.get(str(join_key))): row for row in rows if row.get(str(join_key)) not in {None, ""}}
    metadata_rows = extracted.get("metadata_rows") or [{} for _ in sample_ids]
    missing = []
    for idx, sample_id in enumerate(sample_ids):
        row = metadata_by_key.get(str(sample_id))
        if row is None:
            missing.append(str(sample_id))
            continue
        metadata_rows[idx].update({str(column): row.get(str(column)) for column in metadata_columns})
    if missing:
        raise ApplyBlockedError("MISSING_REQUIRED_METADATA", "External metadata file is missing spectra sample IDs.", missing_sample_ids=missing, missing_count=len(missing))
    updated = dict(extracted)
    updated["metadata_rows"] = metadata_rows
    updated.setdefault("warnings", []).append({"code": "EXTERNAL_METADATA_MERGED", "message": "External metadata file was merged by sample_id.", "severity": "info", "details": {"metadata_file_ref": str(metadata_spec.get("path"))}})
    return updated


def _validate_shapes_and_values(extracted: dict[str, Any], plan: dict[str, Any]) -> None:
    X = extracted["X"]
    if not X or not X[0]:
        raise ApplyBlockedError("EMPTY_X", "X matrix has zero rows or zero columns.")
    n_samples = len(X)
    n_features = len(X[0])
    if any(len(row) != n_features for row in X):
        raise ApplyBlockedError("X_RAGGED", "X rows do not have consistent feature counts.")
    for key in ["sample_ids", "y", "metadata_rows"]:
        values = extracted.get(key)
        if values is not None and len(values) != n_samples:
            raise ApplyBlockedError("ROW_COUNT_MISMATCH", f"{key} row count does not match X.", field=key, expected=n_samples, observed=len(values))
    sample_ids = extracted.get("sample_ids")
    if sample_ids is not None:
        missing = [idx for idx, value in enumerate(sample_ids) if str(value or "").strip() == ""]
        if missing:
            raise ApplyBlockedError("SAMPLE_ID_MISSING", "sample_ids contain missing values; confirm generated replacement IDs before reading.", missing_indices=missing, suggested_arguments={"--allow-generated-sample-ids-for-missing": True})
        duplicates = [value for value in set(sample_ids) if sample_ids.count(value) > 1]
        if duplicates:
            raise ApplyBlockedError("DUPLICATE_SAMPLE_IDS", "sample_ids must be unique for confirmed reader output.", duplicates=duplicates)
    if plan.get("task_hint") == "classification" and extracted.get("y") is None:
        raise ApplyBlockedError("LABEL_REQUIRED_FOR_CLASSIFICATION", "classification read_plan did not produce y.")
    if plan.get("task_hint") == "regression" and extracted.get("y") is None:
        raise ApplyBlockedError("TARGET_REQUIRED_FOR_REGRESSION", "regression read_plan did not produce y.")


def _build_band_axis(plan: dict[str, Any], spectral_columns: list[str]) -> tuple[list[Any], list[dict[str, Any]]]:
    band_spec = plan.get("band_axis") or {}
    warnings: list[dict[str, Any]] = []
    if band_spec.get("source") == "external_file" or plan.get("band_axis_file"):
        values = _load_external_band_axis(plan)
    elif isinstance(band_spec.get("values"), list):
        values = list(band_spec["values"])
    elif band_spec.get("source") == "generated_index":
        values = list(range(len(spectral_columns)))
    else:
        parsed: list[Any] = []
        failed = []
        for column in spectral_columns:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", str(column))
            if match:
                number = float(match.group(0))
                parsed.append(int(number) if number.is_integer() else number)
            else:
                failed.append(column)
        if failed:
            if band_spec.get("required") is True:
                raise ApplyBlockedError("BAND_AXIS_PARSE_FAILED", "Could not parse required band_axis from spectral column names.", columns=failed)
            values = list(range(len(spectral_columns)))
            warnings.append({"code": "BAND_AXIS_FALLBACK_INDEX", "message": "Could not parse all band values; generated index band_axis.", "severity": "warning", "details": {"columns": failed}})
        else:
            values = parsed
    if len(values) != len(spectral_columns):
        raise ApplyBlockedError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=len(spectral_columns), observed=len(values))
    return values, warnings


def _load_external_band_axis(plan: dict[str, Any]) -> list[Any]:
    spec = plan.get("band_axis_file") or {}
    path_report = _resolve_auxiliary_path(spec.get("path"), plan, None)
    resolved = path_report.get("resolved_path")
    if not resolved or not path_report.get("exists"):
        raise ApplyBlockedError("BAND_AXIS_FILE_NOT_FOUND", "band_axis_file could not be resolved.", **path_report)
    path = Path(str(resolved))
    axis_plan = {
        "encoding": spec.get("encoding") or plan.get("encoding") or "utf-8-sig",
        "delimiter": spec.get("delimiter") or (None if path.suffix.lower() in WORKBOOK_SUFFIXES else _delimiter_for_path(path)),
        "skiprows": spec.get("skiprows") or 0,
        "header_row": spec.get("header_row") if spec.get("header_row") is not None else 0,
        "sheet_name": spec.get("sheet_name") or plan.get("sheet_name"),
        "sheet_index": spec.get("sheet_index"),
    }
    rows = _read_workbook_text_table(path, axis_plan) if path.suffix.lower() in WORKBOOK_SUFFIXES else _read_text_table(path, axis_plan)
    if not rows:
        raise ApplyBlockedError("BAND_AXIS_FILE_EMPTY", "band_axis_file contains no data rows.", path=str(path))
    columns = list(rows[0].keys())
    column = spec.get("column") or (plan.get("band_axis") or {}).get("column")
    if column is None:
        if len(columns) == 1:
            column = columns[0]
        else:
            raise ApplyBlockedError("BAND_AXIS_COLUMN_NOT_FOUND", "band_axis_file has multiple columns; specify --band-axis-column.", available=columns)
    if isinstance(column, int) or (isinstance(column, str) and column.isdigit()):
        idx = int(column)
        if idx < 0 or idx >= len(columns):
            raise ApplyBlockedError("BAND_AXIS_COLUMN_NOT_FOUND", "band_axis_file column index is outside header.", column=column, available=columns)
        column = columns[idx]
    column = str(column)
    if column not in columns:
        raise ApplyBlockedError("BAND_AXIS_COLUMN_NOT_FOUND", "band_axis_column does not exist in band_axis_file.", column=column, available=columns)
    return [_parse_axis_value(row.get(column)) for row in rows]


def _load_external_sample_ids(plan: dict[str, Any]) -> tuple[list[str], bool]:
    spec = plan.get("sample_ids_file") or {}
    path_report = _resolve_auxiliary_path(spec.get("path"), plan, None)
    resolved = path_report.get("resolved_path")
    if not resolved or not path_report.get("exists"):
        raise ApplyBlockedError("SAMPLE_IDS_FILE_NOT_FOUND", "sample_ids_file could not be resolved.", **path_report)
    path = Path(str(resolved))
    ids_plan = {
        "encoding": spec.get("encoding") or plan.get("encoding") or "utf-8-sig",
        "delimiter": spec.get("delimiter") or (None if path.suffix.lower() in WORKBOOK_SUFFIXES else _delimiter_for_path(path)),
        "skiprows": spec.get("skiprows") or 0,
        "header_row": spec.get("header_row") if spec.get("header_row") is not None else 0,
        "sheet_name": spec.get("sheet_name") or plan.get("sheet_name"),
        "sheet_index": spec.get("sheet_index"),
    }
    rows = _read_workbook_text_table(path, ids_plan) if path.suffix.lower() in WORKBOOK_SUFFIXES else _read_text_table(path, ids_plan)
    if not rows:
        raise ApplyBlockedError("SAMPLE_IDS_FILE_EMPTY", "sample_ids_file contains no data rows.", path=str(path))
    columns = list(rows[0].keys())
    column = spec.get("column")
    if column is None:
        if "sample_id" in columns:
            column = "sample_id"
        elif len(columns) == 1:
            column = columns[0]
        else:
            raise ApplyBlockedError("SAMPLE_IDS_COLUMN_NOT_FOUND", "sample_ids_file has multiple columns; specify --sample-id-column.", available=columns)
    key = _sample_file_column(columns, column, "sample_id")
    values, filled_missing = _fill_missing_sample_ids([str(row.get(key) or "").strip() for row in rows], plan)
    return values, filled_missing


def _generated_sample_ids(n_samples: int) -> list[str]:
    width = max(3, len(str(n_samples)))
    return [f"sample_{idx:0{width}d}" for idx in range(1, n_samples + 1)]


def _fill_missing_sample_ids(sample_ids: list[Any], plan: dict[str, Any]) -> tuple[list[str], bool]:
    values = [str(value or "").strip() for value in sample_ids]
    missing = [idx for idx, value in enumerate(values) if value == ""]
    if not missing:
        return values, False
    policy = plan.get("missing_sample_id_policy")
    if policy != "generate":
        raise ApplyBlockedError(
            "SAMPLE_ID_MISSING",
            "sample_id column exists but some sample IDs are missing.",
            missing_indices=missing,
            suggested_arguments={"--missing-sample-id-policy": ["generate", "blocked"], "--confirm-generate-missing-sample-ids": True},
        )
    existing = {value for value in values if value}
    width = max(3, len(str(len(missing))))
    for idx in missing:
        generated_index = missing.index(idx) + 1
        candidate = f"generated_sample_{generated_index:0{width}d}"
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"generated_sample_{generated_index:0{width}d}_{suffix}"
        values[idx] = candidate
        existing.add(candidate)
    return values, True


def _build_result(plan: dict[str, Any], output_dir: Path, extracted: dict[str, Any], status: str, warnings: list[dict[str, Any]], strict: bool, path_resolution: dict[str, Any]) -> dict[str, Any]:
    X = extracted["X"]
    return {
        "applied_read_plan_id": plan.get("read_plan_id"),
        "apply_status": status,
        "output_dir": str(output_dir),
        "data_refs": {},
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "has_y": extracted.get("y") is not None,
        "has_sample_ids": extracted.get("sample_ids") is not None,
        "has_metadata": extracted.get("metadata_rows") is not None,
        "missing_value_status": extracted.get("missing_value_status") or _missing_value_status(X),
        "sample_id_status": extracted.get("sample_id_status") or "unknown",
        "sample_id_source": extracted.get("sample_id_source") or "unknown",
        "label_has_missing": bool(extracted.get("label_has_missing")),
        "metadata_has_missing": bool(extracted.get("metadata_has_missing")),
        "sample_orientation": plan.get("sample_orientation"),
        "label_source": extracted.get("label_source") or "none",
        "external_label_used": bool(extracted.get("external_label_used")),
        "alignment_summary": extracted.get("alignment_summary") or _empty_alignment_summary(extracted.get("label_source") or "none", len(X)),
        "band_axis_summary": {
            "count": len(extracted["band_axis"]),
            "unit": plan.get("band_unit"),
            "first": extracted["band_axis"][0] if extracted["band_axis"] else None,
            "last": extracted["band_axis"][-1] if extracted["band_axis"] else None,
        },
        "task_hint": plan.get("task_hint"),
        "original_source_path": path_resolution.get("original_path"),
        "resolved_source_path": path_resolution.get("resolved_path"),
        "source_base_dir": plan.get("source_base_dir"),
        "path_resolution_status": path_resolution.get("status"),
        "path_resolution_notes": {
            "base_dir_used": path_resolution.get("base_dir_used"),
            "attempted_paths": path_resolution.get("attempted_paths", []),
        },
        "warnings": warnings,
        "execution_summary": {
            "read_mode": plan.get("read_mode"),
            "sample_orientation": plan.get("sample_orientation"),
            "label_source": extracted.get("label_source") or "none",
            "alignment_status": (extracted.get("alignment_summary") or {}).get("alignment_status"),
            "file_type": plan.get("file_type"),
            "strict": strict,
            "generated_data_contract": False,
            "generated_package_manifest": False,
        },
    }


def _resolve_spectral_columns(plan: dict[str, Any], columns: list[str]) -> list[str]:
    spec = plan.get("spectral_columns") or {}
    if spec.get("columns"):
        return [_resolve_column_ref(columns, column, "spectral_columns", "COLUMN_NOT_FOUND") for column in spec["columns"]]
    start = spec.get("start")
    end = spec.get("end")
    if start is None or end is None:
        raise ApplyBlockedError("SPECTRAL_COLUMNS_MISSING", "spectral_columns must declare columns or start/end.")
    start_idx = _column_index_for_role(columns, start, default=0, code="COLUMN_RANGE_INVALID")
    end_idx = _column_index_for_role(columns, end, default=len(columns) - 1, code="COLUMN_RANGE_INVALID")
    if start_idx > end_idx:
        raise ApplyBlockedError("COLUMN_RANGE_INVALID", "spectral_start_column appears after spectral_end_column.", start=start, end=end)
    return columns[start_idx : end_idx + 1]


def _uses_spectral_range(plan: dict[str, Any]) -> bool:
    spec = plan.get("spectral_columns") or {}
    layout = plan.get("table_layout") or {}
    return spec.get("start") is not None or spec.get("end") is not None or layout.get("spectral_start_column") is not None or layout.get("spectral_end_column") is not None


def _require_columns(required: list[str], columns: list[str], role: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ApplyBlockedError("COLUMN_NOT_FOUND", f"Declared {role} column does not exist.", role=role, missing=missing, available=columns)


def _resolve_role_column(columns: list[str], value: Any, role: str, *, required: bool = False) -> str | None:
    if value is None or str(value) == "":
        if required:
            raise ApplyBlockedError("COLUMN_NOT_FOUND", f"Declared {role} column does not exist.", role=role, missing=[value], available=columns)
        return None
    return _resolve_column_ref(columns, value, role, "COLUMN_NOT_FOUND")


def _sample_id_source(sample_column: str | None, columns: list[str], plan: dict[str, Any]) -> str:
    if sample_column is None:
        return "generated"
    try:
        index = columns.index(sample_column)
    except ValueError:
        index = None
    if sample_column == "" and index == 0:
        return "source_first_column_empty_header"
    declared = ((plan.get("sample_id") or {}).get("column"))
    if index == 0 and declared in {0, "0"}:
        return "source_first_column"
    return "source_column"


def _resolve_role_columns(columns: list[str], values: list[Any], role: str, *, required: bool = True) -> list[str]:
    resolved: list[str] = []
    missing: list[Any] = []
    for value in values:
        try:
            resolved.append(_resolve_column_ref(columns, value, role, "COLUMN_NOT_FOUND"))
        except ApplyBlockedError:
            missing.append(value)
    if missing and required:
        raise ApplyBlockedError("COLUMN_NOT_FOUND", f"Declared {role} column does not exist.", role=role, missing=missing, available=columns)
    return resolved


def _resolve_column_ref(columns: list[str], value: Any, role: str, code: str) -> str:
    try:
        idx = _column_index_for_role(columns, value, code=code)
        return columns[idx]
    except ApplyBlockedError as exc:
        raise ApplyBlockedError(code, exc.message, role=role, **exc.details) from exc


def _role_column(role: Any) -> str | None:
    if isinstance(role, dict):
        column = role.get("column")
        return str(column) if column is not None and str(column) != "" else None
    return None


def _role_columns(role: Any) -> list[str]:
    if not isinstance(role, dict):
        return []
    columns = role.get("columns")
    if isinstance(columns, list):
        return [str(column) for column in columns if column is not None and str(column) != ""]
    column = role.get("column")
    return [str(column)] if column is not None and str(column) != "" else []


def _role_source(role: Any) -> str | None:
    if isinstance(role, dict):
        source = role.get("source")
        return str(source) if source else None
    return None


def _delimiter_from_plan(plan: dict[str, Any], path: Path) -> str:
    delimiter = plan.get("delimiter")
    if delimiter is not None:
        text = str(delimiter)
        if text.lower() in {"whitespace", "space", "\\s+", "auto_whitespace"}:
            return "whitespace"
        return text
    if path.suffix.lower() == ".tsv":
        return "\t"
    raise ApplyBlockedError("DELIMITER_MISSING", "delimiter must be declared for CSV/TXT apply_read_plan.")


def _delimiter_for_path(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _split_delimited_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.strip().split()
    try:
        return next(csv.reader([line], delimiter=delimiter))
    except Exception:
        return line.split(delimiter)


def _dict_reader(lines: list[str], delimiter: str) -> list[dict[str, str]] | csv.DictReader:
    if delimiter == "whitespace":
        rows = [_split_delimited_line(line, delimiter) for line in lines if line.strip()]
        if not rows:
            return []
        header = rows[0]
        return [{header[idx]: row[idx] if idx < len(row) else "" for idx in range(len(header))} for row in rows[1:]]
    return csv.DictReader(lines, delimiter=delimiter)


def _to_float_value(value: Any, plan: dict[str, Any] | None = None) -> float | str:
    text = str(value).strip()
    if _is_missing_value(text, plan):
        return "nan"
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    number = float(text)
    if math.isinf(number):
        return "nan"
    return number


def _is_missing_value(value: Any, plan: dict[str, Any] | None = None) -> bool:
    text = str(value).strip().lower()
    tokens = {"", "na", "n/a", "nan", "null", "none", "missing", "--", "-", "."}
    for token in (plan or {}).get("missing_value_tokens") or []:
        tokens.add(str(token).strip().lower())
    return text in tokens


def _missing_value_status(X: list[list[Any]]) -> str:
    return "present" if any(str(value).strip().lower() == "nan" for row in X for value in row) else "none"


def _sequence_has_missing(values: Any) -> bool:
    if values is None:
        return False
    if isinstance(values, list):
        for value in values:
            if isinstance(value, list):
                if _sequence_has_missing(value):
                    return True
            elif _is_missing_value(value):
                return True
    return False


def _metadata_has_missing(rows: list[dict[str, Any]] | None) -> bool:
    if not rows:
        return False
    return any(_is_missing_value(value) for row in rows for value in row.values())


def _read_external_table(path: Path, spec: dict[str, Any]) -> list[dict[str, str]]:
    plan = {
        "encoding": spec.get("encoding") or "utf-8-sig",
        "delimiter": spec.get("delimiter"),
        "skiprows": spec.get("skiprows") or 0,
        "header_row": spec.get("header_row"),
    }
    return _read_text_table(path, plan)


def _read_auxiliary_table(path: Path, spec: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, str]]:
    if path.suffix.lower() in WORKBOOK_SUFFIXES:
        aux_plan = {
            **plan,
            "sheet_name": spec.get("sheet_name"),
            "sheet_index": spec.get("sheet_index"),
            "skiprows": spec.get("skiprows") or 0,
            "header_row": spec.get("header_row") if spec.get("header_row") is not None else 0,
        }
        return _read_workbook_text_table(path, aux_plan)
    return _read_external_table(path, spec)


def _read_workbook_label_table(path: Path, spec: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, str]]:
    label_plan = {
        **plan,
        "sheet_name": spec.get("sheet_name"),
        "sheet_index": spec.get("sheet_index"),
        "skiprows": spec.get("skiprows") or 0,
        "header_row": spec.get("header_row") if spec.get("header_row") is not None else 0,
    }
    return _read_workbook_text_table(path, label_plan)


def _sample_files(source_dir: Path, plan: dict[str, Any]) -> list[Path]:
    pattern = str(plan.get("sample_file_pattern") or "*.csv")
    recursive = plan.get("sample_file_recursive")
    recursive = True if recursive is None else bool(recursive)
    candidates = source_dir.rglob("*") if recursive else source_dir.glob("*")
    matched = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and fnmatch.fnmatch(path.relative_to(source_dir).as_posix(), pattern)
    ]
    return sorted(matched, key=lambda path: path.relative_to(source_dir).as_posix())


def _read_sample_file_table(path: Path, plan: dict[str, Any]) -> list[dict[str, str]]:
    sample_spec = plan.get("sample_file_reading") or {}
    file_plan = {
        "encoding": sample_spec.get("encoding") or plan.get("encoding") or "utf-8-sig",
        "delimiter": sample_spec.get("delimiter") or plan.get("delimiter") or ("\t" if path.suffix.lower() in {".tsv", ".txt"} else ","),
        "skiprows": sample_spec.get("skiprows") if sample_spec.get("skiprows") is not None else plan.get("skiprows") or 0,
        "header_row": sample_spec.get("header_row") if sample_spec.get("header_row") is not None else plan.get("header_row"),
    }
    return _read_text_table(path, file_plan)


def _sample_file_column(columns: list[str], value: Any, role: str) -> str:
    missing_code = "SAMPLE_FILE_BAND_COLUMN_NOT_FOUND" if role == "band_axis" else "SAMPLE_FILE_VALUE_COLUMN_NOT_FOUND" if role == "value" else "SAMPLE_FILE_COLUMN_NOT_FOUND"
    if isinstance(value, int):
        if value < 0 or value >= len(columns):
            raise ApplyBlockedError(missing_code, "Sample file column index is outside header.", role=role, column=value, available=columns)
        return columns[value]
    text = str(value)
    if text in columns:
        return text
    if text.isdigit():
        return _sample_file_column(columns, int(text), role)
    raise ApplyBlockedError(missing_code, "Declared sample file column does not exist.", role=role, column=text, available=columns)


def _sample_id_from_file(path: Path, plan: dict[str, Any]) -> str:
    if plan.get("file_name_as_sample_id") is True:
        return path.stem
    return path.name


def _label_from_file(path: Path, source_dir: Path, plan: dict[str, Any]) -> str:
    if _folder_label_enabled(plan):
        relative_parent = path.parent.relative_to(source_dir)
        parts = relative_parent.parts
        if not parts:
            raise ApplyBlockedError("FOLDER_LABEL_NOT_FOUND", "folder_name_as_label requires sample files to be inside class subfolders.", path=str(path))
        return parts[0]
    if _file_label_enabled(plan):
        rule = str(plan.get("file_label_rule") or "prefix_before_underscore")
        if rule == "prefix_before_underscore":
            stem = path.stem
            if "_" not in stem or not stem.split("_", 1)[0]:
                raise ApplyBlockedError("FILE_LABEL_PARSE_FAILED", "file_name_as_label expects a non-empty prefix before underscore.", path=str(path), rule=rule)
            return stem.split("_", 1)[0]
    return path.stem


def _folder_label_enabled(plan: dict[str, Any]) -> bool:
    return plan.get("folder_name_as_label") is True or _role_source(plan.get("label")) == "folder_name"


def _file_label_enabled(plan: dict[str, Any]) -> bool:
    return plan.get("file_name_as_label") is True or _role_source(plan.get("label")) == "file_name"


def _resolve_auxiliary_path(path_value: Any, plan: dict[str, Any], read_plan_path: Path | None) -> dict[str, Any]:
    from .io_utils import resolve_path

    return resolve_path(
        str(path_value) if path_value is not None else None,
        base_dir=plan.get("source_base_dir"),
        read_plan_dir=read_plan_path.parent if read_plan_path else None,
        cwd=Path.cwd(),
    )


def _column_index(header: list[str], value: Any, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ApplyBlockedError("COLUMN_NOT_FOUND", "Required column is missing.")
        return default
    if isinstance(value, int):
        if value < 0 or value >= len(header):
            raise ApplyBlockedError("COLUMN_NOT_FOUND", "Column index is outside header.", index=value, available=header)
        return value
    text = str(value)
    if text.isdigit():
        return _column_index(header, int(text), default=default)
    if text in header:
        return header.index(text)
    raise ApplyBlockedError("COLUMN_NOT_FOUND", "Declared column does not exist.", column=text, available=header)


def _column_index_for_role(header: list[str], value: Any, *, default: int | None = None, code: str = "COLUMN_NOT_FOUND") -> int:
    text = str(value) if value is not None else None
    if text in header:
        return header.index(text)
    normalized = _normalize_column_token(text)
    if normalized is not None:
        for idx, column in enumerate(header):
            if _normalize_column_token(str(column)) == normalized:
                return idx
    try:
        return _column_index(header, value, default=default)
    except ApplyBlockedError as exc:
        raise ApplyBlockedError(code, exc.message, **exc.details) from exc


def _normalize_column_token(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def _column_range(header: list[str], start: Any, end: Any, *, exclude: set[int]) -> list[int]:
    start_idx = _column_index(header, start, default=0)
    end_idx = _column_index(header, end, default=len(header) - 1) if end is not None else len(header) - 1
    if start_idx > end_idx:
        raise ApplyBlockedError("SAMPLE_COLUMN_RANGE_INVALID", "sample_start_column appears after sample_end_column.", start=start, end=end)
    return [idx for idx in range(start_idx, end_idx + 1) if idx not in exclude]


def _parse_axis_value(value: Any) -> Any:
    raw = str(value).strip()
    try:
        num = float(raw)
        return int(num) if num.is_integer() else num
    except ValueError:
        return raw


def _count_duplicate_keys(rows: list[dict[str, str]], key: str) -> int:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _cell_to_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _plan_sheet_name(plan: dict[str, Any]) -> str | None:
    workbook = plan.get("workbook") or {}
    sheet = plan.get("sheet") or {}
    value = plan.get("sheet_name") or workbook.get("spectral_sheet") or workbook.get("selected_sheet") or sheet.get("name")
    return str(value) if value not in {None, ""} else None


def _plan_sheet_index(plan: dict[str, Any]) -> int | None:
    workbook = plan.get("workbook") or {}
    sheet = plan.get("sheet") or {}
    value = workbook.get("sheet_index")
    if value is None:
        value = sheet.get("index")
    if value is None:
        return None
    return int(value)


def _empty_alignment_summary(label_source: str, sample_count: int) -> dict[str, Any]:
    aligned_sources = {"embedded", "folder_name", "file_name"}
    return {
        "alignment_status": "not_applicable" if label_source == "none" else "aligned" if label_source in aligned_sources else "unknown",
        "label_source": label_source,
        "external_label_used": False,
        "aligned_sample_count": sample_count if label_source in aligned_sources else 0,
        "spectra_sample_count": sample_count,
        "missing_label_count": 0,
        "duplicate_label_key_count": 0,
        "unmatched_label_count": 0,
    }


def _base_result(plan: dict[str, Any] | None, output_dir: str | Path | None, status: str) -> dict[str, Any]:
    return {
        "applied_read_plan_id": (plan or {}).get("read_plan_id"),
        "apply_status": status,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "data_refs": {},
        "n_samples": 0,
        "n_features": 0,
        "has_y": False,
        "has_sample_ids": False,
        "has_metadata": False,
        "band_axis_summary": {},
        "task_hint": (plan or {}).get("task_hint"),
        "warnings": [],
        "execution_summary": {"generated_data_contract": False, "generated_package_manifest": False},
    }


def _blocked(message: str, code: str, *, backend: str, plan: dict[str, Any] | None = None, output_dir: str | Path | None = None, **details: Any) -> dict[str, Any]:
    result = _base_result(plan, output_dir, "blocked")
    return error_response("apply_read_plan", message, backend=backend, code=code, result=result, details=details)


class ApplyBlockedError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

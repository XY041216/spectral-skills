"""Read-only preview evidence collector for spectral-reader."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .errors import (
    DECODE_FAILED,
    DEPENDENCY_MISSING,
    EMPTY_FILE,
    INPUT_PATH_NOT_FOUND,
    MALFORMED_FILE,
    PERMISSION_DENIED,
    PREVIEW_LIMIT_REACHED,
    UNSUPPORTED_FILE_TYPE,
)
from .response import error_response, message, ok_response


TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".asc"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".ods"}
NUMPY_SUFFIXES = {".npy", ".npz"}
MAT_SUFFIXES = {".mat"}
HIERARCHICAL_SUFFIXES = {".h5", ".hdf5", ".nc"}
JSON_SUFFIXES = {".json"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | NUMPY_SUFFIXES | MAT_SUFFIXES | HIERARCHICAL_SUFFIXES | JSON_SUFFIXES
COMMENT_PREFIXES = ("#", "//", ";", "[Header]", "[HEADER]")
DELIMITERS = [",", "\t", ";", "|"]
DEFAULT_MAX_AUTO_COLUMNS = 10000
SAMPLE_ID_NAMES = {"sample_id", "sampleid", "sample", "id", "name", "unnamed: 0", "样本编号", "样品编号", "编号"}
LABEL_NAMES = {"label", "class", "category", "type", "group", "类别", "等级", "分类", "品种", "产地"}
TARGET_TOKENS = ("target", "value", "content", "concentration", "amount", "含量", "浓度", "指标")
METADATA_NAMES = {"remark", "remarks", "note", "notes", "no.", "no", "index", "batch", "date", "operator", "备注", "序号", "批次", "日期", "仪器"}
BAND_RE = re.compile(r"^\s*(?:wl_|band_)?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>nm|cm-?1|cm⁻¹)?\s*$", re.IGNORECASE)


def preview_file(
    path: str | None = None,
    *,
    backend: str = "core",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preview a file or folder without producing read plans or data arrays."""

    opts = dict(options or {})
    try:
        result = preview_input(
            path,
            max_lines=int(opts.get("max_lines") or 30),
            max_files=int(opts.get("max_files") or 50),
            max_columns=int(opts.get("max_columns") or opts.get("max_auto_columns") or DEFAULT_MAX_AUTO_COLUMNS),
            encoding=opts.get("encoding"),
        )
    except PermissionError as exc:
        return error_response("preview_file", str(exc), backend=backend, code=PERMISSION_DENIED)
    except Exception as exc:  # keep CLI/MCP JSON-readable
        return error_response("preview_file", str(exc), backend=backend, code=MALFORMED_FILE, details={"exception_type": type(exc).__name__})

    warnings = list(result.pop("warnings", []))
    errors = list(result.pop("errors", []))
    ok = result.get("preview_status") != "blocked" and not errors
    if ok:
        return ok_response("preview_file", result, backend=backend, warnings=warnings)
    return error_response(
        "preview_file",
        errors[0]["message"] if errors else "Preview is blocked.",
        backend=backend,
        code=errors[0]["code"] if errors else "PREVIEW_BLOCKED",
        result=result,
        warnings=warnings,
        details=errors[0].get("details", {}) if errors else {},
    )


def preview_input(
    path: str | None,
    *,
    max_lines: int = 30,
    max_files: int = 50,
    max_columns: int = DEFAULT_MAX_AUTO_COLUMNS,
    encoding: str | None = None,
) -> dict[str, Any]:
    if not path:
        return build_preview_response(None, "unknown", "blocked", errors=[_error(INPUT_PATH_NOT_FOUND, "No input path was provided.")])
    p = Path(path)
    if not p.exists():
        return build_preview_response(str(p), "unknown", "blocked", errors=[_error(INPUT_PATH_NOT_FOUND, f"Input path does not exist: {p}")])
    if p.is_dir():
        return preview_folder(p, max_files=max_files, max_lines=max_lines, max_columns=max_columns, encoding=encoding)
    return preview_single_file(p, max_lines=max_lines, max_columns=max_columns, encoding=encoding)


def preview_folder(
    path: Path,
    *,
    max_files: int,
    max_lines: int,
    max_columns: int,
    encoding: str | None,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    files: list[Path] = []
    try:
        for item in sorted(path.rglob("*")):
            if item.is_file():
                files.append(item)
                if len(files) >= max_files:
                    warnings.append(message(PREVIEW_LIMIT_REACHED, "Folder inventory reached max_files limit.", max_files=max_files))
                    break
    except PermissionError:
        return build_preview_response(str(path), "folder", "blocked", errors=[_error(PERMISSION_DENIED, f"Permission denied while listing folder: {path}")])

    inventory = [file_inventory_item(file, root=path) for file in files]
    primary_candidates = [item for item in inventory if item["suffix"] in sorted(SUPPORTED_SUFFIXES)]
    previews = []
    for candidate in primary_candidates[:3]:
        suffix = candidate["suffix"]
        if suffix in TEXT_SUFFIXES:
            previews.append(preview_text_table(Path(candidate["path"]), max_lines=max_lines, max_columns=max_columns, encoding=encoding))
        elif suffix in EXCEL_SUFFIXES | NUMPY_SUFFIXES | HIERARCHICAL_SUFFIXES | JSON_SUFFIXES:
            previews.append(preview_single_file(Path(candidate["path"]), max_lines=max_lines, max_columns=max_columns, encoding=encoding)["file_previews"][0])

    return build_preview_response(
        str(path),
        "folder",
        "ok" if inventory else "degraded",
        file_inventory=inventory,
        primary_file_candidates=primary_candidates,
        file_previews=previews,
        multi_file_evidence=multi_file_label_evidence(inventory, previews),
        folder_role_candidates=folder_role_candidates(inventory, previews),
        warnings=warnings + ([] if inventory else [message(EMPTY_FILE, "Folder contains no files.", severity="warning")]),
    )


def preview_single_file(path: Path, *, max_lines: int, max_columns: int, encoding: str | None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    inventory = [file_inventory_item(path)]
    if path.stat().st_size == 0:
        return build_preview_response(str(path), "file", "blocked", file_inventory=inventory, errors=[_error(EMPTY_FILE, f"File is empty: {path}")])
    if suffix in TEXT_SUFFIXES:
        preview = preview_text_table(path, max_lines=max_lines, max_columns=max_columns, encoding=encoding)
    elif suffix in EXCEL_SUFFIXES:
        preview = preview_excel(path)
    elif suffix in NUMPY_SUFFIXES:
        preview = preview_numpy(path)
    elif suffix in MAT_SUFFIXES:
        preview = preview_mat(path)
    elif suffix in HIERARCHICAL_SUFFIXES:
        preview = preview_hierarchical(path)
    elif suffix in JSON_SUFFIXES:
        preview = preview_json(path, encoding=encoding)
    else:
        return build_preview_response(str(path), "file", "blocked", file_inventory=inventory, errors=[_error(UNSUPPORTED_FILE_TYPE, f"Unsupported preview file type: {suffix}")])
    if preview.get("preview_status") == "blocked" or preview.get("errors"):
        return build_preview_response(str(path), "file", "blocked", file_inventory=inventory, file_previews=[preview], errors=preview.get("errors") or [_error(MALFORMED_FILE, "File preview failed.")])
    return build_preview_response(str(path), "file", "ok", file_inventory=inventory, primary_file_candidates=inventory, file_previews=[preview])


def preview_text_table(path: Path, *, max_lines: int, max_columns: int, encoding: str | None = None) -> dict[str, Any]:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030", "latin-1"]
    lines: list[str] | None = None
    used_encoding = None
    for enc in encodings:
        if not enc:
            continue
        try:
            with path.open("r", encoding=enc, errors="strict", newline="") as handle:
                lines = []
                for _, line in zip(range(max_lines), handle):
                    lines.append(line.rstrip("\r\n"))
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        return {
            "path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": _size(path),
            "preview_status": "blocked",
            "errors": [_error(DECODE_FAILED, f"Could not decode file with preview encodings: {path}")],
        }
    delimiters = detect_delimiter_candidates(lines)
    delimiter = delimiters[0]["delimiter"] if delimiters else ","
    preamble = detect_leading_preamble(lines, delimiter)
    header_candidates = detect_header_candidates(lines, delimiter)
    best_header = header_candidates[0]["row_index"] if header_candidates else None
    full_column_names = split_line(lines[best_header], delimiter) if best_header is not None and best_header < len(lines) else []
    column_names = full_column_names[:max_columns]
    rows = [
        split_line(line, delimiter)[:max_columns]
        for line in lines[(best_header + 1 if best_header is not None else 0):(best_header + 6 if best_header is not None else min(len(lines), 5))]
    ]
    column_summary = summarize_columns(column_names, rows)
    evidence = column_evidence(column_names)
    samples_as_columns_evidence = detect_samples_as_columns(column_names, rows)
    return {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": _size(path),
        "encoding_hint": used_encoding,
        "line_count_sampled": len(lines),
        "column_count_total": len(full_column_names),
        "column_count_sampled": len(column_names),
        "column_preview_truncated": len(full_column_names) > len(column_names),
        "raw_head_lines": lines,
        "delimiter_candidates": delimiters,
        "leading_preamble_candidates": preamble,
        "header_row_candidates": header_candidates,
        "column_preview": [{"index": idx, "name": name} for idx, name in enumerate(column_names)],
        "row_preview": rows,
        "numeric_column_summary": column_summary["numeric"],
        "string_column_summary": column_summary["string"],
        "empty_column_candidates": column_summary["empty"],
        "band_like_column_evidence": evidence["band_like"],
        "sample_id_like_column_evidence": evidence["sample_id_like"],
        "label_like_column_evidence": evidence["label_like"],
        "metadata_like_column_evidence": evidence["metadata_like"],
        "samples_as_columns_evidence": samples_as_columns_evidence,
        "decision_notes": [
            "Preview evidence only; Agent must build and confirm read_plan.",
            "No X, y, sample_ids, or band_axis were generated.",
        ],
    }


def preview_excel(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    engine = _workbook_engine(suffix)
    dep_error = _workbook_dependency_error(suffix)
    if dep_error:
        return {"path": str(path), "suffix": suffix, "preview_status": "blocked", "errors": [dep_error]}
    try:
        import pandas as pd

        workbook = pd.ExcelFile(path, engine=engine)
        sheet_previews = []
        inventory = []
        for idx, sheet_name in enumerate(workbook.sheet_names):
            raw = pd.read_excel(path, sheet_name=sheet_name, engine=engine, header=None, nrows=8)
            raw = raw.dropna(how="all").dropna(axis=1, how="all")
            head_rows = [[_json_value(value) for value in row] for row in raw.head(5).values.tolist()]
            inventory.append({"sheet_name": sheet_name, "index": idx, "row_count_sampled": int(raw.shape[0]), "column_count_sampled": int(raw.shape[1]), "empty": raw.empty, "head_rows": head_rows})
            if raw.empty:
                sheet_previews.append({"sheet_name": sheet_name, "index": idx, "preview_status": "empty", "head_rows": []})
                continue
            header_row = 0
            column_names = [str(value).strip() for value in raw.iloc[header_row].tolist()]
            data_rows = [[_cell_to_string(value) for value in row] for row in raw.iloc[header_row + 1 : header_row + 6].values.tolist()]
            column_summary = summarize_columns(column_names, data_rows)
            evidence = column_evidence(column_names)
            sheet_previews.append({
                "sheet_name": sheet_name,
                "index": idx,
                "preview_status": "ok",
                "head_rows": head_rows,
                "header_row_candidates": [{"row_index": 0, "field_count": len(column_names), "non_numeric_fields": sum(1 for value in column_names if not _is_number(value)), "score": len(column_names), "line": ",".join(column_names)}],
                "column_preview": [{"index": col_idx, "name": name} for col_idx, name in enumerate(column_names)],
                "row_preview": data_rows,
                "numeric_column_summary": column_summary["numeric"],
                "string_column_summary": column_summary["string"],
                "empty_column_candidates": column_summary["empty"],
                "band_like_column_evidence": evidence["band_like"],
                "sample_id_like_column_evidence": evidence["sample_id_like"],
                "label_like_column_evidence": evidence["label_like"],
                "metadata_like_column_evidence": evidence["metadata_like"],
                "samples_as_columns_evidence": detect_samples_as_columns(column_names, data_rows),
            })
        possible_spectral = [item["sheet_name"] for item in sheet_previews if item.get("band_like_column_evidence") or (item.get("samples_as_columns_evidence") or {}).get("candidate")]
        possible_label = [item["sheet_name"] for item in sheet_previews if item.get("label_like_column_evidence")]
        first = sheet_previews[0] if sheet_previews else {}
        return {
            "path": str(path),
            "suffix": suffix,
            "preview_status": "ok",
            "workbook": {
                "engine": engine,
                "sheet_count": len(workbook.sheet_names),
                "sheet_names": workbook.sheet_names,
                "possible_spectral_sheets": possible_spectral,
                "possible_label_sheets": possible_label,
            },
            "sheet_inventory": inventory,
            "sheet_previews": sheet_previews,
            "header_row_candidates": first.get("header_row_candidates") or [],
            "column_preview": first.get("column_preview") or [],
            "row_preview": first.get("row_preview") or [],
            "numeric_column_summary": first.get("numeric_column_summary") or [],
            "string_column_summary": first.get("string_column_summary") or [],
            "empty_column_candidates": first.get("empty_column_candidates") or [],
            "band_like_column_evidence": first.get("band_like_column_evidence") or [],
            "sample_id_like_column_evidence": first.get("sample_id_like_column_evidence") or [],
            "label_like_column_evidence": first.get("label_like_column_evidence") or [],
            "metadata_like_column_evidence": first.get("metadata_like_column_evidence") or [],
            "samples_as_columns_evidence": first.get("samples_as_columns_evidence") or {"candidate": False},
            "decision_notes": ["Workbook preview only; specify spectral_sheet when multiple candidate sheets exist."],
        }
    except Exception as exc:
        return {"path": str(path), "suffix": suffix, "preview_status": "blocked", "errors": [_error(MALFORMED_FILE, f"Workbook preview failed: {exc}")]}


def preview_numpy(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {"path": str(path), "suffix": path.suffix.lower(), "preview_status": "blocked", "errors": [_error(DEPENDENCY_MISSING, "numpy is required for NPY/NPZ preview.", package="numpy", reason=str(exc))]}
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=True) as data:
                arrays = [_array_summary(name, data[name]) for name in data.files]
            return {"path": str(path), "suffix": ".npz", "array_inventory": arrays, "container_candidates": _container_candidates(arrays)}
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        return {"path": str(path), "suffix": ".npy", "array": _array_summary("__npy_array__", arr)}
    except Exception as exc:
        return {"path": str(path), "suffix": path.suffix.lower(), "preview_status": "blocked", "errors": [_error(MALFORMED_FILE, f"Numpy preview failed: {exc}")]}


def preview_mat(path: Path) -> dict[str, Any]:
    if _is_mat_v73(path):
        return {"path": str(path), "suffix": ".mat", "preview_status": "blocked", "errors": [_error("MAT_V73_NOT_SUPPORTED", "MAT v7.3 is HDF5-based and is not supported in this reader step.")]}
    try:
        import scipy.io
    except Exception as exc:
        return {"path": str(path), "suffix": ".mat", "preview_status": "blocked", "errors": [_error("SCIPY_MISSING", "scipy is required for MAT preview.", package="scipy", reason=str(exc))]}
    try:
        data = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        variables = []
        for name, value in data.items():
            if name.startswith("__"):
                continue
            variables.append(_array_summary(name, value))
        return {"path": str(path), "suffix": ".mat", "variable_inventory": variables, "container_candidates": _container_candidates(variables)}
    except NotImplementedError:
        return {"path": str(path), "suffix": ".mat", "preview_status": "blocked", "errors": [_error("MAT_V73_NOT_SUPPORTED", "MAT v7.3 is HDF5-based and is not supported in this reader step.")]}
    except Exception as exc:
        return {"path": str(path), "suffix": ".mat", "preview_status": "blocked", "errors": [_error(MALFORMED_FILE, f"MAT preview failed: {exc}")]}


def preview_hierarchical(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return preview_hdf5(path)
    return preview_netcdf(path)


def preview_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except Exception as exc:
        return {"path": str(path), "suffix": path.suffix.lower(), "preview_status": "blocked", "errors": [_error("H5PY_MISSING", "h5py is required for HDF5 preview.", package="h5py", reason=str(exc))]}
    try:
        datasets: list[dict[str, Any]] = []
        with h5py.File(path, "r") as handle:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append(_dataset_summary("/" + name.strip("/"), obj))

            handle.visititems(visit)
        return {"path": str(path), "suffix": path.suffix.lower(), "dataset_inventory": datasets, "dataset_candidates": _container_candidates(datasets)}
    except Exception as exc:
        return {"path": str(path), "suffix": path.suffix.lower(), "preview_status": "blocked", "errors": [_error("HDF5_READ_FAILED", f"HDF5 preview failed: {exc}")]}


def preview_netcdf(path: Path) -> dict[str, Any]:
    try:
        import netCDF4
    except Exception as exc:
        return {"path": str(path), "suffix": ".nc", "preview_status": "blocked", "errors": [_error("NETCDF4_MISSING", "netCDF4 is required for NetCDF preview.", package="netCDF4", reason=str(exc))]}
    try:
        variables: list[dict[str, Any]] = []
        dimensions: dict[str, int | None] = {}
        groups: list[str] = []
        with netCDF4.Dataset(path, "r") as handle:
            _collect_netcdf_group(handle, "", variables, dimensions, groups)
        return {
            "path": str(path),
            "suffix": ".nc",
            "groups": groups,
            "dimensions": dimensions,
            "variable_inventory": variables,
            "dataset_candidates": _container_candidates(variables),
        }
    except Exception as exc:
        return {"path": str(path), "suffix": ".nc", "preview_status": "blocked", "errors": [_error("NETCDF_READ_FAILED", f"NetCDF preview failed: {exc}")]}


def preview_json(path: Path, *, encoding: str | None = None) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding=encoding or "utf-8"))
    except UnicodeDecodeError:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"path": str(path), "suffix": ".json", "preview_status": "blocked", "errors": [_error(MALFORMED_FILE, f"JSON preview failed: {exc}")]}
    if isinstance(data, dict):
        top = {"type": "object", "keys": list(data.keys())[:50]}
    elif isinstance(data, list):
        top = {"type": "array", "length_sample": len(data), "first_item_type": type(data[0]).__name__ if data else None}
    else:
        top = {"type": type(data).__name__}
    return {"path": str(path), "suffix": ".json", "json_top_level": top}


def detect_delimiter_candidates(lines: list[str]) -> list[dict[str, Any]]:
    candidates = []
    usable = [line for line in lines if line.strip() and not _is_comment(line)]
    for delimiter in DELIMITERS:
        counts = [len(split_line(line, delimiter)) for line in usable[:10]]
        multi = [count for count in counts if count > 1]
        if multi:
            candidates.append({
                "delimiter": delimiter,
                "display": "\\t" if delimiter == "\t" else delimiter,
                "rows_with_multiple_fields": len(multi),
                "median_field_count": sorted(multi)[len(multi) // 2],
                "field_count_values": counts,
            })
    return sorted(candidates, key=lambda item: (item["rows_with_multiple_fields"], item["median_field_count"]), reverse=True)


def detect_leading_preamble(lines: list[str], delimiter: str) -> list[dict[str, Any]]:
    candidates = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            candidates.append({"row_index": idx, "reason": "leading_blank_line", "line": line})
            continue
        if _is_comment(line):
            candidates.append({"row_index": idx, "reason": "comment_or_header_marker", "line": line})
            continue
        if len(split_line(line, delimiter)) <= 1 and idx < 5:
            candidates.append({"row_index": idx, "reason": "leading_non_tabular_line", "line": line})
            continue
        break
    return candidates


def detect_header_candidates(lines: list[str], delimiter: str) -> list[dict[str, Any]]:
    candidates = []
    for idx, line in enumerate(lines):
        fields = split_line(line, delimiter)
        if len(fields) < 2 or _is_comment(line):
            continue
        non_numeric = sum(1 for field in fields if not _is_number(field))
        next_fields = split_line(lines[idx + 1], delimiter) if idx + 1 < len(lines) else []
        evidence = column_evidence(fields)
        band_like_count = len(evidence["band_like"])
        sample_id_like_count = len(evidence["sample_id_like"])
        label_like_count = len(evidence["label_like"])
        role_score = (band_like_count * 3) + (sample_id_like_count * 4) + label_like_count
        data_row_penalty = 8 if idx > 0 and fields and _looks_like_sample_id(fields[0]) else 0
        score = len(fields) + non_numeric + role_score + (2 if next_fields and len(next_fields) == len(fields) else 0) - data_row_penalty
        if non_numeric:
            candidates.append({"row_index": idx, "field_count": len(fields), "non_numeric_fields": non_numeric, "score": score, "line": line})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def summarize_columns(column_names: list[str], rows: list[list[str]]) -> dict[str, Any]:
    numeric = []
    string = []
    empty = []
    for idx, name in enumerate(column_names):
        values = [row[idx] for row in rows if idx < len(row)]
        nonempty = [value for value in values if str(value).strip()]
        numeric_count = sum(1 for value in nonempty if _is_number(value))
        ratio = numeric_count / len(nonempty) if nonempty else 0
        item = {"index": idx, "name": name, "nonempty_sample_count": len(nonempty), "numeric_ratio": round(ratio, 3)}
        if not nonempty:
            empty.append(item)
        elif ratio >= 0.8:
            numeric.append(item)
        else:
            string.append(item)
    return {"numeric": numeric, "string": string, "empty": empty}


def column_evidence(column_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    evidence = {"band_like": [], "sample_id_like": [], "label_like": [], "metadata_like": []}
    for idx, name in enumerate(column_names):
        normalized = _norm(name)
        band = BAND_RE.match(str(name))
        if band:
            unit = band.group("unit")
            evidence["band_like"].append({"index": idx, "name": name, "reason": "band_like_header", "unit_hint": _normalize_unit(unit)})
        if normalized in SAMPLE_ID_NAMES or "sample" in normalized or "样本" in normalized:
            evidence["sample_id_like"].append({"index": idx, "name": name, "reason": "sample_id_like_name"})
        if normalized in LABEL_NAMES:
            evidence["label_like"].append({"index": idx, "name": name, "reason": "label_like_name"})
        if normalized in METADATA_NAMES:
            evidence["metadata_like"].append({"index": idx, "name": name, "reason": "metadata_like_name"})
        if any(token in normalized for token in TARGET_TOKENS):
            evidence["label_like"].append({"index": idx, "name": name, "reason": "target_like_name"})
    return evidence


def build_preview_response(
    input_path: str | None,
    input_kind: str,
    preview_status: str,
    *,
    file_inventory: list[dict[str, Any]] | None = None,
    primary_file_candidates: list[dict[str, Any]] | None = None,
    file_previews: list[dict[str, Any]] | None = None,
    multi_file_evidence: dict[str, Any] | None = None,
    folder_role_candidates: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "input_kind": input_kind,
        "preview_status": preview_status,
        "file_inventory": file_inventory or [],
        "primary_file_candidates": primary_file_candidates or [],
        "file_previews": file_previews or [],
        "multi_file_evidence": multi_file_evidence or {},
        "folder_role_candidates": folder_role_candidates or {},
        "warnings": warnings or [],
        "errors": errors or [],
    }


def detect_samples_as_columns(column_names: list[str], rows: list[list[str]]) -> dict[str, Any]:
    if not column_names or not rows:
        return {"candidate": False}
    first_values = [row[0] for row in rows if row]
    first_numeric = [_to_float(value) for value in first_values if str(value).strip()]
    first_column_numeric_ratio = len(first_numeric) / len(first_values) if first_values else 0
    first_column_monotonic = _numeric_monotonic(first_numeric)
    data_cells = []
    for row in rows:
        data_cells.extend(row[1:])
    numeric_data = [_to_float(value) for value in data_cells if str(value).strip()]
    data_numeric_ratio = len(numeric_data) / len(data_cells) if data_cells else 0
    sample_like_headers = [name for name in column_names[1:] if _looks_like_sample_id(name)]
    candidate = first_column_numeric_ratio >= 0.8 and data_numeric_ratio >= 0.8 and len(column_names) > 2
    return {
        "candidate": candidate,
        "band_axis_column_candidate": column_names[0],
        "first_column_numeric_ratio": round(first_column_numeric_ratio, 3),
        "first_column_monotonic": first_column_monotonic,
        "sample_columns_numeric_ratio": round(data_numeric_ratio, 3),
        "sample_id_like_headers": sample_like_headers,
        "row_count_sampled": len(rows),
        "sample_column_count_sampled": max(len(column_names) - 1, 0),
        "reason": "first column is numeric/monotonic and following columns are numeric" if candidate else "insufficient evidence for samples-as-columns",
    }


def multi_file_label_evidence(inventory: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    text_files = [item for item in inventory if item.get("suffix") in sorted(TEXT_SUFFIXES)]
    label_candidates = []
    spectra_candidates = []
    for item in text_files:
        name = str(item.get("name") or "").lower()
        if any(token in name for token in ["label", "target", "class", "meta"]):
            label_candidates.append(item)
        else:
            spectra_candidates.append(item)
    shared_sample_id_columns = []
    for preview in previews:
        for item in preview.get("sample_id_like_column_evidence") or []:
            if item.get("name") not in shared_sample_id_columns:
                shared_sample_id_columns.append(item.get("name"))
    label_like_columns = []
    metadata_like_columns = []
    for preview in previews:
        label_like_columns.extend(preview.get("label_like_column_evidence") or [])
        metadata_like_columns.extend(preview.get("metadata_like_column_evidence") or [])
    return {
        "file_inventory_count": len(inventory),
        "possible_spectra_files": spectra_candidates[:5],
        "possible_label_files": label_candidates[:5],
        "shared_sample_id_column_candidates": shared_sample_id_columns,
        "label_like_columns": label_like_columns[:20],
        "metadata_like_columns": metadata_like_columns[:20],
    }


def folder_role_candidates(inventory: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    preview_by_path = {str(item.get("path")): item for item in previews if item.get("path")}
    candidates: dict[str, list[dict[str, Any]]] = {"spectra": [], "label": [], "metadata": [], "band_axis": []}
    for item in inventory:
        suffix = str(item.get("suffix") or "")
        if suffix not in sorted(SUPPORTED_SUFFIXES):
            continue
        name = str(item.get("name") or "").lower()
        rel = str(item.get("relative_path") or item.get("name") or "")
        preview = preview_by_path.get(str(item.get("path"))) or {}
        base = {"path": item.get("path"), "relative_path": rel, "name": item.get("name"), "suffix": suffix}
        if any(token in name for token in ["label", "labels", "target", "targets", "class", "classes", "y"]):
            candidates["label"].append({**base, "reason": "label_like_filename"})
            continue
        if any(token in name for token in ["metadata", "meta", "sample_info", "sample-info", "sampleinfo"]):
            candidates["metadata"].append({**base, "reason": "metadata_like_filename"})
            continue
        if any(token in name for token in ["band", "bands", "wavelength", "wavelengths", "wavenumber", "wavenumbers"]):
            candidates["band_axis"].append({**base, "reason": "band_axis_like_filename"})
            continue
        numeric_cols = len(preview.get("numeric_column_summary") or [])
        has_band_columns = bool(preview.get("band_like_column_evidence"))
        has_sample_id = bool(preview.get("sample_id_like_column_evidence"))
        sample_columns = bool((preview.get("samples_as_columns_evidence") or {}).get("candidate"))
        if has_band_columns or sample_columns or (has_sample_id and numeric_cols >= 2) or numeric_cols >= 3:
            candidates["spectra"].append({
                **base,
                "reason": "spectral_matrix_evidence",
                "numeric_column_count": numeric_cols,
                "has_band_columns": has_band_columns,
                "samples_as_columns": sample_columns,
            })
    return candidates


def file_inventory_item(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    rel = str(path.relative_to(root)) if root else path.name
    return {"path": str(path), "relative_path": rel, "suffix": path.suffix.lower(), "size_bytes": _size(path), "name": path.name}


def split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.strip().split()
    try:
        return next(csv.reader([line], delimiter=delimiter))
    except Exception:
        return line.split(delimiter)


def _is_comment(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in COMMENT_PREFIXES)


def _is_number(value: Any) -> bool:
    try:
        text = str(value).strip()
        if not text:
            return False
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        float(text)
        return True
    except Exception:
        return False


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


def _looks_like_sample_id(value: Any) -> bool:
    normalized = _norm(value)
    return normalized in SAMPLE_ID_NAMES or normalized.startswith("s") or "sample" in normalized


def _to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        return float(text)
    except Exception:
        return None


def _numeric_monotonic(values: list[float | None]) -> str:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return "unknown"
    if all(a < b for a, b in zip(clean, clean[1:])):
        return "increasing"
    if all(a > b for a, b in zip(clean, clean[1:])):
        return "decreasing"
    return "non_monotonic"


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return "numeric_header"
    unit = unit.lower().replace("cm⁻¹", "cm-1").replace("cm1", "cm-1")
    return "cm-1" if "cm" in unit else unit


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _error(code: str, text: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": text, "severity": "error", "details": details}


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


def _workbook_engine(suffix: str) -> str:
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    if suffix == ".ods":
        return "odf"
    return "openpyxl"


def _workbook_dependency_error(suffix: str) -> dict[str, Any] | None:
    import importlib.util

    if suffix in {".xlsx", ".xlsm"} and importlib.util.find_spec("openpyxl") is None:
        return _error("EXCEL_ENGINE_MISSING", "openpyxl is required for .xlsx/.xlsm preview.", package="openpyxl")
    if suffix == ".xls" and importlib.util.find_spec("xlrd") is None:
        return _error("EXCEL_ENGINE_MISSING", "xlrd is required for .xls preview.", package="xlrd")
    if suffix == ".ods" and importlib.util.find_spec("odf") is None:
        return _error("ODS_ENGINE_MISSING", "odfpy is required for .ods preview.", package="odfpy")
    if importlib.util.find_spec("pandas") is None:
        return _error(DEPENDENCY_MISSING, "pandas is required for workbook preview.", package="pandas")
    return None


def _array_summary(name: str, array: Any) -> dict[str, Any]:
    try:
        import numpy as np

        arr = np.asarray(array)
        numeric = bool(np.issubdtype(arr.dtype, np.number)) and not np.issubdtype(arr.dtype, np.complexfloating)
        string_like = bool(np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_) or arr.dtype == object)
        return {
            "name": str(name),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "ndim": int(arr.ndim),
            "numeric": numeric,
            "string_like": string_like,
            "can_be_X": bool(arr.ndim == 2 and numeric),
            "can_be_vector": bool(arr.ndim in {1, 2} and 1 in arr.shape),
        }
    except Exception:
        return {"name": str(name), "shape": [], "dtype": type(array).__name__, "ndim": None, "numeric": False, "string_like": False, "can_be_X": False, "can_be_vector": False}


def _dataset_summary(name: str, dataset: Any) -> dict[str, Any]:
    try:
        import numpy as np

        dtype = np.dtype(dataset.dtype)
        numeric = bool(np.issubdtype(dtype, np.number)) and not np.issubdtype(dtype, np.complexfloating)
        string_like = bool(np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_) or dtype == object)
        shape = list(dataset.shape)
        ndim = len(shape)
        return {
            "name": str(name),
            "path": str(name),
            "shape": shape,
            "dtype": str(dtype),
            "ndim": ndim,
            "numeric": numeric,
            "string_like": string_like,
            "can_be_X": bool(ndim == 2 and numeric),
            "can_be_vector": bool(ndim in {1, 2} and (not shape or 1 in shape)),
        }
    except Exception:
        return {"name": str(name), "path": str(name), "shape": [], "dtype": "unknown", "ndim": None, "numeric": False, "string_like": False, "can_be_X": False, "can_be_vector": False}


def _container_candidates(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    x_candidates = [item["name"] for item in items if item.get("can_be_X")]
    vector_candidates = [item["name"] for item in items if item.get("can_be_vector")]
    sample_candidates = [name for name in vector_candidates if any(token in name.lower() for token in ["sample", "id"])]
    band_candidates = [name for name in vector_candidates if any(token in name.lower() for token in ["band", "axis", "wave", "wavelength", "wavenumber"])]
    y_candidates = [name for name in vector_candidates if any(token in name.lower() for token in ["y", "label", "class", "target"])]
    metadata_candidates = [item["name"] for item in items if item.get("ndim") == 2 and not item.get("can_be_X")]
    return {
        "X": x_candidates,
        "y": y_candidates,
        "sample_ids": sample_candidates,
        "band_axis": band_candidates,
        "metadata": metadata_candidates,
    }


def _collect_netcdf_group(group: Any, prefix: str, variables: list[dict[str, Any]], dimensions: dict[str, int | None], groups: list[str]) -> None:
    group_path = prefix or "/"
    groups.append(group_path)
    for name, dim in group.dimensions.items():
        dim_path = f"{group_path.rstrip('/')}/{name}" if group_path != "/" else name
        try:
            dimensions[dim_path] = None if dim.isunlimited() else len(dim)
        except Exception:
            dimensions[dim_path] = None
    for name, var in group.variables.items():
        var_path = f"{group_path.rstrip('/')}/{name}" if group_path != "/" else name
        variables.append(_dataset_summary(var_path, var))
    for name, child in group.groups.items():
        child_prefix = f"{group_path.rstrip('/')}/{name}" if group_path != "/" else name
        _collect_netcdf_group(child, child_prefix, variables, dimensions, groups)


def _is_mat_v73(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(256)
        return b"MATLAB 7.3 MAT-file" in head or head.startswith(b"\x89HDF")
    except OSError:
        return False


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return "numeric_header"
    normalized = unit.lower().replace("cm1", "cm-1")
    return "cm-1" if "cm" in normalized else normalized

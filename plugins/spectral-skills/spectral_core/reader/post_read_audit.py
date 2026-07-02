"""Post-read sanity audit against the original table structure."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


class PostReadAuditBlocked(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def audit_post_read(source_path: Path, plan: dict[str, Any], extracted: dict[str, Any], band_axis: list[Any]) -> list[dict[str, Any]]:
    """Block suspicious successful reads before downstream contracts are built."""

    warnings: list[dict[str, Any]] = []
    X = extracted.get("X") or []
    spectral_columns = [str(value) for value in extracted.get("spectral_columns") or []]
    n_features = len(X[0]) if X else 0
    if not n_features:
        raise PostReadAuditBlocked("POST_READ_EMPTY_X", "Post-read audit found zero spectral features.")

    _audit_band_axis(band_axis, n_features)
    warnings.extend(_audit_text_or_workbook_header(source_path, plan, spectral_columns, n_features))
    return warnings


def _audit_band_axis(band_axis: list[Any], n_features: int) -> None:
    if len(band_axis) != n_features:
        raise PostReadAuditBlocked(
            "POST_READ_BAND_AXIS_LENGTH_MISMATCH",
            "band_axis length does not match X feature count after reading.",
            expected=n_features,
            observed=len(band_axis),
        )
    values = [_to_float(value) for value in band_axis]
    numeric = [value for value in values if value is not None]
    if len(numeric) < 3 or len(numeric) != len(values):
        return
    increasing = all(a < b for a, b in zip(numeric, numeric[1:]))
    decreasing = all(a > b for a, b in zip(numeric, numeric[1:]))
    if not (increasing or decreasing):
        raise PostReadAuditBlocked(
            "POST_READ_BAND_AXIS_NOT_MONOTONIC",
            "Numeric band_axis is not strictly monotonic.",
            first=numeric[0],
            last=numeric[-1],
            n_features=n_features,
        )


def _audit_text_or_workbook_header(source_path: Path, plan: dict[str, Any], spectral_columns: list[str], n_features: int) -> list[dict[str, Any]]:
    if plan.get("read_mode") != "matrix_file" or plan.get("sample_orientation") != "rows":
        return []
    suffix = source_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        header, data_rows = _read_text_header_and_rows(source_path, plan)
    elif suffix in {".xlsx", ".xls", ".xlsm", ".ods"}:
        header, data_rows = _read_workbook_header_and_rows(source_path, plan)
    else:
        return []
    if not header:
        return []

    role_columns = _declared_role_columns(plan)
    numeric_band_columns = _numeric_band_header_columns(header, data_rows, role_columns)
    if not numeric_band_columns:
        return []

    decision_source = str((plan.get("table_layout") or {}).get("spectral_columns_decision_source") or "")
    if decision_source != "user_specified":
        candidate_count = len(numeric_band_columns)
        far_below = candidate_count >= 100 and n_features < max(50, int(candidate_count * 0.5))
        gap = candidate_count - n_features
        if far_below and gap >= 20:
            raise PostReadAuditBlocked(
                "POST_READ_FEATURE_COUNT_SUSPICIOUS",
                "Post-read audit found many numeric band-like columns in the source header, but X contains far fewer features.",
                source_numeric_band_columns=candidate_count,
                extracted_features=n_features,
                suggested_arguments={
                    "--spectral-start-column": numeric_band_columns[0],
                    "--spectral-end-column": numeric_band_columns[-1],
                },
            )

    selected = set(spectral_columns)
    role_overlap = sorted(selected & role_columns)
    if role_overlap:
        raise PostReadAuditBlocked(
            "POST_READ_ROLE_COLUMN_IN_X",
            "Declared sample, label, target, or metadata columns are still present in X.",
            columns=role_overlap,
        )

    return []


def _read_text_header_and_rows(source_path: Path, plan: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    delimiter = str(plan.get("delimiter") or ("\t" if source_path.suffix.lower() == ".tsv" else ","))
    if delimiter.lower() in {"whitespace", "space", "\\s+", "auto_whitespace"}:
        delimiter = " "
    encoding = str(plan.get("encoding") or "utf-8-sig")
    header_row = int(plan.get("header_row") if plan.get("header_row") is not None else 0)
    with source_path.open("r", encoding=encoding, newline="") as handle:
        lines = handle.readlines()
    if header_row < 0 or header_row >= len(lines):
        return [], []
    if delimiter == " ":
        header = lines[header_row].strip().split()
        rows = [line.strip().split() for line in lines[header_row + 1 : header_row + 6] if line.strip()]
    else:
        header = next(csv.reader([lines[header_row]], delimiter=delimiter))
        rows = list(csv.reader(lines[header_row + 1 : header_row + 6], delimiter=delimiter))
    return [str(value).strip() for value in header], rows


def _read_workbook_header_and_rows(source_path: Path, plan: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    try:
        import pandas as pd
    except ImportError:
        return [], []
    sheet = (plan.get("workbook") or {}).get("spectral_sheet") or plan.get("sheet_name") or 0
    header_row = int(plan.get("header_row") if plan.get("header_row") is not None else 0)
    try:
        df = pd.read_excel(source_path, sheet_name=sheet, header=None, nrows=header_row + 6)
    except Exception:
        return [], []
    if header_row < 0 or header_row >= len(df):
        return [], []
    rows = df.fillna("").astype(str).values.tolist()
    return [str(value).strip() for value in rows[header_row]], rows[header_row + 1 : header_row + 6]


def _numeric_band_header_columns(header: list[str], data_rows: list[list[str]], excluded: set[str]) -> list[str]:
    candidates: list[str] = []
    for idx, name in enumerate(header):
        text = str(name).strip()
        if not text or text in excluded or _to_float(text) is None:
            continue
        values = [row[idx] for row in data_rows if idx < len(row) and str(row[idx]).strip()]
        if not values:
            continue
        numeric = sum(1 for value in values if _to_float(value) is not None or _is_known_missing(value))
        if numeric / len(values) >= 0.8:
            candidates.append(text)
    return candidates


def _declared_role_columns(plan: dict[str, Any]) -> set[str]:
    columns: set[str] = set()
    for role_name in ["sample_id", "label", "target", "metadata"]:
        role = plan.get(role_name) or {}
        if isinstance(role, dict):
            if role.get("column") is not None:
                columns.add(str(role["column"]))
            for value in role.get("columns") or []:
                columns.add(str(value))
    return {value for value in columns if value}


def _to_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _is_known_missing(value: Any) -> bool:
    return str(value).strip().lower() in {"", "na", "n/a", "nan", "null", "none", "missing", "--", "-", "."}

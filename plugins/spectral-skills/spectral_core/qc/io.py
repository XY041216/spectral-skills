"""Read spectral-reader standard packages for QC tools."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file


MISSING_TOKENS = {"", "na", "nan", "null", "none", "missing"}


class QCInputError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class SpectralQCPackage:
    root: Path
    contract: dict[str, Any]
    X: list[list[float | None]]
    feature_names: list[str]
    sample_ids: list[str]
    band_axis: list[Any]
    band_axis_header: list[str]
    y: list[list[Any]] | None
    y_header: list[str]
    metadata: list[dict[str, Any]] | None

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def load_standard_package(package_dir: str | Path) -> SpectralQCPackage:
    root = Path(package_dir)
    if not root.exists():
        raise QCInputError("PACKAGE_DIR_NOT_FOUND", "QC package directory does not exist.", path=str(root))
    contract_path = root / "data_contract.json"
    if not contract_path.exists():
        raise QCInputError("DATA_CONTRACT_MISSING", "data_contract.json is required for spectral QC.", path=str(contract_path))
    contract = load_json_file(contract_path)
    files = dict(contract.get("files") or {})

    x_path = _resolve_required(root, files.get("X") or contract.get("X") or "X.csv", "X")
    band_path = _resolve_required(root, files.get("band_axis") or contract.get("band_axis_ref") or "band_axis.csv", "band_axis")
    sample_path = _resolve_optional(root, files.get("sample_ids") or contract.get("sample_ids") or "sample_ids.csv")
    y_path = _resolve_optional(root, files.get("y") or contract.get("y"))
    metadata_path = _resolve_optional(root, files.get("metadata") or contract.get("metadata"))

    feature_names, X = _read_X(x_path)
    sample_ids = _read_single_column(sample_path, "sample_id") if sample_path else [f"sample_{idx + 1}" for idx in range(len(X))]
    band_header, band_axis = _read_band_axis(band_path)
    y_header, y = _read_table(y_path) if y_path else ([], None)
    metadata = _read_dict_table(metadata_path) if metadata_path else None

    _assert_shape(X, feature_names, sample_ids, band_axis, y, metadata)
    return SpectralQCPackage(
        root=root,
        contract=contract,
        X=X,
        feature_names=feature_names,
        sample_ids=sample_ids,
        band_axis=band_axis,
        band_axis_header=band_header,
        y=y,
        y_header=y_header,
        metadata=metadata,
    )


def numeric_matrix(package: SpectralQCPackage, *, fill: str = "median") -> list[list[float]]:
    if fill in {"linear", "nearest"}:
        return [_fill_row_by_axis(row, fill=fill) for row in package.X]
    columns = list(zip(*package.X)) if package.X else []
    fill_values: list[float] = []
    for col in columns:
        observed = [float(value) for value in col if value is not None and math.isfinite(float(value))]
        if not observed:
            fill_values.append(0.0)
        elif fill == "mean":
            fill_values.append(sum(observed) / len(observed))
        elif fill == "zero":
            fill_values.append(0.0)
        else:
            fill_values.append(_median(observed))
    return [[fill_values[idx] if value is None else float(value) for idx, value in enumerate(row)] for row in package.X]


def _fill_row_by_axis(row: list[float | None], *, fill: str) -> list[float]:
    observed = [(idx, float(value)) for idx, value in enumerate(row) if value is not None and math.isfinite(float(value))]
    if not observed:
        return [0.0 for _ in row]
    result: list[float] = []
    for idx, value in enumerate(row):
        if value is not None and math.isfinite(float(value)):
            result.append(float(value))
            continue
        left = next(((j, v) for j, v in reversed(observed) if j < idx), None)
        right = next(((j, v) for j, v in observed if j > idx), None)
        if left is None and right is None:
            result.append(0.0)
        elif left is None:
            result.append(right[1])
        elif right is None:
            result.append(left[1])
        elif fill == "nearest":
            result.append(left[1] if idx - left[0] <= right[0] - idx else right[1])
        else:
            span = right[0] - left[0]
            weight = (idx - left[0]) / span if span else 0.0
            result.append(left[1] + (right[1] - left[1]) * weight)
    return result


def _resolve_required(root: Path, ref: Any, role: str) -> Path:
    path = _resolve_optional(root, ref)
    if path is None or not path.exists():
        raise QCInputError("STANDARD_FILE_MISSING", f"{role} file is missing from the standard package.", role=role, ref=ref)
    return path


def _resolve_optional(root: Path, ref: Any) -> Path | None:
    if ref in {None, ""}:
        return None
    path = Path(str(ref))
    candidate = path if path.is_absolute() else root / path
    return candidate if candidate.exists() else None


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _read_X(path: Path) -> tuple[list[str], list[list[float | None]]]:
    rows = _read_csv(path)
    if not rows:
        raise QCInputError("X_EMPTY", "X.csv is empty.", path=str(path))
    header = [str(value) for value in rows[0]]
    matrix: list[list[float | None]] = []
    for row_idx, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise QCInputError("X_ROW_WIDTH_MISMATCH", "X.csv row width does not match header.", row=row_idx, expected=len(header), observed=len(row))
        matrix.append([_to_float_or_none(value) for value in row])
    if not matrix or not header:
        raise QCInputError("X_EMPTY", "X.csv must contain at least one sample and one feature.", path=str(path))
    return header, matrix


def _read_band_axis(path: Path) -> tuple[list[str], list[Any]]:
    rows = _read_csv(path)
    if not rows:
        raise QCInputError("BAND_AXIS_EMPTY", "band_axis.csv is empty.", path=str(path))
    header = rows[0]
    value_index = 1 if len(header) > 1 and header[1].lower() == "value" else 0
    values = [row[value_index] if len(row) > value_index else "" for row in rows[1:]]
    return header, values


def _read_single_column(path: Path, default_name: str) -> list[str]:
    header, rows = _read_table(path)
    if not rows:
        return []
    return [str(row[0]) for row in rows]


def _read_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_dict_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assert_shape(
    X: list[list[float | None]],
    feature_names: list[str],
    sample_ids: list[str],
    band_axis: list[Any],
    y: list[list[Any]] | None,
    metadata: list[dict[str, Any]] | None,
) -> None:
    n_samples = len(X)
    n_features = len(feature_names)
    if len(sample_ids) != n_samples:
        raise QCInputError("SAMPLE_IDS_ROW_MISMATCH", "sample_ids row count must equal X sample count.", expected=n_samples, observed=len(sample_ids))
    if len(band_axis) != n_features:
        raise QCInputError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=n_features, observed=len(band_axis))
    if y is not None and len(y) != n_samples:
        raise QCInputError("Y_ROW_MISMATCH", "y row count must equal X sample count.", expected=n_samples, observed=len(y))
    if metadata is not None and len(metadata) != n_samples:
        raise QCInputError("METADATA_ROW_MISMATCH", "metadata row count must equal X sample count.", expected=n_samples, observed=len(metadata))


def _to_float_or_none(value: Any) -> float | None:
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise QCInputError("X_NON_NUMERIC", "X.csv contains non-numeric spectral values.", value=text) from exc
    if not math.isfinite(number):
        return None
    return number


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0

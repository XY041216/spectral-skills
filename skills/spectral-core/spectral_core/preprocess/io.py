"""Read standard spectral packages and split contracts for preprocessing."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file
from spectral_core.splitter.contract_reader import SplitContractError, SplitPartition, load_split_contract_info


MISSING_TOKENS = {"", "na", "nan", "null", "none", "missing"}


class PreprocessInputError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class PreprocessPackage:
    root: Path
    contract_path: Path
    contract: dict[str, Any]
    X: list[list[float]]
    feature_names: list[str]
    sample_ids: list[str]
    band_axis_header: list[str]
    band_axis_rows: list[list[Any]]
    y_header: list[str]
    y_rows: list[list[Any]] | None
    metadata_header: list[str]
    metadata_rows: list[list[Any]] | None

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass
class SplitInfo:
    path: Path | None
    contract: dict[str, Any] | None
    assignments: dict[str, list[int]]
    split_type: str = "holdout"
    method: str | None = None
    partitions: list[SplitPartition] | None = None
    folds: list[dict[str, Any]] | None = None
    repeats: list[dict[str, Any]] | None = None

    @property
    def train_indices(self) -> list[int]:
        return self.assignments.get("train", [])


def load_preprocess_package(package_dir: str | Path) -> PreprocessPackage:
    root = Path(package_dir)
    if not root.exists():
        raise PreprocessInputError("PACKAGE_DIR_NOT_FOUND", "Standard package directory does not exist.", path=str(root))
    contract_path = root / "data_contract.json"
    if not contract_path.exists():
        raise PreprocessInputError("DATA_CONTRACT_MISSING", "data_contract.json is required for spectral preprocessing.", path=str(contract_path))

    contract = load_json_file(contract_path)
    files = dict(contract.get("files") or {})
    x_path = _resolve_required(root, files.get("X") or contract.get("X") or "X.csv", "X")
    sample_path = _resolve_required(root, files.get("sample_ids") or contract.get("sample_ids") or "sample_ids.csv", "sample_ids")
    band_path = _resolve_required(root, files.get("band_axis") or contract.get("band_axis_ref") or "band_axis.csv", "band_axis")
    y_path = _resolve_optional(root, files.get("y") or contract.get("y"))
    metadata_path = _resolve_optional(root, files.get("metadata") or contract.get("metadata"))

    feature_names, X = _read_X(x_path)
    sample_ids = _read_single_column(sample_path)
    _assert_unique_sample_ids(sample_ids)
    band_header, band_rows = _read_table(band_path)
    y_header, y_rows = _read_table(y_path) if y_path else ([], None)
    metadata_header, metadata_rows = _read_table(metadata_path) if metadata_path else ([], None)

    _assert_shape(X, feature_names, sample_ids, band_rows, y_rows, metadata_rows)
    return PreprocessPackage(
        root=root,
        contract_path=contract_path,
        contract=contract,
        X=X,
        feature_names=feature_names,
        sample_ids=sample_ids,
        band_axis_header=band_header,
        band_axis_rows=band_rows,
        y_header=y_header,
        y_rows=y_rows,
        metadata_header=metadata_header,
        metadata_rows=metadata_rows,
    )


def load_split_info(split_contract: str | Path | None, package: PreprocessPackage) -> SplitInfo:
    if split_contract is None:
        return SplitInfo(path=None, contract=None, assignments={})
    try:
        info = load_split_contract_info(split_contract, n_samples=package.n_samples, sample_ids=package.sample_ids)
    except SplitContractError as exc:
        raise PreprocessInputError(exc.code, exc.message, **exc.details) from exc
    return SplitInfo(
        path=info.path,
        contract=info.contract,
        assignments=info.assignments,
        split_type=info.split_type,
        method=info.method,
        partitions=info.partitions,
        folds=info.folds,
        repeats=info.repeats,
    )


def _resolve_required(root: Path, ref: Any, role: str) -> Path:
    path = _resolve_optional(root, ref)
    if path is None or not path.exists():
        raise PreprocessInputError("STANDARD_FILE_MISSING", f"{role} file is missing from the standard package.", role=role, ref=ref)
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


def _read_X(path: Path) -> tuple[list[str], list[list[float]]]:
    rows = _read_csv(path)
    if not rows:
        raise PreprocessInputError("X_EMPTY", "X.csv is empty.", path=str(path))
    header = [str(value) for value in rows[0]]
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise PreprocessInputError("X_ROW_WIDTH_MISMATCH", "X.csv row width does not match header.", row=row_idx, expected=len(header), observed=len(row))
        matrix.append([_to_float(value) for value in row])
    if not matrix or not header:
        raise PreprocessInputError("X_EMPTY", "X.csv must contain at least one sample and one feature.", path=str(path))
    return header, matrix


def _read_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_single_column(path: Path) -> list[str]:
    _, rows = _read_table(path)
    return [str(row[0]).strip() if row else "" for row in rows]


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assert_unique_sample_ids(sample_ids: list[str]) -> None:
    if any(sample_id == "" for sample_id in sample_ids):
        raise PreprocessInputError("SAMPLE_ID_EMPTY", "sample_ids.csv contains empty sample IDs.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen and sample_id not in duplicates:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise PreprocessInputError("SAMPLE_ID_DUPLICATE", "sample_ids must be unique before preprocessing.", sample_ids=duplicates)


def _assert_shape(
    X: list[list[float]],
    feature_names: list[str],
    sample_ids: list[str],
    band_rows: list[list[Any]],
    y_rows: list[list[Any]] | None,
    metadata_rows: list[list[Any]] | None,
) -> None:
    n_samples = len(X)
    n_features = len(feature_names)
    if len(sample_ids) != n_samples:
        raise PreprocessInputError("SAMPLE_IDS_ROW_MISMATCH", "sample_ids row count must equal X sample count.", expected=n_samples, observed=len(sample_ids))
    if len(band_rows) != n_features:
        raise PreprocessInputError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=n_features, observed=len(band_rows))
    if y_rows is not None and len(y_rows) != n_samples:
        raise PreprocessInputError("Y_ROW_MISMATCH", "y row count must equal X sample count.", expected=n_samples, observed=len(y_rows))
    if metadata_rows is not None and len(metadata_rows) != n_samples:
        raise PreprocessInputError("METADATA_ROW_MISMATCH", "metadata row count must equal X sample count.", expected=n_samples, observed=len(metadata_rows))


def _to_float(value: Any) -> float:
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        raise PreprocessInputError("X_MISSING_VALUES", "X.csv contains missing values; handle them in spectral-qc before preprocessing.")
    try:
        number = float(text)
    except ValueError as exc:
        raise PreprocessInputError("X_NON_NUMERIC", "X.csv contains non-numeric spectral values.", value=text) from exc
    if not math.isfinite(number):
        raise PreprocessInputError("X_NON_FINITE", "X.csv contains non-finite spectral values.", value=text)
    return number

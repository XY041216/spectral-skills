"""Read spectral-reader/QC standard packages for splitting."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file


class SplitInputError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class SpectralSplitPackage:
    root: Path
    contract_path: Path
    contract: dict[str, Any]
    sample_ids: list[str]
    x: list[list[float]]
    n_features: int
    labels: list[str] | None
    y_header: list[str]
    metadata_header: list[str]
    metadata_rows: list[dict[str, str]]
    task_hint: str

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)


def load_split_package(package_dir: str | Path) -> SpectralSplitPackage:
    root = Path(package_dir)
    if not root.exists():
        raise SplitInputError("PACKAGE_DIR_NOT_FOUND", "Standard package directory does not exist.", path=str(root))
    contract_path = root / "data_contract.json"
    if not contract_path.exists():
        raise SplitInputError("DATA_CONTRACT_MISSING", "data_contract.json is required for spectral splitting.", path=str(contract_path))

    contract = load_json_file(contract_path)
    files = dict(contract.get("files") or {})
    x_path = _resolve_required(root, files.get("X") or contract.get("X") or "X.csv", "X")
    sample_path = _resolve_required(root, files.get("sample_ids") or contract.get("sample_ids") or "sample_ids.csv", "sample_ids")
    band_path = _resolve_required(root, files.get("band_axis") or contract.get("band_axis_ref") or "band_axis.csv", "band_axis")
    y_path = _resolve_optional(root, files.get("y") or contract.get("y"))

    x_rows = _read_csv(x_path)
    if len(x_rows) < 2 or not x_rows[0]:
        raise SplitInputError("X_EMPTY", "X.csv must contain at least one sample and one feature.", path=str(x_path))
    n_features = len(x_rows[0])
    n_x_samples = len(x_rows) - 1
    try:
        x = [[float(value) for value in row] for row in x_rows[1:]]
    except ValueError as exc:
        raise SplitInputError("X_NON_NUMERIC", "X.csv contains non-numeric spectral values; split methods that use X distances are unsafe.", path=str(x_path)) from exc
    non_finite = [
        {"sample_index": row_idx, "feature_index": col_idx, "value": value}
        for row_idx, row in enumerate(x)
        for col_idx, value in enumerate(row)
        if not math.isfinite(value)
    ]
    if non_finite:
        raise SplitInputError(
            "X_NON_FINITE",
            "X.csv contains NaN or infinite spectral values; splitting must wait until the standard package is valid.",
            examples=non_finite[:20],
        )

    sample_ids = _read_single_column(sample_path)
    _assert_unique_sample_ids(sample_ids)
    if len(sample_ids) != n_x_samples:
        raise SplitInputError("SAMPLE_IDS_ROW_MISMATCH", "sample_ids row count must equal X sample count.", expected=n_x_samples, observed=len(sample_ids))

    band_rows = _read_csv(band_path)
    if len(band_rows) - 1 != n_features:
        raise SplitInputError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length must equal X feature count.", expected=n_features, observed=len(band_rows) - 1)

    y_header: list[str] = []
    labels: list[str] | None = None
    if y_path is not None:
        y_header, y_rows = _read_table(y_path)
        if len(y_rows) != n_x_samples:
            raise SplitInputError("Y_ROW_MISMATCH", "y row count must equal X sample count.", expected=n_x_samples, observed=len(y_rows))
        labels = [str(row[0]).strip() if row else "" for row in y_rows]
        if any(label == "" for label in labels):
            raise SplitInputError("Y_EMPTY_LABEL", "y.csv contains empty labels or targets; stratified splitting is unsafe.", path=str(y_path))

    metadata_header: list[str] = []
    metadata_rows: list[dict[str, str]] = []
    metadata_path = _resolve_optional(root, files.get("metadata") or contract.get("metadata"))
    if metadata_path is not None:
        metadata_header, raw_metadata_rows = _read_table(metadata_path)
        if len(raw_metadata_rows) != n_x_samples:
            raise SplitInputError("METADATA_ROW_MISMATCH", "metadata row count must equal X sample count.", expected=n_x_samples, observed=len(raw_metadata_rows))
        metadata_rows = [
            {metadata_header[idx]: str(value).strip() for idx, value in enumerate(row[: len(metadata_header)])}
            for row in raw_metadata_rows
        ]

    task_hint = str(contract.get("task_hint") or contract.get("task_type") or "unknown").lower()
    return SpectralSplitPackage(
        root=root,
        contract_path=contract_path,
        contract=contract,
        sample_ids=sample_ids,
        x=x,
        n_features=n_features,
        labels=labels,
        y_header=y_header,
        metadata_header=metadata_header,
        metadata_rows=metadata_rows,
        task_hint=task_hint,
    )


def _resolve_required(root: Path, ref: Any, role: str) -> Path:
    path = _resolve_optional(root, ref)
    if path is None or not path.exists():
        raise SplitInputError("STANDARD_FILE_MISSING", f"{role} file is missing from the standard package.", role=role, ref=ref)
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


def _read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_single_column(path: Path) -> list[str]:
    _, rows = _read_table(path)
    return [str(row[0]).strip() if row else "" for row in rows]


def _assert_unique_sample_ids(sample_ids: list[str]) -> None:
    if any(sample_id == "" for sample_id in sample_ids):
        raise SplitInputError("SAMPLE_ID_EMPTY", "sample_ids.csv contains empty sample IDs.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen and sample_id not in duplicates:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise SplitInputError("SAMPLE_ID_DUPLICATE", "sample_ids must be unique before splitting.", sample_ids=duplicates)

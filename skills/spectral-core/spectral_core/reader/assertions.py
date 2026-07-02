"""Runtime assertions for reliable reader outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class ReaderAssertionError(Exception):
    """Raised when a reader hard assertion blocks formal output."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_issue(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def assert_required_refs_exist(base_dir: str | Path, refs: dict[str, Any], required_keys: list[str]) -> None:
    root = Path(base_dir)
    for key in required_keys:
        ref = refs.get(key)
        if not ref:
            raise ReaderAssertionError("DATA_REF_MISSING", f"{key} is missing.", ref_key=key)
        path = _resolve(root, str(ref))
        if not path.exists():
            raise ReaderAssertionError("DATA_REF_NOT_FOUND", f"{key} does not exist.", ref_key=key, path=str(path))


def assert_X_exists(base_dir: str | Path, X_ref: str | None) -> Path:
    if not X_ref:
        raise ReaderAssertionError("X_REF_MISSING", "X_ref is missing.")
    path = _resolve(Path(base_dir), X_ref)
    if not path.exists():
        raise ReaderAssertionError("X_NOT_FOUND", "X.csv does not exist.", path=str(path))
    return path


def assert_X_numeric(X_path: str | Path) -> None:
    _, rows = _csv_rows(Path(X_path))
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if str(value).strip().lower() in {"", "na", "n/a", "nan", "null", "none", "missing"}:
                continue
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ReaderAssertionError("X_NON_NUMERIC", "X contains a non-numeric value.", row_index=row_index, column_index=column_index, value=value) from exc


def assert_non_empty_matrix(X_path: str | Path) -> tuple[int, int]:
    header, rows = _csv_rows(Path(X_path))
    if not rows or not header:
        raise ReaderAssertionError("EMPTY_X", "X matrix has zero rows or zero columns.")
    n_features = len(header)
    for row in rows:
        if len(row) != n_features:
            raise ReaderAssertionError("X_RAGGED", "X rows do not have consistent feature counts.", expected=n_features, observed=len(row))
    return len(rows), n_features


def assert_y_aligned(base_dir: str | Path, y_ref: str | None, n_samples: int) -> None:
    _assert_rows_aligned(base_dir, y_ref, n_samples, "y_ref")


def assert_sample_ids_aligned(base_dir: str | Path, sample_ids_ref: str | None, n_samples: int) -> None:
    _assert_rows_aligned(base_dir, sample_ids_ref, n_samples, "sample_ids_ref")


def assert_metadata_aligned(base_dir: str | Path, metadata_ref: str | None, n_samples: int) -> None:
    _assert_rows_aligned(base_dir, metadata_ref, n_samples, "metadata_ref")


def assert_band_axis_aligned(base_dir: str | Path, band_axis_ref: str | None, n_features: int) -> None:
    if not band_axis_ref:
        raise ReaderAssertionError("BAND_AXIS_REF_MISSING", "band_axis_ref is missing.")
    path = _resolve(Path(base_dir), band_axis_ref)
    if not path.exists():
        raise ReaderAssertionError("BAND_AXIS_NOT_FOUND", "band_axis.csv does not exist.", path=str(path))
    _, rows = _csv_rows(path)
    if len(rows) != n_features:
        raise ReaderAssertionError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length does not match X feature count.", expected=n_features, observed=len(rows))


def assert_confirmed_or_blocked_status(status: str | None) -> None:
    if status not in {"confirmed", "blocked", "provisional"}:
        raise ReaderAssertionError("STATUS_INVALID", "Reader status must be confirmed, provisional, or blocked.", status=status)


def assert_unique_sample_ids(sample_ids: list[Any] | None) -> None:
    if sample_ids is None:
        return
    duplicates = sorted({value for value in sample_ids if sample_ids.count(value) > 1})
    if duplicates:
        raise ReaderAssertionError("DUPLICATE_SAMPLE_IDS", "sample_ids must be unique.", duplicates=duplicates)


def assert_label_alignment(sample_ids: list[Any], label_keys: set[str]) -> None:
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in label_keys]
    if missing:
        raise ReaderAssertionError("LABEL_ALIGNMENT_FAILED", "External labels do not align to sample_ids.", missing_sample_ids=missing)


def assert_no_unmatched_required_labels(missing_sample_ids: list[str] | None) -> None:
    if missing_sample_ids:
        raise ReaderAssertionError("MISSING_REQUIRED_LABELS", "Required labels are missing.", missing_sample_ids=missing_sample_ids, missing_count=len(missing_sample_ids))


def assert_band_axis_column_valid(header: list[str], column: str | int | None) -> None:
    if column is None:
        raise ReaderAssertionError("BAND_AXIS_COLUMN_MISSING", "samples_as_columns requires a band_axis_column.")
    if isinstance(column, int):
        if column < 0 or column >= len(header):
            raise ReaderAssertionError("BAND_AXIS_COLUMN_INVALID", "band_axis_column index is outside the header.", column=column, available=header)
        return
    if str(column) not in header:
        raise ReaderAssertionError("BAND_AXIS_COLUMN_INVALID", "band_axis_column does not exist.", column=column, available=header)


def assert_samples_as_columns_matrix_valid(X: list[list[Any]], sample_ids: list[Any], band_axis: list[Any]) -> None:
    if not X or not X[0]:
        raise ReaderAssertionError("EMPTY_X", "samples-as-columns extraction produced an empty X matrix.")
    if len(X) != len(sample_ids):
        raise ReaderAssertionError("SAMPLE_ID_LENGTH_MISMATCH", "sample_ids length does not match X rows.", expected=len(X), observed=len(sample_ids))
    if len(X[0]) != len(band_axis):
        raise ReaderAssertionError("BAND_AXIS_LENGTH_MISMATCH", "band_axis length does not match X columns.", expected=len(X[0]), observed=len(band_axis))


def assert_external_label_file_valid(path: str | Path | None) -> None:
    if not path:
        raise ReaderAssertionError("LABEL_FILE_PATH_MISSING", "label_file.path is missing.")
    if not Path(path).exists():
        raise ReaderAssertionError("LABEL_FILE_NOT_FOUND", "label_file.path does not exist.", path=str(path))


def assert_join_key_exists(columns: list[str], join_key: str | None) -> None:
    if not join_key:
        raise ReaderAssertionError("JOIN_KEY_MISSING", "alignment_plan.join_key is missing.")
    if join_key not in columns:
        raise ReaderAssertionError("JOIN_KEY_NOT_FOUND", "join_key does not exist in external label file.", join_key=join_key, available=columns)


def assert_no_duplicate_label_keys(keys: list[str]) -> None:
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ReaderAssertionError("DUPLICATE_LABEL_KEYS", "External label file has duplicate join keys.", duplicates=duplicates)


def _assert_rows_aligned(base_dir: str | Path, ref: str | None, n_samples: int, ref_key: str) -> None:
    if not ref:
        return
    path = _resolve(Path(base_dir), ref)
    if not path.exists():
        raise ReaderAssertionError("DATA_REF_NOT_FOUND", f"{ref_key} does not exist.", ref_key=ref_key, path=str(path))
    _, rows = _csv_rows(path)
    if len(rows) != n_samples:
        raise ReaderAssertionError("ROW_COUNT_MISMATCH", f"{ref_key} row count does not match X.", ref_key=ref_key, expected=n_samples, observed=len(rows))


def _csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _resolve(root: Path, ref: str) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else root / path

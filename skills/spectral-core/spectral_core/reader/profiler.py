"""Minimal spectral data profiler for apply_read_plan outputs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import load_json_file, write_json_file
from .response import error_response, ok_response


def profile_spectral_data(
    apply_dir: str | Path | None = None,
    *,
    output: str | Path | None = None,
    strict: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    if apply_dir is None:
        return error_response("profile_spectral_data", "--apply-dir is required.", backend=backend, code="APPLY_DIR_REQUIRED")
    try:
        loaded = load_apply_outputs(apply_dir)
        profile = build_profile_summary(loaded, strict=strict)
        if output:
            out = Path(output)
            write_json_file(out, profile, ensure_ascii=False)
            profile["profile_ref"] = str(out)
        return ok_response("profile_spectral_data", profile, backend=backend, warnings=profile.get("profile_warnings", []))
    except ProfileBlockedError as exc:
        return error_response("profile_spectral_data", exc.message, backend=backend, code=exc.code, result={"profile_status": "blocked"}, details=exc.details)
    except Exception as exc:
        return error_response("profile_spectral_data", f"Profiling failed: {exc}", backend=backend, code="PROFILE_FAILED")


def load_apply_outputs(apply_dir: str | Path) -> dict[str, Any]:
    root = Path(apply_dir)
    if not root.exists():
        raise ProfileBlockedError("APPLY_DIR_NOT_FOUND", "apply_dir does not exist.", apply_dir=str(root))
    apply_result_path = root / "apply_read_plan_result.json"
    if not apply_result_path.exists():
        raise ProfileBlockedError("APPLY_RESULT_NOT_FOUND", "apply_read_plan_result.json is missing.", apply_dir=str(root))
    apply_result = load_json_file(apply_result_path)
    refs = apply_result.get("data_refs") or {}
    X_path = _resolve_ref(root, refs.get("X_ref"))
    band_path = _resolve_ref(root, refs.get("band_axis_ref"))
    if not X_path or not X_path.exists():
        raise ProfileBlockedError("X_REF_MISSING", "X_ref is missing or does not exist.", ref=refs.get("X_ref"))
    if not band_path or not band_path.exists():
        raise ProfileBlockedError("BAND_AXIS_REF_MISSING", "band_axis_ref is missing or does not exist.", ref=refs.get("band_axis_ref"))

    X_header, X_rows_raw = _read_csv_rows(X_path)
    X = [[_to_float(value) for value in row] for row in X_rows_raw]
    band_header, band_rows = _read_csv_rows(band_path)
    y = _read_first_column(_resolve_ref(root, refs.get("y_ref")))
    sample_ids = _read_first_column(_resolve_ref(root, refs.get("sample_ids_ref")))
    metadata = _read_dicts(_resolve_ref(root, refs.get("metadata_ref")))
    band_values = [_parse_band_value(row, band_header) for row in band_rows]
    band_unit = _parse_band_unit(band_rows, band_header)
    return {
        "apply_dir": str(root),
        "apply_result": apply_result,
        "X_header": X_header,
        "X": X,
        "y": y,
        "sample_ids": sample_ids,
        "band_axis": band_values,
        "band_unit": band_unit,
        "metadata": metadata,
    }


def build_profile_summary(loaded: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    X = loaded["X"]
    if not X or not X[0]:
        raise ProfileBlockedError("EMPTY_X", "X is empty.")
    n_samples = len(X)
    n_features = len(X[0])
    band_axis = loaded["band_axis"]
    y = loaded.get("y")
    sample_ids = loaded.get("sample_ids")
    metadata = loaded.get("metadata")
    warnings: list[dict[str, Any]] = []
    missing_count = sum(1 for row in X for value in row if value is None)
    columns = list(zip(*X))
    constant_count = sum(1 for col in columns if len(set(col)) <= 1)
    low_variance_count = sum(1 for col in columns if _variance(col) < 1e-12)
    duplicate_sample_id_count = 0
    if sample_ids:
        duplicate_sample_id_count = sum(count - 1 for count in Counter(sample_ids).values() if count > 1)
    class_distribution = None
    target_summary = None
    if y:
        y_numbers = [_try_float(value) for value in y]
        if all(value is not None for value in y_numbers):
            nums = [float(value) for value in y_numbers if value is not None]
            target_summary = {"count": len(nums), "min": min(nums), "max": max(nums), "mean": mean(nums)}
        else:
            class_distribution = dict(Counter(str(value) for value in y))
    p_over_n = n_features / n_samples if n_samples else None
    small_sample_warning = bool(n_samples < 20 or (p_over_n is not None and p_over_n > 10))
    if small_sample_warning:
        warnings.append({"code": "SMALL_SAMPLE_RISK", "message": "Sample count is small or p/n is high.", "severity": "warning", "details": {"n_samples": n_samples, "n_features": n_features, "p_over_n_ratio": p_over_n}})
    if duplicate_sample_id_count:
        warnings.append({"code": "DUPLICATE_SAMPLE_IDS", "message": "Duplicate sample IDs were found.", "severity": "warning", "details": {"duplicate_count": duplicate_sample_id_count}})

    return {
        "profile_status": "ok",
        "apply_dir": loaded["apply_dir"],
        "sample_orientation": (loaded["apply_result"].get("execution_summary") or {}).get("sample_orientation") or loaded["apply_result"].get("sample_orientation"),
        "alignment_summary": loaded["apply_result"].get("alignment_summary") or {},
        "external_label_used": bool(loaded["apply_result"].get("external_label_used")),
        "label_source": _label_source(loaded["apply_result"], y),
        "n_samples": n_samples,
        "n_features": n_features,
        "has_y": y is not None,
        "has_sample_ids": sample_ids is not None,
        "has_metadata": metadata is not None,
        "band_axis_unit": loaded.get("band_unit"),
        "band_axis_start": band_axis[0] if band_axis else None,
        "band_axis_end": band_axis[-1] if band_axis else None,
        "band_axis_monotonicity": _monotonicity(band_axis),
        "missing_value_summary": {"X_missing_count": missing_count, "X_missing_fraction": missing_count / (n_samples * n_features)},
        "constant_band_count": constant_count,
        "low_variance_band_count": low_variance_count,
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "missing_label_count": (loaded["apply_result"].get("alignment_summary") or {}).get("missing_label_count", 0),
        "duplicate_label_key_count": (loaded["apply_result"].get("alignment_summary") or {}).get("duplicate_label_key_count", 0),
        "class_distribution": class_distribution,
        "target_summary": target_summary,
        "p_over_n_ratio": p_over_n,
        "small_sample_warning": small_sample_warning,
        "profile_warnings": warnings,
        "profile_boundaries": {"no_qc_removal": True, "no_downstream_modeling": True, "strict": strict},
    }


def _resolve_ref(root: Path, ref: str | None) -> Path | None:
    if not ref:
        return None
    path = Path(ref)
    return path if path.is_absolute() else root / path


def _label_source(apply_result: dict[str, Any], y: list[str] | None) -> str:
    declared = apply_result.get("label_source")
    if declared:
        return str(declared)
    return "embedded" if y is not None else "none"


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ProfileBlockedError("EMPTY_CSV", "CSV file is empty.", path=str(path))
    return rows[0], rows[1:]


def _read_first_column(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    _, rows = _read_csv_rows(path)
    return [row[0] if row else "" for row in rows]


def _read_dicts(path: Path | None) -> list[dict[str, str]] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _variance(values: tuple[float | None, ...]) -> float:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return 0.0
    avg = mean(nums)
    return sum((value - avg) ** 2 for value in nums) / len(nums)


def _parse_band_value(row: list[str], header: list[str]) -> Any:
    idx = header.index("value") if "value" in header else 1
    raw = row[idx] if idx < len(row) else ""
    num = _try_float(raw)
    return int(num) if num is not None and num.is_integer() else num if num is not None else raw


def _parse_band_unit(rows: list[list[str]], header: list[str]) -> str | None:
    if "unit" not in header or not rows:
        return None
    idx = header.index("unit")
    return rows[0][idx] if idx < len(rows[0]) else None


def _monotonicity(values: list[Any]) -> str:
    nums = [_try_float(value) for value in values]
    if not nums or any(value is None for value in nums):
        return "unknown"
    clean = [float(value) for value in nums if value is not None]
    if all(a < b for a, b in zip(clean, clean[1:])):
        return "increasing"
    if all(a > b for a, b in zip(clean, clean[1:])):
        return "decreasing"
    return "non_monotonic"


class ProfileBlockedError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

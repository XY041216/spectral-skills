"""First-stage leakage-safe spectral preprocessing methods."""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - exercised only when numpy is unavailable.
    np = None  # type: ignore[assignment]

try:
    from scipy import sparse
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.signal import savgol_filter
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover - exercised only when scipy is unavailable.
    sparse = None  # type: ignore[assignment]
    gaussian_filter1d = None  # type: ignore[assignment]
    median_filter = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]
    spsolve = None  # type: ignore[assignment]


class PreprocessMethodError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


SUPPORTED_METHODS = {
    "none",
    "snv",
    "msc",
    "sg_smoothing",
    "first_derivative",
    "second_derivative",
    "moving_average",
    "gaussian_smoothing",
    "median_filter",
    "detrend",
    "snv_detrend",
    "linear_baseline",
    "polynomial_baseline",
    "rubberband_baseline",
    "als_baseline",
    "mean_centering",
    "standardization",
    "minmax_scaling",
    "robust_scaling",
    "pareto_scaling",
    "l2_normalization",
    "area_normalization",
    "max_abs_normalization",
    "reflectance_to_absorbance",
    "transmittance_to_absorbance",
    "log_transform",
    "band_range_select",
    "remove_band_ranges",
}

TRAIN_FIT_METHODS = {"msc", "mean_centering", "standardization", "minmax_scaling", "robust_scaling", "pareto_scaling"}
SG_METHODS = {"sg_smoothing", "first_derivative", "second_derivative"}
BASELINE_METHODS = {"linear_baseline", "polynomial_baseline", "rubberband_baseline", "als_baseline"}
BAND_CHANGING_METHODS = {"band_range_select", "remove_band_ranges"}
ABSORBANCE_METHODS = {"reflectance_to_absorbance", "transmittance_to_absorbance"}
ADVANCED_METHODS = {
    "osc",
    "emsc",
    "wavelet",
    "wavelet_denoising",
    "pca",
    "spa",
    "cars",
    "vip",
    "uve",
    "auto_search",
    "optimizer",
    "airpls_baseline",
    "arpls_baseline",
    "continuum_removal",
    "resample_band_axis",
}


def parse_methods(methods: list[str] | str | None) -> list[str]:
    if methods is None:
        raise PreprocessMethodError("PREPROCESS_METHOD_REQUIRED", "Please choose preprocessing methods: none, snv, msc, sg_smoothing, first_derivative, second_derivative, mean_centering, or standardization.")
    if isinstance(methods, str):
        raw = [item.strip() for item in methods.split(",") if item.strip()]
    else:
        raw = [str(item).strip() for item in methods if str(item).strip()]
    if not raw:
        raise PreprocessMethodError("PREPROCESS_METHOD_REQUIRED", "Please choose preprocessing methods: none, snv, msc, sg_smoothing, first_derivative, second_derivative, mean_centering, or standardization.")
    normalized = [_normalize_method(item) for item in raw]
    for method in normalized:
        if method in ADVANCED_METHODS:
            raise PreprocessMethodError("PREPROCESS_METHOD_NOT_IMPLEMENTED", "This first preprocess version does not implement advanced preprocessing, feature selection, or optimizer methods.", method=method)
        if method not in SUPPORTED_METHODS:
            raise PreprocessMethodError("PREPROCESS_METHOD_UNSUPPORTED", "Unsupported preprocessing method for this MVP.", method=method)
    if "none" in normalized and len(normalized) > 1:
        raise PreprocessMethodError("PREPROCESS_NONE_EXCLUSIVE", "`none` cannot be combined with other preprocessing methods.")
    return normalized


def requires_train_fit(methods: list[str]) -> bool:
    return any(method in TRAIN_FIT_METHODS for method in methods)


def requires_baseline_confirmation(methods: list[str]) -> bool:
    return any(method in BASELINE_METHODS for method in methods)


def requires_band_change_confirmation(methods: list[str]) -> bool:
    return any(method in BAND_CHANGING_METHODS for method in methods)


def requires_absorbance_confirmation(methods: list[str]) -> bool:
    return any(method in ABSORBANCE_METHODS for method in methods)


def apply_preprocess_methods(
    X: list[list[float]],
    *,
    methods: list[str],
    train_indices: list[int],
    window_length: int | None,
    polyorder: int | None,
    sigma: float | None = None,
    poly_degree: int | None = None,
    als_lambda: float | None = None,
    als_p: float | None = None,
    als_iter: int | None = None,
    band_range: str | None = None,
    remove_band_ranges: str | None = None,
    feature_names: list[str] | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    current = [list(row) for row in X]
    feature_indices = list(range(len(X[0]) if X else 0))
    feature_values = _feature_values(feature_names, len(feature_indices))
    state: dict[str, Any] = {
        "methods": [],
        "fit_scope": "train_only" if requires_train_fit(methods) else "not_applicable_stateless_or_per_sample",
        "transform_scope": "train_val_test",
        "feature_indices": feature_indices,
    }
    for method in methods:
        if method == "none":
            state["methods"].append({"method": "none", "parameters": {}, "fitted": {}})
        elif method == "band_range_select":
            current, feature_indices, method_state = _apply_band_range_select(current, feature_indices, feature_values, band_range)
            state["feature_indices"] = feature_indices
            state["methods"].append(method_state)
        elif method == "remove_band_ranges":
            current, feature_indices, method_state = _apply_remove_band_ranges(current, feature_indices, feature_values, remove_band_ranges)
            state["feature_indices"] = feature_indices
            state["methods"].append(method_state)
        elif method == "reflectance_to_absorbance":
            current, method_state = _apply_absorbance(current, method=method)
            state["methods"].append(method_state)
        elif method == "transmittance_to_absorbance":
            current, method_state = _apply_absorbance(current, method=method)
            state["methods"].append(method_state)
        elif method == "log_transform":
            current, method_state = _apply_log_transform(current)
            state["methods"].append(method_state)
        elif method == "detrend":
            current, method_state = _apply_detrend(current)
            state["methods"].append(method_state)
        elif method == "snv_detrend":
            current, snv_state = _apply_snv(current)
            current, detrend_state = _apply_detrend(current)
            state["methods"].append({"method": "snv_detrend", "parameters": {}, "fitted": {"scope": "per_sample"}, "steps": [snv_state, detrend_state]})
        elif method == "snv":
            current, method_state = _apply_snv(current)
            state["methods"].append(method_state)
        elif method == "mean_centering":
            current, method_state = _apply_mean_centering(current, train_indices)
            state["methods"].append(method_state)
        elif method == "standardization":
            current, method_state = _apply_standardization(current, train_indices)
            state["methods"].append(method_state)
        elif method == "minmax_scaling":
            current, method_state = _apply_minmax_scaling(current, train_indices)
            state["methods"].append(method_state)
        elif method == "robust_scaling":
            current, method_state = _apply_robust_scaling(current, train_indices)
            state["methods"].append(method_state)
        elif method == "pareto_scaling":
            current, method_state = _apply_pareto_scaling(current, train_indices)
            state["methods"].append(method_state)
        elif method == "msc":
            current, method_state = _apply_msc(current, train_indices)
            state["methods"].append(method_state)
        elif method in SG_METHODS:
            current, method_state = _apply_savgol(current, method=method, window_length=window_length, polyorder=polyorder)
            state["methods"].append(method_state)
        elif method == "moving_average":
            current, method_state = _apply_moving_average(current, window_length=window_length)
            state["methods"].append(method_state)
        elif method == "gaussian_smoothing":
            current, method_state = _apply_gaussian_smoothing(current, sigma=sigma)
            state["methods"].append(method_state)
        elif method == "median_filter":
            current, method_state = _apply_median_filter(current, window_length=window_length)
            state["methods"].append(method_state)
        elif method == "linear_baseline":
            current, method_state = _apply_linear_baseline(current)
            state["methods"].append(method_state)
        elif method == "polynomial_baseline":
            current, method_state = _apply_polynomial_baseline(current, degree=poly_degree)
            state["methods"].append(method_state)
        elif method == "rubberband_baseline":
            current, method_state = _apply_rubberband_baseline(current)
            state["methods"].append(method_state)
        elif method == "als_baseline":
            current, method_state = _apply_als_baseline(current, lam=als_lambda, p=als_p, n_iter=als_iter)
            state["methods"].append(method_state)
        elif method == "l2_normalization":
            current, method_state = _apply_l2_normalization(current)
            state["methods"].append(method_state)
        elif method == "area_normalization":
            current, method_state = _apply_area_normalization(current)
            state["methods"].append(method_state)
        elif method == "max_abs_normalization":
            current, method_state = _apply_max_abs_normalization(current)
            state["methods"].append(method_state)
        else:  # pragma: no cover - parse_methods prevents this.
            raise PreprocessMethodError("PREPROCESS_METHOD_UNSUPPORTED", "Unsupported preprocessing method.", method=method)
    _assert_shape_unchanged(X, current)
    return current, state


def _normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "savitzky_golay": "sg_smoothing",
        "sg": "sg_smoothing",
        "savgol": "sg_smoothing",
        "1st_derivative": "first_derivative",
        "derivative_1": "first_derivative",
        "2nd_derivative": "second_derivative",
        "derivative_2": "second_derivative",
        "mean_center": "mean_centering",
        "center": "mean_centering",
        "zscore": "standardization",
        "standardize": "standardization",
        "minmax": "minmax_scaling",
        "min_max": "minmax_scaling",
        "robust": "robust_scaling",
        "pareto": "pareto_scaling",
        "l2": "l2_normalization",
        "area": "area_normalization",
        "max_abs": "max_abs_normalization",
        "reflectance_absorbance": "reflectance_to_absorbance",
        "transmittance_absorbance": "transmittance_to_absorbance",
        "baseline_linear": "linear_baseline",
        "baseline_polynomial": "polynomial_baseline",
        "poly_baseline": "polynomial_baseline",
        "rubberband": "rubberband_baseline",
        "als": "als_baseline",
        "band_select": "band_range_select",
        "select_band_range": "band_range_select",
        "remove_band_range": "remove_band_ranges",
    }
    return aliases.get(normalized, normalized)


def _feature_values(feature_names: list[str] | None, n_features: int) -> list[float]:
    output: list[float] = []
    for idx in range(n_features):
        raw = feature_names[idx] if feature_names and idx < len(feature_names) else idx
        try:
            output.append(float(str(raw).strip()))
        except ValueError:
            output.append(float(idx))
    return output


def _apply_band_range_select(
    X: list[list[float]],
    feature_indices: list[int],
    feature_values: list[float],
    band_range: str | None,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    ranges = _parse_ranges(band_range, code="BAND_RANGE_REQUIRED")
    keep_positions = [
        pos for pos, original_idx in enumerate(feature_indices)
        if _value_in_ranges(feature_values[original_idx], ranges)
    ]
    return _slice_features(X, feature_indices, keep_positions, "band_range_select", {"band_range": band_range, "ranges": ranges})


def _apply_remove_band_ranges(
    X: list[list[float]],
    feature_indices: list[int],
    feature_values: list[float],
    remove_band_ranges: str | None,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    ranges = _parse_ranges(remove_band_ranges, code="REMOVE_BAND_RANGES_REQUIRED")
    keep_positions = [
        pos for pos, original_idx in enumerate(feature_indices)
        if not _value_in_ranges(feature_values[original_idx], ranges)
    ]
    return _slice_features(X, feature_indices, keep_positions, "remove_band_ranges", {"remove_band_ranges": remove_band_ranges, "ranges": ranges})


def _slice_features(
    X: list[list[float]],
    feature_indices: list[int],
    keep_positions: list[int],
    method: str,
    parameters: dict[str, Any],
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    if not keep_positions:
        raise PreprocessMethodError("BAND_SELECTION_EMPTY", "Band range preprocessing removed every feature.", method=method)
    output = [[row[pos] for pos in keep_positions] for row in X]
    new_indices = [feature_indices[pos] for pos in keep_positions]
    return output, new_indices, {"method": method, "parameters": parameters, "fitted": {"scope": "deterministic_band_axis", "selected_feature_indices": new_indices, "output_n_features": len(new_indices)}}


def _parse_ranges(raw: str | None, *, code: str) -> list[tuple[float, float]]:
    if raw is None or not str(raw).strip():
        raise PreprocessMethodError(code, "Band range preprocessing requires a range string such as 900:1300 or 4000:10000,5200:5400.")
    ranges: list[tuple[float, float]] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        delimiter = ":" if ":" in text else "-" if "-" in text[1:] else None
        if delimiter is None:
            raise PreprocessMethodError("BAND_RANGE_INVALID", "Band ranges must use start:end syntax.", value=text)
        start_text, end_text = text.split(delimiter, 1)
        try:
            start = float(start_text.strip())
            end = float(end_text.strip())
        except ValueError as exc:
            raise PreprocessMethodError("BAND_RANGE_INVALID", "Band range bounds must be numeric.", value=text) from exc
        low, high = (start, end) if start <= end else (end, start)
        ranges.append((low, high))
    if not ranges:
        raise PreprocessMethodError(code, "Band range preprocessing requires at least one range.")
    return ranges


def _value_in_ranges(value: float, ranges: list[tuple[float, float]]) -> bool:
    return any(low <= value <= high for low, high in ranges)


def _apply_snv(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row in X:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        std = math.sqrt(variance)
        if std == 0:
            raise PreprocessMethodError("SNV_ZERO_VARIANCE_SPECTRUM", "SNV cannot transform a constant spectrum.")
        output.append([(value - mean) / std for value in row])
    return output, {"method": "snv", "parameters": {}, "fitted": {"scope": "per_sample", "requires_train_fit": False}}


def _apply_mean_centering(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    means = _column_means([X[idx] for idx in train_indices])
    output = [[value - means[col_idx] for col_idx, value in enumerate(row)] for row in X]
    return output, {"method": "mean_centering", "parameters": {}, "fitted": {"mean_vector": means, "fit_sample_count": len(train_indices)}}


def _apply_standardization(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    train = [X[idx] for idx in train_indices]
    means = _column_means(train)
    stds = _column_stds(train, means)
    if any(std == 0 for std in stds):
        zero_cols = [idx for idx, std in enumerate(stds) if std == 0]
        raise PreprocessMethodError("STANDARDIZATION_ZERO_STD", "Standardization cannot divide by zero train-set standard deviation.", band_indices=zero_cols)
    output = [[(value - means[col_idx]) / stds[col_idx] for col_idx, value in enumerate(row)] for row in X]
    return output, {"method": "standardization", "parameters": {}, "fitted": {"mean_vector": means, "std_vector": stds, "fit_sample_count": len(train_indices)}}


def _apply_minmax_scaling(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    train = [X[idx] for idx in train_indices]
    mins = [min(row[col_idx] for row in train) for col_idx in range(len(train[0]))]
    maxs = [max(row[col_idx] for row in train) for col_idx in range(len(train[0]))]
    ranges = [maxs[idx] - mins[idx] for idx in range(len(mins))]
    if any(value == 0 for value in ranges):
        zero_cols = [idx for idx, value in enumerate(ranges) if value == 0]
        raise PreprocessMethodError("MINMAX_ZERO_RANGE", "Min-max scaling cannot divide by zero train-set range.", band_indices=zero_cols)
    output = [[(value - mins[col_idx]) / ranges[col_idx] for col_idx, value in enumerate(row)] for row in X]
    return output, {"method": "minmax_scaling", "parameters": {"feature_range": [0, 1]}, "fitted": {"min_vector": mins, "max_vector": maxs, "fit_sample_count": len(train_indices)}}


def _apply_robust_scaling(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    train = [X[idx] for idx in train_indices]
    medians = [_quantile([row[col_idx] for row in train], 0.5) for col_idx in range(len(train[0]))]
    q1 = [_quantile([row[col_idx] for row in train], 0.25) for col_idx in range(len(train[0]))]
    q3 = [_quantile([row[col_idx] for row in train], 0.75) for col_idx in range(len(train[0]))]
    iqrs = [q3[idx] - q1[idx] for idx in range(len(q1))]
    if any(value == 0 for value in iqrs):
        zero_cols = [idx for idx, value in enumerate(iqrs) if value == 0]
        raise PreprocessMethodError("ROBUST_SCALING_ZERO_IQR", "Robust scaling cannot divide by zero train-set IQR.", band_indices=zero_cols)
    output = [[(value - medians[col_idx]) / iqrs[col_idx] for col_idx, value in enumerate(row)] for row in X]
    return output, {"method": "robust_scaling", "parameters": {"quantile_range": [25, 75]}, "fitted": {"median_vector": medians, "iqr_vector": iqrs, "fit_sample_count": len(train_indices)}}


def _apply_pareto_scaling(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    train = [X[idx] for idx in train_indices]
    means = _column_means(train)
    stds = _column_stds(train, means)
    denominators = [math.sqrt(std) for std in stds]
    if any(value == 0 for value in denominators):
        zero_cols = [idx for idx, value in enumerate(denominators) if value == 0]
        raise PreprocessMethodError("PARETO_ZERO_STD", "Pareto scaling cannot divide by zero train-set standard deviation.", band_indices=zero_cols)
    output = [[(value - means[col_idx]) / denominators[col_idx] for col_idx, value in enumerate(row)] for row in X]
    return output, {"method": "pareto_scaling", "parameters": {}, "fitted": {"mean_vector": means, "sqrt_std_vector": denominators, "fit_sample_count": len(train_indices)}}


def _apply_msc(X: list[list[float]], train_indices: list[int]) -> tuple[list[list[float]], dict[str, Any]]:
    reference = _column_means([X[idx] for idx in train_indices])
    ref_mean = sum(reference) / len(reference)
    ref_centered = [value - ref_mean for value in reference]
    ref_ss = sum(value * value for value in ref_centered)
    if ref_ss == 0:
        raise PreprocessMethodError("MSC_REFERENCE_ZERO_VARIANCE", "MSC reference spectrum has zero variance.")
    output = []
    coefficients = []
    for row in X:
        row_mean = sum(row) / len(row)
        row_centered = [value - row_mean for value in row]
        slope = sum(row_centered[idx] * ref_centered[idx] for idx in range(len(row))) / ref_ss
        if slope == 0:
            raise PreprocessMethodError("MSC_ZERO_SLOPE", "MSC produced zero slope for a spectrum.")
        intercept = row_mean - slope * ref_mean
        output.append([(value - intercept) / slope for value in row])
        coefficients.append({"intercept": intercept, "slope": slope})
    return output, {"method": "msc", "parameters": {}, "fitted": {"reference_spectrum": reference, "fit_sample_count": len(train_indices), "coefficients": coefficients}}


def _apply_l2_normalization(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row_idx, row in enumerate(X):
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            raise PreprocessMethodError("L2_ZERO_NORM", "L2 normalization cannot transform a zero spectrum.", sample_index=row_idx)
        output.append([value / norm for value in row])
    return output, {"method": "l2_normalization", "parameters": {}, "fitted": {"scope": "per_sample"}}


def _apply_area_normalization(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row_idx, row in enumerate(X):
        area = sum(row)
        if area == 0:
            raise PreprocessMethodError("AREA_ZERO_SUM", "Area normalization cannot transform a spectrum with zero area.", sample_index=row_idx)
        output.append([value / area for value in row])
    return output, {"method": "area_normalization", "parameters": {"area": "sum"}, "fitted": {"scope": "per_sample"}}


def _apply_max_abs_normalization(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row_idx, row in enumerate(X):
        scale = max(abs(value) for value in row)
        if scale == 0:
            raise PreprocessMethodError("MAX_ABS_ZERO_SCALE", "Max-abs normalization cannot transform a zero spectrum.", sample_index=row_idx)
        output.append([value / scale for value in row])
    return output, {"method": "max_abs_normalization", "parameters": {}, "fitted": {"scope": "per_sample"}}


def _apply_savgol(
    X: list[list[float]],
    *,
    method: str,
    window_length: int | None,
    polyorder: int | None,
) -> tuple[list[list[float]], dict[str, Any]]:
    if savgol_filter is None:
        raise PreprocessMethodError("SCIPY_REQUIRED", "Savitzky-Golay preprocessing requires scipy.")
    _validate_sg_params(X, window_length, polyorder)
    deriv = 0 if method == "sg_smoothing" else 1 if method == "first_derivative" else 2
    output = [
        [float(value) for value in savgol_filter(row, window_length=window_length, polyorder=polyorder, deriv=deriv, mode="interp")]
        for row in X
    ]
    return output, {"method": method, "parameters": {"window_length": window_length, "polyorder": polyorder, "derivative_order": deriv}, "fitted": {"scope": "deterministic_per_spectrum"}}


def _apply_moving_average(X: list[list[float]], *, window_length: int | None) -> tuple[list[list[float]], dict[str, Any]]:
    window = _smoothing_window(window_length)
    radius = window // 2
    output = []
    for row in X:
        smoothed = []
        for idx in range(len(row)):
            start = max(0, idx - radius)
            end = min(len(row), idx + radius + 1)
            smoothed.append(sum(row[start:end]) / (end - start))
        output.append(smoothed)
    return output, {"method": "moving_average", "parameters": {"window_length": window}, "fitted": {"scope": "deterministic_per_spectrum"}}


def _apply_gaussian_smoothing(X: list[list[float]], *, sigma: float | None) -> tuple[list[list[float]], dict[str, Any]]:
    if gaussian_filter1d is None:
        raise PreprocessMethodError("SCIPY_REQUIRED", "Gaussian smoothing requires scipy.")
    selected_sigma = 1.0 if sigma is None else float(sigma)
    if selected_sigma <= 0:
        raise PreprocessMethodError("GAUSSIAN_SIGMA_INVALID", "Gaussian smoothing sigma must be positive.", sigma=selected_sigma)
    output = [[float(value) for value in gaussian_filter1d(row, sigma=selected_sigma, mode="nearest")] for row in X]
    return output, {"method": "gaussian_smoothing", "parameters": {"sigma": selected_sigma}, "fitted": {"scope": "deterministic_per_spectrum"}}


def _apply_median_filter(X: list[list[float]], *, window_length: int | None) -> tuple[list[list[float]], dict[str, Any]]:
    if median_filter is None:
        raise PreprocessMethodError("SCIPY_REQUIRED", "Median filter preprocessing requires scipy.")
    window = _smoothing_window(window_length)
    output = [[float(value) for value in median_filter(row, size=window, mode="nearest")] for row in X]
    return output, {"method": "median_filter", "parameters": {"window_length": window}, "fitted": {"scope": "deterministic_per_spectrum"}}


def _apply_detrend(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    coefficients = []
    x = list(range(len(X[0]) if X else 0))
    x_mean = sum(x) / len(x)
    x_ss = sum((value - x_mean) ** 2 for value in x)
    if x_ss == 0:
        raise PreprocessMethodError("DETREND_TOO_FEW_BANDS", "Detrend requires at least two bands.")
    for row in X:
        y_mean = sum(row) / len(row)
        slope = sum((x[idx] - x_mean) * (row[idx] - y_mean) for idx in range(len(row))) / x_ss
        intercept = y_mean - slope * x_mean
        trend_mean = sum(intercept + slope * value for value in x) / len(x)
        output.append([value - (intercept + slope * x[idx]) + trend_mean for idx, value in enumerate(row)])
        coefficients.append({"intercept": intercept, "slope": slope})
    return output, {"method": "detrend", "parameters": {"degree": 1, "preserve_mean": True}, "fitted": {"scope": "per_sample", "coefficients": coefficients}}


def _apply_linear_baseline(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row in X:
        if len(row) < 2:
            raise PreprocessMethodError("LINEAR_BASELINE_TOO_FEW_BANDS", "Linear baseline correction requires at least two bands.")
        slope = (row[-1] - row[0]) / (len(row) - 1)
        output.append([value - (row[0] + slope * idx) for idx, value in enumerate(row)])
    return output, {"method": "linear_baseline", "parameters": {"anchor": "endpoints"}, "fitted": {"scope": "per_sample"}}


def _apply_polynomial_baseline(X: list[list[float]], *, degree: int | None) -> tuple[list[list[float]], dict[str, Any]]:
    if np is None:
        raise PreprocessMethodError("NUMPY_REQUIRED", "Polynomial baseline correction requires numpy.")
    selected_degree = 2 if degree is None else int(degree)
    n_features = len(X[0]) if X else 0
    if selected_degree < 1 or selected_degree >= n_features:
        raise PreprocessMethodError("POLYNOMIAL_BASELINE_DEGREE_INVALID", "Polynomial baseline degree must be at least 1 and less than band count.", degree=selected_degree, n_features=n_features)
    x = np.arange(n_features, dtype=float)
    output = []
    coefficients = []
    for row in X:
        coeff = np.polyfit(x, np.asarray(row, dtype=float), deg=selected_degree)
        baseline = np.polyval(coeff, x)
        output.append([float(row[idx] - baseline[idx]) for idx in range(n_features)])
        coefficients.append([float(value) for value in coeff])
    return output, {"method": "polynomial_baseline", "parameters": {"degree": selected_degree}, "fitted": {"scope": "per_sample", "coefficients": coefficients}}


def _apply_rubberband_baseline(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    hulls = []
    for row in X:
        hull = _lower_hull([(float(idx), value) for idx, value in enumerate(row)])
        baseline = _interpolate_hull(hull, len(row))
        output.append([value - baseline[idx] for idx, value in enumerate(row)])
        hulls.append([int(point[0]) for point in hull])
    return output, {"method": "rubberband_baseline", "parameters": {}, "fitted": {"scope": "per_sample", "hull_indices": hulls}}


def _apply_als_baseline(
    X: list[list[float]],
    *,
    lam: float | None,
    p: float | None,
    n_iter: int | None,
) -> tuple[list[list[float]], dict[str, Any]]:
    if np is None or sparse is None or spsolve is None:
        raise PreprocessMethodError("SCIPY_REQUIRED", "ALS baseline correction requires numpy and scipy.")
    selected_lam = 100000.0 if lam is None else float(lam)
    selected_p = 0.001 if p is None else float(p)
    selected_iter = 10 if n_iter is None else int(n_iter)
    if selected_lam <= 0 or not (0 < selected_p < 1) or selected_iter <= 0:
        raise PreprocessMethodError("ALS_PARAMETERS_INVALID", "ALS baseline requires lambda > 0, 0 < p < 1, and positive iterations.", lam=selected_lam, p=selected_p, n_iter=selected_iter)
    n_features = len(X[0]) if X else 0
    if n_features < 3:
        raise PreprocessMethodError("ALS_TOO_FEW_BANDS", "ALS baseline requires at least three bands.")
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n_features - 2, n_features))
    smoothness = selected_lam * diff.T @ diff
    output = []
    for row in X:
        y = np.asarray(row, dtype=float)
        weights = np.ones(n_features)
        baseline = y
        for _ in range(selected_iter):
            W = sparse.spdiags(weights, 0, n_features, n_features)
            baseline = spsolve(W + smoothness, weights * y)
            weights = selected_p * (y > baseline) + (1 - selected_p) * (y <= baseline)
        output.append([float(y[idx] - baseline[idx]) for idx in range(n_features)])
    return output, {"method": "als_baseline", "parameters": {"lambda": selected_lam, "p": selected_p, "n_iter": selected_iter}, "fitted": {"scope": "per_sample"}}


def _apply_absorbance(X: list[list[float]], *, method: str) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row_idx, row in enumerate(X):
        if any(value <= 0 for value in row):
            raise PreprocessMethodError("ABSORBANCE_NON_POSITIVE_INPUT", "Absorbance conversion requires strictly positive reflectance/transmittance values.", sample_index=row_idx)
        output.append([-math.log10(value) for value in row])
    return output, {"method": method, "parameters": {"formula": "-log10(x)"}, "fitted": {"scope": "deterministic_per_value"}}


def _apply_log_transform(X: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    output = []
    for row_idx, row in enumerate(X):
        if any(value <= 0 for value in row):
            raise PreprocessMethodError("LOG_NON_POSITIVE_INPUT", "Log transform requires strictly positive values.", sample_index=row_idx)
        output.append([math.log(value) for value in row])
    return output, {"method": "log_transform", "parameters": {"base": "e"}, "fitted": {"scope": "deterministic_per_value"}}


def _validate_sg_params(X: list[list[float]], window_length: int | None, polyorder: int | None) -> None:
    if window_length is None or polyorder is None:
        raise PreprocessMethodError("SG_PARAMETERS_REQUIRED", "Please confirm Savitzky-Golay window_length and polyorder, for example window_length=11 and polyorder=2.")
    if window_length <= 0 or window_length % 2 == 0:
        raise PreprocessMethodError("SG_WINDOW_INVALID", "Savitzky-Golay window_length must be a positive odd integer.", window_length=window_length)
    if polyorder < 0 or polyorder >= window_length:
        raise PreprocessMethodError("SG_POLYORDER_INVALID", "Savitzky-Golay polyorder must be non-negative and less than window_length.", polyorder=polyorder, window_length=window_length)
    n_features = len(X[0]) if X else 0
    if window_length > n_features:
        raise PreprocessMethodError("SG_WINDOW_TOO_LONG", "Savitzky-Golay window_length cannot exceed the number of bands.", window_length=window_length, n_features=n_features)


def _smoothing_window(window_length: int | None) -> int:
    window = 5 if window_length is None else int(window_length)
    if window <= 0 or window % 2 == 0:
        raise PreprocessMethodError("SMOOTHING_WINDOW_INVALID", "Smoothing window_length must be a positive odd integer.", window_length=window)
    return window


def _column_means(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise PreprocessMethodError("TRAIN_SPLIT_EMPTY", "Cannot fit preprocessing parameters without train samples.")
    n_features = len(rows[0])
    return [sum(row[col_idx] for row in rows) / len(rows) for col_idx in range(n_features)]


def _column_stds(rows: list[list[float]], means: list[float]) -> list[float]:
    return [
        math.sqrt(sum((row[col_idx] - means[col_idx]) ** 2 for row in rows) / len(rows))
        for col_idx in range(len(means))
    ]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PreprocessMethodError("TRAIN_SPLIT_EMPTY", "Cannot compute quantiles without train samples.")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _lower_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    hull: list[tuple[float, float]] = []
    for point in points:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull


def _cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])


def _interpolate_hull(hull: list[tuple[float, float]], n_features: int) -> list[float]:
    if len(hull) < 2:
        return [hull[0][1] if hull else 0.0 for _ in range(n_features)]
    baseline = [0.0 for _ in range(n_features)]
    for left, right in zip(hull, hull[1:]):
        start = int(left[0])
        end = int(right[0])
        width = max(1, end - start)
        for idx in range(start, end + 1):
            fraction = (idx - start) / width
            baseline[idx] = left[1] * (1 - fraction) + right[1] * fraction
    return baseline


def _assert_shape_unchanged(before: list[list[float]], after: list[list[float]]) -> None:
    if len(before) != len(after):
        raise PreprocessMethodError("PREPROCESS_SAMPLE_COUNT_CHANGED", "Preprocess MVP must not change sample count.")
    if after and any(len(row) != len(after[0]) for row in after):
        raise PreprocessMethodError("PREPROCESS_ROW_WIDTH_MISMATCH", "Preprocessing produced inconsistent row widths.")

"""Standard non-destructive spectral QC checks."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .io import SpectralQCPackage, numeric_matrix


STANDARD_OUTLIER_METHODS = [
    "robust_zscore",
    "pca_hotelling_t2",
    "pca_q_residual",
    "mahalanobis_on_pca",
    "similarity_to_mean",
    "classwise_similarity",
    "baseline_drift_score",
    "spike_detection",
]


def run_basic_checks(
    package: SpectralQCPackage,
    *,
    mode: str = "check",
    low_variance_threshold: float = 1e-12,
    intensity_z_threshold: float = 3.5,
) -> dict[str, Any]:
    matrix = numeric_matrix(package)
    task_type = _task_type(package)
    checks = [
        contract_consistency_check(package),
        missing_value_check(package),
        numeric_quality_check(package),
        band_axis_check(package),
        constant_band_check(package),
        low_variance_band_check(package, threshold=low_variance_threshold),
        label_or_target_check(package, task_type=task_type),
        sample_quality_check(package, matrix=matrix, threshold=intensity_z_threshold),
        spike_check(package, matrix=matrix),
        baseline_drift_check(package, matrix=matrix),
        similarity_check(package, matrix=matrix, task_type=task_type),
        near_duplicate_check(package, matrix=matrix),
        pca_outlier_check(package, matrix=matrix),
    ]
    _refine_spike_check(checks, package=package)
    blocked_reasons = [issue for check in checks for issue in check.get("blocked_reasons", [])]
    warnings = [issue for check in checks for issue in check.get("warnings", [])]
    outlier_reasons = _collect_outlier_reasons(checks)
    high, medium, low, preview, outlier_records = _classify_outliers(package, outlier_reasons)
    outlier_detection = _outlier_detection_summary(outlier_records)
    recommended_actions = _recommended_actions(checks, high, medium, warnings, blocked_reasons)
    status = "blocked" if blocked_reasons else "warning" if warnings or high or medium else "passed"
    summary = {
        "missing_values": _find_check(checks, "missing_check").get("total_missing_values", 0),
        "constant_bands": _find_check(checks, "constant_band_check").get("constant_band_count", 0),
        "low_variance_bands": _find_check(checks, "low_variance_band_check").get("low_variance_band_count", 0),
        "high_confidence_outliers": len(high),
        "medium_confidence_outliers": len(medium),
        "exact_duplicate_pairs": _find_check(checks, "near_duplicate_check").get("exact_duplicate_pair_count", 0),
        "strict_near_duplicate_pairs": _find_check(checks, "near_duplicate_check").get("strict_near_duplicate_pair_count", 0),
        "near_duplicate_pairs": _find_check(checks, "near_duplicate_check").get("near_duplicate_pair_count", 0),
        "label_warnings": len(_find_check(checks, "label_distribution_check").get("warnings", [])),
    }
    duplicate_source = _find_check(checks, "near_duplicate_check")
    duplicate_check = _duplicate_result_summary(duplicate_source)
    global_similarity_risk = _global_similarity_risk_summary(duplicate_source)
    result = {
        "schema_version": "0.1.0",
        "stage": "spectral-check",
        "mode": mode,
        "status": status,
        "input_package": str(package.root),
        "output_package": None,
        "next_package_for_downstream": str(package.root),
        "package_dir": str(package.root),
        "data_shape": {"n_samples": package.n_samples, "n_features": package.n_features},
        "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
        "task_type": task_type,
        "checks_run": [check["check"] for check in checks],
        "checks": checks,
        "summary": summary,
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "duplicate_check": duplicate_check,
        "global_similarity_risk": global_similarity_risk,
        "recommended_actions": recommended_actions,
        "requires_user_confirmation": any(action.get("confirmation_required") for action in recommended_actions),
        "next_step_recommendation": "blocked" if blocked_reasons else "splitter",
        "outlier_detection": outlier_detection,
        "outlier_preview": preview,
        "outlier_groups": {
            "high_confidence_outliers": high,
            "medium_confidence_outliers": medium,
            "low_confidence_outliers": low,
        },
    }
    return result


def contract_consistency_check(package: SpectralQCPackage) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    shape = package.contract.get("shape") or {}
    _expect_count(blocked, "contract.shape.n_samples", shape.get("n_samples"), package.n_samples)
    _expect_count(blocked, "contract.shape.n_features", shape.get("n_features"), package.n_features)
    _expect_count(blocked, "contract.n_samples", package.contract.get("n_samples"), package.n_samples)
    _expect_count(blocked, "contract.n_features", package.contract.get("n_features"), package.n_features)
    band_axis = package.contract.get("band_axis") if isinstance(package.contract.get("band_axis"), dict) else {}
    _expect_count(blocked, "contract.band_axis.count", band_axis.get("count"), package.n_features)
    empty_ids = [idx for idx, sample_id in enumerate(package.sample_ids) if not str(sample_id).strip()]
    duplicated_ids = [sample_id for sample_id, count in Counter(package.sample_ids).items() if count > 1]
    if empty_ids:
        blocked.append(_issue("SAMPLE_ID_EMPTY", "sample_id contains empty values.", sample_indices=empty_ids[:20], count=len(empty_ids)))
    if duplicated_ids:
        blocked.append(_issue("SAMPLE_ID_DUPLICATED", "sample_id contains duplicates.", sample_ids=duplicated_ids[:20], count=len(duplicated_ids)))
    task_hint = str(package.contract.get("task_hint") or "").lower()
    if package.y is None and ("class" in task_hint or "regression" in task_hint):
        blocked.append(_issue("Y_REQUIRED_FOR_SUPERVISED_TASK", "y.csv is required for classification or regression QC.", task_hint=task_hint))
    return {
        "check": "contract_consistency_check",
        "status": "blocked" if blocked else "passed",
        "warnings": warnings,
        "blocked_reasons": blocked,
        "files_present": {"X": True, "sample_ids": True, "band_axis": True, "data_contract": True, "y": package.y is not None},
    }


def missing_value_check(package: SpectralQCPackage) -> dict[str, Any]:
    total = 0
    per_sample: list[dict[str, Any]] = []
    per_band_counts = [0 for _ in package.feature_names]
    for idx, row in enumerate(package.X):
        count = sum(1 for value in row if value is None)
        total += count
        per_sample.append({"sample_id": package.sample_ids[idx], "missing_count": count, "missing_rate": _rate(count, package.n_features)})
        for col_idx, value in enumerate(row):
            if value is None:
                per_band_counts[col_idx] += 1
    per_band = [
        {"band": package.feature_names[idx], "band_axis": package.band_axis[idx], "missing_count": count, "missing_rate": _rate(count, package.n_samples)}
        for idx, count in enumerate(per_band_counts)
        if count
    ]
    warnings = []
    if total:
        warnings.append(_issue("MISSING_VALUES_PRESENT", "X contains missing, NaN, inf, or -inf values.", total_missing_values=total))
    return {
        "check": "missing_check",
        "total_missing_values": total,
        "missing_rate": _rate(total, package.n_samples * package.n_features),
        "samples_with_missing": [item for item in per_sample if item["missing_count"]][:20],
        "bands_with_missing": per_band[:20],
        "warnings": warnings,
        "blocked_reasons": [],
    }


def numeric_quality_check(package: SpectralQCPackage) -> dict[str, Any]:
    matrix = numeric_matrix(package, fill="zero")
    all_zero_samples = [idx for idx, row in enumerate(matrix) if all(value == 0 for value in row)]
    all_zero_bands = [idx for idx in range(package.n_features) if all(row[idx] == 0 for row in matrix)]
    values = [abs(value) for row in matrix for value in row]
    max_abs = max(values) if values else 0.0
    warnings = []
    if all_zero_samples:
        warnings.append(_issue("ALL_ZERO_SAMPLES", "Some spectra are entirely zero.", count=len(all_zero_samples)))
    if max_abs > 1e12:
        warnings.append(_issue("EXTREME_NUMERIC_RANGE", "Spectral intensity range is extremely large.", max_abs=max_abs))
    return {
        "check": "numeric_quality_check",
        "non_numeric_values": 0,
        "illegal_numeric_values": _find_check([missing_value_check(package)], "missing_check").get("total_missing_values", 0),
        "all_zero_sample_count": len(all_zero_samples),
        "all_zero_band_count": len(all_zero_bands),
        "max_abs_value": max_abs,
        "warnings": warnings,
        "blocked_reasons": [],
    }


def band_axis_check(package: SpectralQCPackage) -> dict[str, Any]:
    values = [_to_float(value) for value in package.band_axis]
    numeric = all(value is not None for value in values)
    warnings: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    duplicates = [value for value, count in Counter(package.band_axis).items() if count > 1]
    diffs = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)] if numeric else []
    increasing = bool(diffs) and all(diff > 0 for diff in diffs)
    decreasing = bool(diffs) and all(diff < 0 for diff in diffs)
    monotonic = increasing or decreasing or package.n_features <= 1
    unit = _band_unit(package)
    if not numeric:
        warnings.append(_issue("BAND_AXIS_NON_NUMERIC", "band_axis is not numeric; downstream wavelength/wavenumber logic may be limited."))
    if duplicates:
        warnings.append(_issue("BAND_AXIS_DUPLICATED", "band_axis contains duplicate values.", duplicate_count=len(duplicates), preview=duplicates[:20]))
    if numeric and not monotonic:
        warnings.append(_issue("BAND_AXIS_NOT_MONOTONIC", "Numeric band_axis is not strictly increasing or decreasing."))
    if unit in {"", "unknown", "none", None}:
        warnings.append(_issue("BAND_AXIS_UNIT_UNKNOWN", "band_axis unit is unknown."))
    gap_warning = None
    if numeric and diffs:
        abs_diffs = [abs(diff) for diff in diffs if diff != 0]
        median_gap = _median(abs_diffs) if abs_diffs else 0
        max_gap = max(abs_diffs) if abs_diffs else 0
        if median_gap > 0 and max_gap > median_gap * 10:
            gap_warning = _issue("BAND_AXIS_LARGE_GAP", "band_axis has an unusually large adjacent gap.", max_gap=max_gap, median_gap=median_gap)
            warnings.append(gap_warning)
    return {
        "check": "band_axis_check",
        "is_numeric": numeric,
        "is_monotonic": monotonic,
        "direction": "increasing" if increasing else "decreasing" if decreasing else "unknown",
        "unit": unit,
        "duplicate_band_count": len(duplicates),
        "large_gap": gap_warning,
        "warnings": warnings,
        "blocked_reasons": blocked,
    }


def constant_band_check(package: SpectralQCPackage) -> dict[str, Any]:
    candidates = []
    for idx, name in enumerate(package.feature_names):
        values = [row[idx] for row in package.X if row[idx] is not None]
        if values and len(set(values)) <= 1:
            candidates.append({"band_index": idx, "band": name, "band_axis": package.band_axis[idx]})
    return {"check": "constant_band_check", "constant_band_count": len(candidates), "constant_bands": candidates[:20], "warnings": [], "blocked_reasons": []}


def low_variance_band_check(package: SpectralQCPackage, *, threshold: float) -> dict[str, Any]:
    candidates = []
    for idx, name in enumerate(package.feature_names):
        values = [float(row[idx]) for row in package.X if row[idx] is not None]
        if len(values) < 2:
            continue
        variance = _variance(values)
        if variance <= threshold:
            candidates.append({"band_index": idx, "band": name, "band_axis": package.band_axis[idx], "variance": variance})
    return {"check": "low_variance_band_check", "threshold": threshold, "low_variance_band_count": len(candidates), "low_variance_bands": candidates[:20], "warnings": [], "blocked_reasons": []}


def label_or_target_check(package: SpectralQCPackage, *, task_type: str) -> dict[str, Any]:
    if not package.y or not package.y_header:
        warning = _issue("Y_ABSENT", "y.csv is absent; supervised checks are not available.")
        return {"check": "label_distribution_check", "status": "not_applicable", "warnings": [warning], "blocked_reasons": []}
    values = [row[0] if row else "" for row in package.y]
    if task_type == "regression" or (task_type != "classification" and all(_is_number(value) for value in values)):
        return regression_target_check(package, values)
    return class_distribution_check(package, values)


def class_distribution_check(package: SpectralQCPackage, values: list[Any] | None = None) -> dict[str, Any]:
    labels = [str(value).strip() for value in (values if values is not None else [row[0] if row else "" for row in package.y or []])]
    counts = dict(Counter(labels))
    empty = counts.pop("", 0)
    smallest = min(counts.values()) if counts else 0
    largest = max(counts.values()) if counts else 0
    imbalance = (largest / smallest) if smallest else None
    warnings: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    if empty:
        warnings.append(_issue("EMPTY_LABELS", "Some labels are empty.", count=empty))
    if smallest and smallest < 3:
        blocked.append(_issue("CLASS_COUNT_TOO_SMALL", "At least one class has fewer than 3 samples.", min_class_count=smallest))
    elif smallest and smallest < 5:
        warnings.append(_issue("CLASS_COUNT_LOW", "At least one class has fewer than 5 samples; evaluation may be unstable.", min_class_count=smallest))
    if imbalance and imbalance > 10:
        warnings.append(_issue("CLASS_IMBALANCE_SEVERE", "Class imbalance ratio is greater than 10.", imbalance_ratio=imbalance))
    elif imbalance and imbalance > 5:
        warnings.append(_issue("CLASS_IMBALANCE", "Class imbalance ratio is greater than 5.", imbalance_ratio=imbalance))
    return {
        "check": "label_distribution_check",
        "label": package.y_header[0] if package.y_header else "y",
        "class_count": len(counts),
        "class_counts": counts,
        "min_class_count": smallest,
        "max_class_count": largest,
        "imbalance_ratio": imbalance,
        "empty_label_count": empty,
        "warnings": warnings,
        "blocked_reasons": blocked,
    }


def regression_target_check(package: SpectralQCPackage, values: list[Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    numeric: list[float] = []
    missing = 0
    for value in values:
        if _is_number(value):
            numeric.append(float(value))
        else:
            missing += 1
    if missing:
        blocked.append(_issue("TARGET_NON_NUMERIC_OR_MISSING", "Regression target contains missing or non-numeric values.", count=missing))
    std = _std(numeric)
    if std <= 1e-12 and numeric:
        warnings.append(_issue("TARGET_NEAR_CONSTANT", "Regression target is nearly constant.", std=std))
    outliers = _iqr_candidates(numeric) if numeric else []
    if outliers:
        warnings.append(_issue("TARGET_OUTLIERS", "Regression target has extreme values.", count=len(outliers)))
    return {
        "check": "target_regression_check",
        "target": package.y_header[0] if package.y_header else "y",
        "missing_or_non_numeric": missing,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "mean": _mean(numeric) if numeric else None,
        "std": std,
        "target_outlier_count": len(outliers),
        "warnings": warnings,
        "blocked_reasons": blocked,
    }


def sample_quality_check(package: SpectralQCPackage, *, matrix: list[list[float]], threshold: float) -> dict[str, Any]:
    stats = []
    metrics: dict[str, list[float]] = defaultdict(list)
    for idx, row in enumerate(matrix):
        first_diff = [row[i + 1] - row[i] for i in range(len(row) - 1)]
        second_diff = [first_diff[i + 1] - first_diff[i] for i in range(len(first_diff) - 1)]
        item = {
            "sample_id": package.sample_ids[idx],
            "sample_index": idx,
            "mean": _mean(row),
            "std": _std(row),
            "min": min(row) if row else 0.0,
            "max": max(row) if row else 0.0,
            "range": (max(row) - min(row)) if row else 0.0,
            "area": sum(row),
            "l2_norm": math.sqrt(sum(value * value for value in row)),
            "first_diff_std": _std(first_diff),
            "second_diff_std": _std(second_diff),
            "max_adjacent_jump": max([abs(value) for value in first_diff], default=0.0),
            "roughness": _mean([abs(value) for value in second_diff]) if second_diff else 0.0,
        }
        stats.append(item)
        for key in ["mean", "std", "range", "area", "l2_norm", "first_diff_std", "second_diff_std", "max_adjacent_jump", "roughness"]:
            metrics[key].append(float(item[key]))
    candidates = []
    for metric, values in metrics.items():
        scores = _robust_zscores(values)
        for idx, score in enumerate(scores):
            if abs(score) > threshold:
                candidates.append({"sample_id": package.sample_ids[idx], "sample_index": idx, "metric": metric, "score": score})
    return {
        "check": "sample_intensity_quality",
        "method": "robust_zscore",
        "threshold": threshold,
        "candidate_count": len({item["sample_index"] for item in candidates}),
        "candidate_samples": candidates[:20],
        "sample_stat_preview": stats[:5],
        "warnings": [],
        "blocked_reasons": [],
    }


def spike_check(package: SpectralQCPackage, *, matrix: list[list[float]], window_radius: int = 2, threshold: float = 6.0, score_cap: float = 999.0, mad_epsilon: float = 1e-8) -> dict[str, Any]:
    candidates = []
    severity_groups: dict[str, list[dict[str, Any]]] = {"minor": [], "moderate": [], "severe": []}
    total_capped_scores = 0
    total_small_mad_points = 0
    for sample_idx, row in enumerate(matrix):
        scores = []
        capped_scores = 0
        small_mad_points = 0
        for idx, value in enumerate(row):
            lo = max(0, idx - window_radius)
            hi = min(len(row), idx + window_radius + 1)
            local = row[lo:hi]
            med = _median(local)
            mad = _median([abs(v - med) for v in local])
            delta = abs(value - med)
            if mad < mad_epsilon:
                small_mad_points += 1
                score = 0.0 if delta < mad_epsilon else score_cap
                capped_scores += 1 if score >= score_cap else 0
            else:
                raw_score = delta / (1.4826 * mad)
                score = min(raw_score, score_cap)
                capped_scores += 1 if raw_score > score_cap else 0
            scores.append(score)
        spike_count = sum(1 for score in scores if score > threshold)
        max_score = max(scores, default=0.0)
        if spike_count:
            spike_ratio = _rate(spike_count, package.n_features)
            severity = _initial_spike_severity(max_score=max_score, spike_ratio=spike_ratio, spike_count=spike_count, n_features=package.n_features)
            item = {
                "sample_id": package.sample_ids[sample_idx],
                "sample_index": sample_idx,
                "spike_count": spike_count,
                "max_spike_score": max_score,
                "spike_ratio": spike_ratio,
                "risk_level": severity,
                "corroborating_reasons": [],
                "score_capped": bool(capped_scores),
                "score_capped_count": capped_scores,
                "local_mad_too_small_count": small_mad_points,
            }
            candidates.append(item)
            severity_groups[severity].append(item)
        total_capped_scores += capped_scores
        total_small_mad_points += small_mad_points
    return {
        "check": "spike_check",
        "method": "hampel_local_mad",
        "threshold": threshold,
        "score_cap": score_cap,
        "mad_epsilon": mad_epsilon,
        "score_capped_count": total_capped_scores,
        "local_mad_too_small_count": total_small_mad_points,
        "spike_sample_count": len(candidates),
        "minor_spike_sample_count": len(severity_groups["minor"]),
        "moderate_spike_sample_count": len(severity_groups["moderate"]),
        "severe_spike_sample_count": len(severity_groups["severe"]),
        "spike_samples": candidates[:20],
        "minor_spike_samples": severity_groups["minor"][:20],
        "moderate_spike_samples": severity_groups["moderate"][:20],
        "severe_spike_samples": severity_groups["severe"][:20],
        "warnings": [_issue("SPIKE_CANDIDATES", "Some spectra contain local spike candidates; severe spike status requires independent roughness, PCA residual, or similarity evidence.", count=len(candidates), severe_count=len(severity_groups["severe"]))] if candidates else [],
        "blocked_reasons": [],
    }


def baseline_drift_check(package: SpectralQCPackage, *, matrix: list[list[float]]) -> dict[str, Any]:
    slopes = []
    curvatures = []
    offsets = []
    mean_spectrum = _column_mean(matrix)
    for row in matrix:
        slopes.append((row[-1] - row[0]) / max(1, len(row) - 1))
        diffs = [row[i + 1] - row[i] for i in range(len(row) - 1)]
        second = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        curvatures.append(_mean([abs(value) for value in second]) if second else 0.0)
        offsets.append(_mean([row[idx] - mean_spectrum[idx] for idx in range(len(row))]))
    slope_scores = _robust_zscores(slopes)
    curvature_scores = _robust_zscores(curvatures)
    offset_scores = _robust_zscores(offsets)
    candidates = []
    for idx in range(package.n_samples):
        if max(abs(slope_scores[idx]), abs(curvature_scores[idx]), abs(offset_scores[idx])) > 3.5:
            candidates.append(
                {
                    "sample_id": package.sample_ids[idx],
                    "sample_index": idx,
                    "baseline_slope": slopes[idx],
                    "baseline_curvature": curvatures[idx],
                    "mean_offset": offsets[idx],
                }
            )
    return {
        "check": "baseline_drift_check",
        "baseline_drift_sample_count": len(candidates),
        "baseline_drift_samples": candidates[:20],
        "warnings": [_issue("BASELINE_DRIFT_CANDIDATES", "Some spectra show baseline drift or global offset risk.", count=len(candidates))] if candidates else [],
        "blocked_reasons": [],
    }


def similarity_check(package: SpectralQCPackage, *, matrix: list[list[float]], task_type: str) -> dict[str, Any]:
    mean_spectrum = _column_mean(matrix)
    low_similarity = []
    for idx, row in enumerate(matrix):
        pearson = _pearson(row, mean_spectrum)
        cosine = _cosine(row, mean_spectrum)
        if pearson < 0.95 or cosine < 0.98:
            low_similarity.append({"sample_id": package.sample_ids[idx], "sample_index": idx, "pearson": pearson, "cosine": cosine})
    classwise_outliers = []
    suspected_mislabels = []
    if task_type == "classification" and package.y:
        labels = [str(row[0]) if row else "" for row in package.y]
        class_means = {label: _column_mean([matrix[idx] for idx, value in enumerate(labels) if value == label]) for label in set(labels)}
        for idx, row in enumerate(matrix):
            own = labels[idx]
            own_corr = _pearson(row, class_means[own])
            other = [(label, _pearson(row, avg)) for label, avg in class_means.items() if label != own]
            best_other = max(other, key=lambda item: item[1], default=(None, -1.0))
            if own_corr < 0.95:
                classwise_outliers.append({"sample_id": package.sample_ids[idx], "sample_index": idx, "class": own, "own_class_pearson": own_corr})
            if best_other[0] is not None and best_other[1] - own_corr > 0.05:
                suspected_mislabels.append({"sample_id": package.sample_ids[idx], "sample_index": idx, "class": own, "nearest_other_class": best_other[0], "own_class_pearson": own_corr, "other_class_pearson": best_other[1]})
    return {
        "check": "similarity_to_mean",
        "low_similarity_samples": low_similarity[:20],
        "low_similarity_count": len(low_similarity),
        "suspected_classwise_outliers": classwise_outliers[:20],
        "suspected_mislabels": suspected_mislabels[:20],
        "warnings": [_issue("LOW_SIMILARITY_SAMPLES", "Some spectra have low similarity to the mean or class mean.", count=len(low_similarity) + len(classwise_outliers))] if low_similarity or classwise_outliers else [],
        "blocked_reasons": [],
    }


def near_duplicate_check(package: SpectralQCPackage, *, matrix: list[list[float]]) -> dict[str, Any]:
    exact_pairs = []
    strict_near_pairs = []
    high_similarity_pairs = []
    exact_label_conflicts = []
    strict_label_conflicts = []
    high_similarity_cross_label_pairs = []
    labels = [str(row[0]) if row else "" for row in package.y] if package.y else None
    for i in range(package.n_samples):
        for j in range(i + 1, package.n_samples):
            same = matrix[i] == matrix[j]
            corr = _pearson(matrix[i], matrix[j])
            cos = _cosine(matrix[i], matrix[j])
            relative_rmse = _relative_rmse(matrix[i], matrix[j])
            sam_angle = _sam_angle(matrix[i], matrix[j])
            pair = {
                "sample_ids": [package.sample_ids[i], package.sample_ids[j]],
                "indices": [i, j],
                "pearson": corr,
                "cosine": cos,
                "relative_rmse": relative_rmse,
                "sam_angle": sam_angle,
            }
            if same:
                exact_pairs.append(pair)
                if labels and labels[i] != labels[j]:
                    exact_label_conflicts.append({**pair, "labels": [labels[i], labels[j]]})
                continue
            if _is_strict_near_duplicate(corr=corr, cosine=cos, relative_rmse=relative_rmse, sam_angle=sam_angle):
                strict_near_pairs.append(pair)
                if labels and labels[i] != labels[j]:
                    strict_label_conflicts.append({**pair, "labels": [labels[i], labels[j]]})
                continue
            if corr > 0.999 or cos > 0.999:
                high_similarity_pairs.append(pair)
                if labels and labels[i] != labels[j]:
                    high_similarity_cross_label_pairs.append({**pair, "labels": [labels[i], labels[j]]})
    warnings = []
    blocked = []
    if exact_pairs:
        warnings.append(_issue("EXACT_DUPLICATE_SPECTRA", "Exact duplicate spectra may be replicate scans and can cause train/test leakage if split independently.", pair_count=len(exact_pairs)))
    if strict_near_pairs:
        warnings.append(_issue("STRICT_NEAR_DUPLICATE_SPECTRA", "Strict near-duplicate spectra may be replicate scans; consider group-aware splitting or manual review.", pair_count=len(strict_near_pairs), label_conflict_count=len(strict_label_conflicts)))
    if high_similarity_pairs:
        warnings.append(_issue("HIGH_SIMILARITY_SPECTRA", "Many spectra have highly similar global shape; this suggests possible class overlap or subtle local differences, not confirmed duplicates.", pair_count=len(high_similarity_pairs), cross_label_pair_count=len(high_similarity_cross_label_pairs)))
    if exact_label_conflicts:
        blocked.append(_issue("EXACT_DUPLICATE_LABEL_CONFLICT", "Exact duplicate spectra have conflicting labels; manual review is required.", pair_count=len(exact_label_conflicts)))
    return {
        "check": "near_duplicate_check",
        "exact_duplicate_pair_count": len(exact_pairs),
        "exact_duplicate_pairs": exact_pairs[:20],
        "exact_duplicate_groups": _duplicate_groups(package, exact_pairs),
        "exact_duplicate_label_conflict_count": len(exact_label_conflicts),
        "exact_duplicate_label_conflicts": exact_label_conflicts[:20],
        "strict_near_duplicate_pair_count": len(strict_near_pairs),
        "strict_near_duplicate_pairs": strict_near_pairs[:20],
        "strict_near_duplicate_groups": _duplicate_groups(package, strict_near_pairs),
        "strict_near_duplicate_label_conflict_count": len(strict_label_conflicts),
        "strict_near_duplicate_label_conflicts": strict_label_conflicts[:20],
        "high_similarity_pair_count": len(high_similarity_pairs),
        "high_similarity_pairs": high_similarity_pairs[:20],
        "high_similarity_cross_label_pair_count": len(high_similarity_cross_label_pairs),
        "high_similarity_cross_label_pairs": high_similarity_cross_label_pairs[:20],
        "interpretation": "High correlation or cosine similarity alone means global spectral-shape similarity, not confirmed duplicated samples.",
        "recommended_action": "do_not_delete_by_default",
        "recommended_split_strategy": "stratified_or_group_aware_if_replicates_exist",
        "near_duplicate_pair_count": len(strict_near_pairs),
        "potential_replicates": (exact_pairs + strict_near_pairs)[:20],
        "potential_leakage_groups": _duplicate_groups(package, exact_pairs + strict_near_pairs),
        "label_conflict_pairs": (exact_label_conflicts + strict_label_conflicts)[:20],
        "warnings": warnings,
        "blocked_reasons": blocked,
    }


def pca_outlier_check(package: SpectralQCPackage, *, matrix: list[list[float]]) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception:
        warning = _issue("NUMPY_UNAVAILABLE", "PCA outlier checks require numpy.")
        return {"check": "pca_outlier_check", "status": "skipped", "warnings": [warning], "blocked_reasons": []}
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[0] < 3 or arr.shape[1] < 1:
        return {"check": "pca_outlier_check", "status": "skipped", "reason": "too_few_samples_or_features", "warnings": [], "blocked_reasons": []}
    centered = arr - arr.mean(axis=0)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(10, arr.shape[0] - 1, arr.shape[1], vt.shape[0])
    components = vt[:k].T
    scores = centered @ components
    reconstructed = scores @ components.T
    residual = centered - reconstructed
    score_var = np.var(scores, axis=0, ddof=1)
    score_var[score_var == 0] = 1.0
    t2 = np.sum((scores**2) / score_var, axis=1)
    q_residual = np.sum(residual**2, axis=1)
    cov = np.cov(scores, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-9)
    mahal = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", scores, inv, scores), 0.0))
    t2_thr = _quantile([float(v) for v in t2], 0.975)
    q_thr = _quantile([float(v) for v in q_residual], 0.975)
    m_thr = _quantile([float(v) for v in mahal], 0.975)
    candidates = []
    for idx in range(package.n_samples):
        reasons = []
        if float(t2[idx]) > t2_thr:
            reasons.append("pca_hotelling_t2")
        if float(q_residual[idx]) > q_thr:
            reasons.append("pca_q_residual")
        if float(mahal[idx]) > m_thr:
            reasons.append("mahalanobis_on_pca")
        if reasons:
            candidates.append({"sample_id": package.sample_ids[idx], "sample_index": idx, "reasons": reasons, "t2": float(t2[idx]), "q_residual": float(q_residual[idx]), "mahalanobis_on_pca": float(mahal[idx])})
    return {
        "check": "pca_outlier_check",
        "n_components": k,
        "thresholds": {"hotelling_t2": t2_thr, "q_residual": q_thr, "mahalanobis_on_pca": m_thr},
        "candidate_count": len(candidates),
        "outlier_candidates": candidates[:20],
        "warnings": [],
        "blocked_reasons": [],
    }


def target_outlier_check(package: SpectralQCPackage) -> dict[str, Any]:
    if not package.y or not package.y_header:
        return {"check": "target_outliers", "status": "not_applicable", "reason": "y.csv is absent."}
    values = [row[0] if row else "" for row in package.y]
    if not all(_is_number(value) for value in values):
        return {"check": "target_outliers", "status": "not_applicable", "reason": "y is not numeric."}
    result = regression_target_check(package, values)
    return {
        "check": "target_outliers",
        "target": result.get("target"),
        "target_outlier_count": result.get("target_outlier_count", 0),
        "target_outliers": [],
    }


def intensity_outlier_check(package: SpectralQCPackage, *, threshold: float) -> dict[str, Any]:
    check = sample_quality_check(package, matrix=numeric_matrix(package), threshold=threshold)
    return {
        "check": "intensity_outliers",
        "method": "ROBUST_ZSCORE",
        "threshold": threshold,
        "outlier_sample_count": check["candidate_count"],
        "outlier_samples": check["candidate_samples"],
    }


def _collect_outlier_reasons(checks: list[dict[str, Any]]) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = defaultdict(set)
    for item in _find_check(checks, "sample_intensity_quality").get("candidate_samples", []):
        reasons[int(item["sample_index"])].add("robust_zscore")
    for item in _find_check(checks, "spike_check").get("severe_spike_samples", []):
        reasons[int(item["sample_index"])].add("spike_detection")
    for item in _find_check(checks, "baseline_drift_check").get("baseline_drift_samples", []):
        reasons[int(item["sample_index"])].add("baseline_drift_score")
    for item in _find_check(checks, "similarity_to_mean").get("low_similarity_samples", []):
        reasons[int(item["sample_index"])].add("similarity_to_mean")
    for item in _find_check(checks, "similarity_to_mean").get("suspected_classwise_outliers", []):
        reasons[int(item["sample_index"])].add("classwise_similarity")
    for item in _find_check(checks, "similarity_to_mean").get("suspected_mislabels", []):
        reasons[int(item["sample_index"])].add("suspected_mislabel")
    for item in _find_check(checks, "pca_outlier_check").get("outlier_candidates", []):
        for reason in item.get("reasons", []):
            reasons[int(item["sample_index"])].add(str(reason))
    return reasons


def _classify_outliers(package: SpectralQCPackage, reasons: dict[int, set[str]]) -> tuple[list[str], list[str], list[str], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    high: list[str] = []
    medium: list[str] = []
    low: list[str] = []
    preview: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {"high": [], "medium": [], "low": []}
    for idx, item_reasons in sorted(reasons.items()):
        has_pca = any(reason.startswith("pca_") or reason == "mahalanobis_on_pca" for reason in item_reasons)
        risk = "high" if len(item_reasons) >= 2 and has_pca else "medium" if len(item_reasons) >= 1 else "low"
        sample_id = package.sample_ids[idx]
        record = {
            "sample_id": sample_id,
            "sample_index": idx,
            "risk_level": risk,
            "triggered_by": sorted(item_reasons),
        }
        if risk == "high":
            high.append(sample_id)
        elif risk == "medium":
            medium.append(sample_id)
        else:
            low.append(sample_id)
        records[risk].append(record)
        if len(preview) < 10:
            preview.append({"sample_id": sample_id, "risk_level": risk, "reasons": sorted(item_reasons)})
    return high, medium, low, preview, records


def _outlier_detection_summary(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "strategy": "standard",
        "methods_run": list(STANDARD_OUTLIER_METHODS),
        "advanced_methods_not_run_by_default": ["half_resampling_outlier", "mccv_outlier"],
        "recommended_action": "mark_only" if any(records.values()) else "none",
        "high_confidence_outliers": records.get("high", []),
        "medium_confidence_outliers": records.get("medium", []),
        "low_confidence_outliers": records.get("low", []),
        "method_selection_options": [
            "standard_multi_method_consensus",
            "pca_hotelling_t2_q_residual_pca_md",
            "mahalanobis_on_pca",
            "robust_zscore_mad_iqr",
            "half_resampling_outlier",
            "mccv_outlier",
            "custom",
        ],
    }


def _recommended_actions(checks: list[dict[str, Any]], high: list[str], medium: list[str], warnings: list[dict[str, Any]], blocked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if blocked:
        actions.append({"action": "fix_blocking_qc_issues", "confirmation_required": False, "reason_count": len(blocked)})
    if _find_check(checks, "missing_check").get("total_missing_values", 0):
        actions.append({"action": "review_missing_values", "confirmation_required": False})
    constant_count = _find_check(checks, "constant_band_check").get("constant_band_count", 0)
    if constant_count:
        actions.append({"action": "review_constant_bands", "confirmation_required": False, "candidate_count": constant_count})
    if high or medium:
        actions.append({"action": "mark_outlier_samples", "confirmation_required": False, "high_confidence": len(high), "medium_confidence": len(medium)})
    duplicate_check = _find_check(checks, "near_duplicate_check")
    if duplicate_check.get("exact_duplicate_pair_count", 0) or duplicate_check.get("strict_near_duplicate_pair_count", 0):
        actions.append({"action": "use_group_aware_split_for_replicates", "confirmation_required": False})
    if duplicate_check.get("high_similarity_pair_count", 0):
        actions.append({"action": "review_global_similarity_class_overlap", "confirmation_required": False, "pair_count": duplicate_check.get("high_similarity_pair_count", 0)})
    if any(issue.get("code") == "BASELINE_DRIFT_CANDIDATES" for issue in warnings):
        actions.append({"action": "consider_preprocess_snv_msc_derivative_or_baseline_correction", "confirmation_required": False})
    return actions


def _refine_spike_check(checks: list[dict[str, Any]], *, package: SpectralQCPackage) -> None:
    spike = _find_check(checks, "spike_check")
    if not spike.get("spike_samples"):
        return
    corroboration = _spike_corroboration_by_sample(checks)
    groups: dict[str, list[dict[str, Any]]] = {"minor": [], "moderate": [], "severe": []}
    refined = []
    for item in spike.get("spike_samples", []):
        idx = int(item["sample_index"])
        reasons = sorted(corroboration.get(idx, set()))
        severe_candidate = _passes_strict_severe_spike_gate(item, n_features=package.n_features)
        if severe_candidate and reasons:
            risk = "severe"
        elif severe_candidate or item.get("risk_level") == "moderate":
            risk = "moderate"
        else:
            risk = "minor"
        updated = {**item, "risk_level": risk, "corroborating_reasons": reasons}
        refined.append(updated)
        groups[risk].append(updated)
    spike["spike_samples"] = refined[:20]
    spike["minor_spike_sample_count"] = len(groups["minor"])
    spike["moderate_spike_sample_count"] = len(groups["moderate"])
    spike["severe_spike_sample_count"] = len(groups["severe"])
    spike["minor_spike_samples"] = groups["minor"][:20]
    spike["moderate_spike_samples"] = groups["moderate"][:20]
    spike["severe_spike_samples"] = groups["severe"][:20]
    if refined:
        spike["warnings"] = [
            _issue(
                "SPIKE_CANDIDATES",
                "Some spectra contain local spike candidates; severe spike status requires independent roughness, PCA residual, or similarity evidence.",
                count=len(refined),
                severe_count=len(groups["severe"]),
                moderate_count=len(groups["moderate"]),
            )
        ]


def _spike_corroboration_by_sample(checks: list[dict[str, Any]]) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = defaultdict(set)
    for item in _find_check(checks, "sample_intensity_quality").get("candidate_samples", []):
        metric = str(item.get("metric") or "")
        if metric in {"roughness", "first_diff_std", "second_diff_std", "max_adjacent_jump"}:
            reasons[int(item["sample_index"])].add(metric)
    for item in _find_check(checks, "pca_outlier_check").get("outlier_candidates", []):
        item_reasons = set(str(reason) for reason in item.get("reasons", []))
        if "pca_q_residual" in item_reasons:
            reasons[int(item["sample_index"])].add("pca_q_residual")
    for item in _find_check(checks, "similarity_to_mean").get("low_similarity_samples", []):
        reasons[int(item["sample_index"])].add("low_similarity")
    return reasons


def _passes_strict_severe_spike_gate(item: dict[str, Any], *, n_features: int) -> bool:
    spike_count = int(item.get("spike_count") or 0)
    spike_ratio = float(item.get("spike_ratio") or 0.0)
    max_score = float(item.get("max_spike_score") or 0.0)
    min_count = max(5, math.ceil(n_features * 0.003))
    return spike_count >= min_count and spike_ratio >= 0.003 and max_score >= 15.0


def _duplicate_result_summary(check: dict[str, Any]) -> dict[str, Any]:
    exact_conflicts = int(check.get("exact_duplicate_label_conflict_count") or 0)
    strict_conflicts = int(check.get("strict_near_duplicate_label_conflict_count") or 0)
    return {
        "status": "blocked" if exact_conflicts else "warning" if (check.get("exact_duplicate_pair_count") or check.get("strict_near_duplicate_pair_count")) else "passed",
        "exact_duplicate_pairs": int(check.get("exact_duplicate_pair_count") or 0),
        "exact_duplicate_label_conflicts": exact_conflicts,
        "strict_near_duplicate_pairs": int(check.get("strict_near_duplicate_pair_count") or 0),
        "strict_near_duplicate_label_conflicts": strict_conflicts,
        "interpretation": "Exact and strict near-duplicate checks only; global high similarity is reported separately.",
        "recommended_action": "review_exact_or_strict_duplicates_only" if (exact_conflicts or strict_conflicts or check.get("exact_duplicate_pair_count") or check.get("strict_near_duplicate_pair_count")) else "none",
        "recommended_split_strategy": check.get("recommended_split_strategy") or "standard_split",
    }


def _global_similarity_risk_summary(check: dict[str, Any]) -> dict[str, Any]:
    high_similarity = int(check.get("high_similarity_pair_count") or 0)
    cross_label = int(check.get("high_similarity_cross_label_pair_count") or 0)
    return {
        "status": "warning" if high_similarity else "passed",
        "high_similarity_pairs": high_similarity,
        "cross_label_high_similarity_pairs": cross_label,
        "interpretation": "Global spectral shapes are highly similar; this is class-overlap or subtle-local-difference risk, not duplicate-sample evidence.",
        "recommended_action": "do_not_delete_by_default",
        "recommended_next_step": "continue_to_splitter; consider preprocessing, PCA, or feature selection during modeling",
    }


def _expect_count(blocked: list[dict[str, Any]], field: str, observed: Any, expected: int) -> None:
    if observed is not None and int(observed) != expected:
        blocked.append(_issue("CONTRACT_COUNT_MISMATCH", f"{field} does not match actual files.", field=field, expected=expected, observed=observed))


def _find_check(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((item for item in checks if item.get("check") == name), {})


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _task_type(package: SpectralQCPackage) -> str:
    hint = str(package.contract.get("task_hint") or "").lower()
    if "class" in hint:
        return "classification"
    if "regression" in hint:
        return "regression"
    if package.y:
        values = [row[0] if row else "" for row in package.y]
        return "regression" if values and all(_is_number(value) for value in values) else "classification"
    return "unsupervised"


def _band_unit(package: SpectralQCPackage) -> str | None:
    axis = package.contract.get("band_axis") if isinstance(package.contract.get("band_axis"), dict) else {}
    if axis.get("unit"):
        return str(axis.get("unit"))
    header = [str(item).strip().lower() for item in package.band_axis_header]
    if "unit" in header:
        return "from_band_axis"
    return package.contract.get("band_unit") or "unknown"


def _to_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _initial_spike_severity(*, max_score: float, spike_ratio: float, spike_count: int, n_features: int) -> str:
    if _passes_strict_severe_spike_gate({"max_spike_score": max_score, "spike_ratio": spike_ratio, "spike_count": spike_count}, n_features=n_features):
        return "moderate"
    if max_score >= 8.0 or spike_ratio >= 0.003:
        return "moderate"
    return "minor"


def _is_strict_near_duplicate(*, corr: float, cosine: float, relative_rmse: float, sam_angle: float) -> bool:
    return corr > 0.9999 and cosine > 0.9999 and relative_rmse < 1e-3 and sam_angle < 0.01


def _relative_rmse(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    rmse = (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5
    scale = max((_rms(a) + _rms(b)) / 2.0, 1e-12)
    return rmse / scale


def _sam_angle(a: list[float], b: list[float]) -> float:
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if den == 0:
        return 0.0
    cos = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / den))
    return math.acos(cos)


def _rms(values: list[float]) -> float:
    return (sum(value * value for value in values) / len(values)) ** 0.5 if values else 0.0


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return _variance(values) ** 0.5


def _variance(values: list[float]) -> float:
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _robust_zscores(values: list[float]) -> list[float]:
    med = _median(values)
    deviations = [abs(value - med) for value in values]
    mad = _median(deviations)
    scale = 1.4826 * mad
    if scale == 0:
        scale = max(_variance(values) ** 0.5, 1e-12)
    return [(value - med) / scale for value in values]


def _iqr_candidates(values: list[float], multiplier: float = 1.5) -> list[int]:
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [idx for idx, value in enumerate(values) if value < lower or value > upper]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _column_mean(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return []
    return [_mean([row[idx] for row in matrix]) for idx in range(len(matrix[0]))]


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    va = [x - ma for x in a]
    vb = [x - mb for x in b]
    den = math.sqrt(sum(x * x for x in va) * sum(y * y for y in vb))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(va, vb)) / den


def _cosine(a: list[float], b: list[float]) -> float:
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def _duplicate_groups(package: SpectralQCPackage, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[set[int]] = []
    for pair in pairs:
        pair_set = set(pair.get("indices", []))
        merged = False
        for group in groups:
            if group & pair_set:
                group.update(pair_set)
                merged = True
                break
        if not merged:
            groups.append(set(pair_set))
    return [{"sample_ids": [package.sample_ids[idx] for idx in sorted(group)], "indices": sorted(group)} for group in groups[:20]]

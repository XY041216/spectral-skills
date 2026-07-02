"""Registry for compact QC method selection."""

from __future__ import annotations

from typing import Any


DETECTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "NOE": {
        "method_id": "NOE",
        "display_name": "No Outlier Elimination",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 1,
        "default_parameters": {},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "MD": {
        "method_id": "MD",
        "display_name": "Mahalanobis Distance",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 5,
        "default_parameters": {"threshold": "quantile_0.975", "ridge": 1e-9},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "PCA_DISTANCE": {
        "method_id": "PCA_DISTANCE",
        "display_name": "PCA Score Distance",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 5,
        "default_parameters": {"components": "auto", "threshold": "quantile_0.975"},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "ROBUST_ZSCORE": {
        "method_id": "ROBUST_ZSCORE",
        "display_name": "Robust Z-score",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 3,
        "default_parameters": {"threshold": 3.5},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "IQR": {
        "method_id": "IQR",
        "display_name": "Interquartile Range",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 3,
        "default_parameters": {"multiplier": 1.5},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "MAD": {
        "method_id": "MAD",
        "display_name": "Median Absolute Deviation",
        "requires_y": False,
        "supported_task_types": ["classification", "regression", "unsupervised", "unknown"],
        "min_samples": 3,
        "default_parameters": {"threshold": 3.5},
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold"],
        "confirmation_required_for_removal": True,
    },
    "HALF_RESAMPLING_OUTLIER": {
        "method_id": "half_resampling_outlier",
        "display_name": "Half Resampling Outlier Stability",
        "requires_y": True,
        "supported_task_types": ["classification", "regression"],
        "min_samples": 5,
        "default_parameters": {
            "n_resamples": 100,
            "sample_fraction": 0.5,
            "base_model": "auto",
            "outlier_metric": "auto",
            "threshold": "percentile_95",
        },
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold", "resampling_outlier_control"],
        "confirmation_required_for_removal": True,
    },
    "MCCV_OUTLIER": {
        "method_id": "mccv_outlier",
        "display_name": "Monte Carlo Cross-Validation Outlier Stability",
        "requires_y": True,
        "supported_task_types": ["classification", "regression"],
        "min_samples": 5,
        "default_parameters": {
            "n_resamples": 100,
            "train_ratio": 0.7,
            "base_model": "auto",
            "outlier_metric": "auto",
            "threshold": "percentile_95",
        },
        "outputs": ["outlier_sample_candidates", "outlier_scores", "threshold", "resampling_outlier_control"],
        "confirmation_required_for_removal": True,
    },
}


ALIASES = {
    "MAHALANOBIS": "MD",
    "MAHALANOBIS_DISTANCE": "MD",
    "PCA": "PCA_DISTANCE",
    "PCA_SCORE_DISTANCE": "PCA_DISTANCE",
    "ROBUST-ZSCORE": "ROBUST_ZSCORE",
    "ROBUST_Z": "ROBUST_ZSCORE",
    "HR": "HALF_RESAMPLING_OUTLIER",
    "HALF_RESAMPLING": "HALF_RESAMPLING_OUTLIER",
    "HALF-RESAMPLING-OUTLIER": "HALF_RESAMPLING_OUTLIER",
    "MCCV": "MCCV_OUTLIER",
    "MONTE_CARLO_CV": "MCCV_OUTLIER",
    "MONTE_CARLO_CROSS_VALIDATION": "MCCV_OUTLIER",
}


def list_methods() -> list[dict[str, Any]]:
    return [dict(value) for value in DETECTOR_REGISTRY.values()]


def normalize_method_id(method_id: str) -> str:
    key = method_id.strip().upper().replace(" ", "_")
    return ALIASES.get(key, key)


def get_method(method_id: str) -> dict[str, Any]:
    key = normalize_method_id(method_id)
    if key not in DETECTOR_REGISTRY:
        raise KeyError(key)
    return dict(DETECTOR_REGISTRY[key])

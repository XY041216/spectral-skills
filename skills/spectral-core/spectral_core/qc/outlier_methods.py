"""First-phase sample outlier methods for spectral QC."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .io import SpectralQCPackage, numeric_matrix
from .detector_registry import get_method, normalize_method_id


class QCMethodError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def run_outlier_methods(package: SpectralQCPackage, methods: list[str] | None = None, *, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    method_ids = [normalize_method_id(method) for method in (methods or ["NOE"])]
    outputs = []
    for method_id in method_ids:
        outputs.append(run_outlier_method(package, method_id, parameters=parameters or {}))
    return {
        "status": "ready",
        "package_dir": str(package.root),
        "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
        "methods": outputs,
        "recommended_actions": _recommended_actions(outputs),
    }


def run_outlier_method(package: SpectralQCPackage, method_id: str, *, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    method = get_method(method_id)
    method_key = method["method_id"]
    if package.n_samples < int(method["min_samples"]):
        raise QCMethodError("METHOD_MIN_SAMPLES_NOT_MET", f"{method_key} requires at least {method['min_samples']} samples.", method=method_key, observed=package.n_samples)
    params = dict(method.get("default_parameters") or {})
    params.update(parameters or {})
    if method_key == "NOE":
        scores = [0.0 for _ in package.sample_ids]
        candidates: list[int] = []
        threshold: float | None = None
    elif method_key == "MD":
        scores = _mahalanobis_scores(numeric_matrix(package), ridge=float(params.get("ridge", 1e-9)))
        threshold = _threshold(scores, params.get("threshold", "quantile_0.975"))
        candidates = [idx for idx, score in enumerate(scores) if score > threshold]
    elif method_key == "PCA_DISTANCE":
        scores = _pca_distance_scores(numeric_matrix(package), params.get("components", "auto"))
        threshold = _threshold(scores, params.get("threshold", "quantile_0.975"))
        candidates = [idx for idx, score in enumerate(scores) if score > threshold]
    elif method_key == "ROBUST_ZSCORE":
        scores = _robust_zscores(_row_intensities(numeric_matrix(package)))
        threshold = float(params.get("threshold", 3.5))
        candidates = [idx for idx, score in enumerate(scores) if abs(score) > threshold]
    elif method_key == "IQR":
        intensities = _row_intensities(numeric_matrix(package))
        lower, upper = _iqr_bounds(intensities, multiplier=float(params.get("multiplier", 1.5)))
        scores = intensities
        threshold = None
        candidates = [idx for idx, score in enumerate(scores) if score < lower or score > upper]
    elif method_key == "MAD":
        scores = _robust_zscores(_row_intensities(numeric_matrix(package)))
        threshold = float(params.get("threshold", 3.5))
        candidates = [idx for idx, score in enumerate(scores) if abs(score) > threshold]
    elif method_key in {"half_resampling_outlier", "mccv_outlier"}:
        output = _resampling_outlier_scores(package, method_key=method_key, params=params)
        scores = output["scores"]
        threshold = float(output["threshold"])
        candidates = [idx for idx, score in enumerate(scores) if score >= threshold and score > 0]
    else:
        raise QCMethodError("METHOD_NOT_IMPLEMENTED", f"{method_key} is registered but not implemented.", method=method_key)
    result = {
        "method_id": method_key,
        "display_name": method["display_name"],
        "parameters": params,
        "threshold": threshold,
        "outlier_sample_count": len(candidates),
        "outlier_sample_candidates": [
            {"sample_id": package.sample_ids[idx], "sample_index": idx, "score": scores[idx]} for idx in candidates
        ],
        "outlier_scores": [{"sample_id": sample_id, "sample_index": idx, "score": scores[idx]} for idx, sample_id in enumerate(package.sample_ids)],
        "confirmation_required_for_removal": bool(method["confirmation_required_for_removal"]),
    }
    if method_key in {"half_resampling_outlier", "mccv_outlier"}:
        result.update(output["metadata"])
        details_by_index = {int(item["sample_index"]): item for item in output["metadata"].get("resampling_details", [])}
        risk_samples = [details_by_index[idx] for idx in candidates if idx in details_by_index]
        result["resampling_risk_samples"] = risk_samples
        result["outlier_sample_candidates"] = risk_samples
        if output["metadata"].get("task_type") == "classification":
            result["classification_instability_candidates"] = risk_samples
        else:
            result["regression_prediction_instability_candidates"] = risk_samples
    return result


def summarize_resampling_outlier_control(outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    methods = [output for output in outputs if output.get("method_id") in {"half_resampling_outlier", "mccv_outlier"}]
    if not methods:
        return None
    high = sum(int(method.get("outlier_sample_count") or 0) for method in methods)
    medium = sum(int(method.get("medium_confidence_outlier_count") or 0) for method in methods)
    first = methods[0]
    params = first.get("parameters") or {}
    summary: dict[str, Any] = {
        "methods_run": [method["method_id"] for method in methods],
        "n_resamples": int(params.get("n_resamples") or first.get("n_resamples") or 0),
        "base_model": first.get("base_model"),
        "score_type": first.get("score_type"),
        "risk_semantics": first.get("risk_semantics"),
        "input_pipeline": first.get("input_pipeline"),
        "evaluation_summary": first.get("evaluation_summary"),
        "high_confidence_outliers": high,
        "medium_confidence_outliers": medium,
        "recommended_action": "mark_only",
        "cleaning_caution": first.get("cleaning_caution"),
    }
    if "train_ratio" in params or first.get("train_ratio") is not None:
        summary["train_ratio"] = float(params.get("train_ratio") or first.get("train_ratio"))
    if "sample_fraction" in params or first.get("sample_fraction") is not None:
        summary["sample_fraction"] = float(params.get("sample_fraction") or first.get("sample_fraction"))
    return summary


def _recommended_actions(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for output in outputs:
        if output["outlier_sample_count"]:
            actions.append(
                {
                    "action": "review_outlier_sample_candidates",
                    "method_id": output["method_id"],
                    "candidate_count": output["outlier_sample_count"],
                    "confirmation_required_for_removal": True,
                }
            )
    return actions


def _resampling_outlier_scores(package: SpectralQCPackage, *, method_key: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.svm import SVC, SVR
    except Exception as exc:
        raise QCMethodError("SKLEARN_REQUIRED", "HR/MCCV outlier control requires numpy and scikit-learn.", method=method_key) from exc

    if not package.y:
        raise QCMethodError("Y_REQUIRED_FOR_RESAMPLING_OUTLIER", "HR/MCCV outlier control requires y.csv.", method=method_key)

    task_type = _task_type(package)
    if task_type not in {"classification", "regression"}:
        raise QCMethodError("SUPERVISED_TASK_REQUIRED_FOR_RESAMPLING_OUTLIER", "HR/MCCV outlier control requires a classification or regression task.", method=method_key)

    n_resamples = max(1, int(params.get("n_resamples", 100)))
    train_fraction = float(params.get("sample_fraction" if method_key == "half_resampling_outlier" else "train_ratio", 0.5 if method_key == "half_resampling_outlier" else 0.7))
    train_fraction = min(0.9, max(0.1, train_fraction))
    base_model = str(params.get("base_model") or "auto").lower()
    if base_model == "auto":
        base_model = "svm" if task_type == "classification" else "pls"
    outlier_metric = str(params.get("outlier_metric") or "auto").lower()
    if outlier_metric == "auto":
        outlier_metric = "misclassification_frequency" if task_type == "classification" else "mean_absolute_residual"

    X = np.asarray(numeric_matrix(package), dtype=float)
    y_raw = [row[0] if row else "" for row in package.y]
    rng = np.random.default_rng(int(params.get("random_state", 17)))
    train_size = max(2, min(package.n_samples - 1, int(round(package.n_samples * train_fraction))))
    if task_type == "classification":
        labels = [str(value) for value in y_raw]
        classes = sorted(set(labels))
        if len(classes) < 2:
            raise QCMethodError("CLASSIFICATION_CLASSES_REQUIRED", "Classification resampling outlier control requires at least two classes.", method=method_key)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for _ in range(n_resamples):
            train_idx = _sample_train_indices(rng, package.n_samples, train_size, y=y)
            if len(set(y[train_idx])) < 2:
                continue
            validation_idx = [idx for idx in range(package.n_samples) if idx not in set(train_idx)]
            if not validation_idx:
                continue
            model = _classification_model(base_model, LogisticRegression, Pipeline, PLSRegression, SVC, StandardScaler, len(classes))
            try:
                model.fit(X[train_idx], y[train_idx])
                predicted = model.predict(X[validation_idx])
                probabilities = model.predict_proba(X[validation_idx]) if hasattr(model, "predict_proba") else None
            except Exception:
                continue
            for pos, sample_idx in enumerate(validation_idx):
                pred = int(round(float(predicted[pos])))
                pred = max(0, min(len(classes) - 1, pred))
                true = int(y[sample_idx])
                true_prob = None
                if probabilities is not None:
                    true_prob = float(probabilities[pos][true])
                observations[sample_idx].append({"error": int(pred != true), "predicted": pred, "true_probability": true_prob})
        scores, details = _classification_resampling_scores(package, labels, observations, outlier_metric)
    else:
        y = np.asarray([float(value) for value in y_raw], dtype=float)
        observations = defaultdict(list)
        for _ in range(n_resamples):
            train_idx = _sample_train_indices(rng, package.n_samples, train_size)
            validation_idx = [idx for idx in range(package.n_samples) if idx not in set(train_idx)]
            if not validation_idx:
                continue
            model = _regression_model(base_model, Pipeline, PLSRegression, Ridge, StandardScaler, SVR, train_size, X.shape[1])
            try:
                model.fit(X[train_idx], y[train_idx])
                predicted = model.predict(X[validation_idx])
            except Exception:
                continue
            predicted = np.asarray(predicted, dtype=float).reshape(-1)
            for pos, sample_idx in enumerate(validation_idx):
                residual = float(abs(y[sample_idx] - predicted[pos]))
                observations[sample_idx].append({"residual": residual, "predicted": float(predicted[pos])})
        scores, details = _regression_resampling_scores(package, observations, outlier_metric, y_std=float(np.std(y, ddof=1)) or 1.0)

    if not any(details[idx]["n_evaluations"] for idx in range(package.n_samples)):
        raise QCMethodError("RESAMPLING_OUTLIER_NO_EVALUATIONS", "HR/MCCV could not produce validation evaluations; use mark-only or a simpler QC method.", method=method_key)
    threshold = _threshold(scores, params.get("threshold", "percentile_95"))
    medium_threshold = _threshold(scores, "percentile_75")
    candidates = [idx for idx, score in enumerate(scores) if score >= threshold and score > 0]
    medium = [idx for idx, score in enumerate(scores) if idx not in set(candidates) and score >= medium_threshold and score > 0]
    evaluation_counts = [int(details[idx].get("n_evaluations") or 0) for idx in range(package.n_samples)]
    min_evaluations_for_stable_estimate = int(params.get("min_evaluations_for_stable_estimate", 20))
    for idx in range(package.n_samples):
        details[idx]["sample_id"] = package.sample_ids[idx]
        details[idx]["sample_index"] = idx
        details[idx]["score"] = scores[idx]
        details[idx]["risk_level"] = "high" if idx in candidates else "medium" if idx in medium else "low"
        details[idx]["unstable_estimate"] = int(details[idx].get("n_evaluations") or 0) < min_evaluations_for_stable_estimate
    risk_semantics = "classification_instability_risk" if task_type == "classification" else "prediction_residual_stability_risk"
    evaluation_summary = {
        "min_evaluations": min(evaluation_counts) if evaluation_counts else 0,
        "max_evaluations": max(evaluation_counts) if evaluation_counts else 0,
        "mean_evaluations": sum(evaluation_counts) / len(evaluation_counts) if evaluation_counts else 0.0,
        "min_evaluations_for_stable_estimate": min_evaluations_for_stable_estimate,
        "low_evaluation_sample_count": sum(1 for value in evaluation_counts if value < min_evaluations_for_stable_estimate),
        "low_evaluation_warning": any(value < min_evaluations_for_stable_estimate for value in evaluation_counts),
    }
    return {
        "scores": scores,
        "threshold": threshold,
        "metadata": {
            "n_resamples": n_resamples,
            "train_ratio": train_fraction if method_key == "mccv_outlier" else None,
            "sample_fraction": train_fraction if method_key == "half_resampling_outlier" else None,
            "base_model": base_model,
            "score_type": outlier_metric,
            "task_type": task_type,
            "risk_semantics": risk_semantics,
            "input_pipeline": {
                "package_dir": str(package.root),
                "preprocess": "current_package",
                "feature": "current_package",
                "base_model": base_model,
                "note": "Scores describe stability risk for the current input package and base model; they are not absolute evidence of spectral acquisition error.",
            },
            "evaluation_summary": evaluation_summary,
            "medium_confidence_outlier_count": len(medium),
            "recommended_action": "mark_only",
            "cleaning_caution": "Do not delete MCCV/HR-only candidates unless manual review confirms mislabeled or invalid samples.",
            "resampling_details": [details[idx] for idx in range(package.n_samples)],
        },
    }


def _classification_model(base_model: str, LogisticRegression: Any, Pipeline: Any, PLSRegression: Any, SVC: Any, StandardScaler: Any, n_classes: int) -> Any:
    if base_model in {"logistic", "logistic_regression"}:
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    if base_model == "pls":
        return _PLSClassifier(PLSRegression(n_components=max(1, min(n_classes - 1, 2))))
    return Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="linear", probability=True, class_weight="balanced"))])


def _regression_model(base_model: str, Pipeline: Any, PLSRegression: Any, Ridge: Any, StandardScaler: Any, SVR: Any, train_size: int, n_features: int) -> Any:
    if base_model == "svm":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))])
    if base_model == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    n_components = max(1, min(2, train_size - 1, n_features))
    return Pipeline([("scale", StandardScaler()), ("model", PLSRegression(n_components=n_components))])


class _PLSClassifier:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.classes_: list[int] = []

    def fit(self, X: Any, y: Any) -> "_PLSClassifier":
        self.classes_ = sorted(set(int(value) for value in y))
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        raw = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        return np.asarray([min(self.classes_, key=lambda label: abs(label - value)) for value in raw], dtype=int)


def _sample_train_indices(rng: Any, n_samples: int, train_size: int, *, y: Any | None = None) -> Any:
    import numpy as np

    indices = np.arange(n_samples)
    if y is None:
        return rng.choice(indices, size=train_size, replace=False)
    for _ in range(50):
        train_idx = rng.choice(indices, size=train_size, replace=False)
        if len(set(int(y[idx]) for idx in train_idx)) >= 2:
            return train_idx
    return rng.choice(indices, size=train_size, replace=False)


def _classification_resampling_scores(package: SpectralQCPackage, labels: list[str], observations: dict[int, list[dict[str, Any]]], metric: str) -> tuple[list[float], dict[int, dict[str, Any]]]:
    class_errors: dict[str, list[float]] = defaultdict(list)
    raw_errors: dict[int, float] = {}
    for idx in range(package.n_samples):
        items = observations.get(idx, [])
        error_frequency = sum(item["error"] for item in items) / len(items) if items else 0.0
        raw_errors[idx] = error_frequency
        class_errors[labels[idx]].append(error_frequency)
    class_mean_error = {label: (sum(values) / len(values) if values else 0.0) for label, values in class_errors.items()}
    scores: list[float] = []
    details: dict[int, dict[str, Any]] = {}
    for idx in range(package.n_samples):
        items = observations.get(idx, [])
        predictions = [int(item["predicted"]) for item in items]
        true_probs = [float(item["true_probability"]) for item in items if item.get("true_probability") is not None]
        instability = 0.0
        if predictions:
            most_common = Counter(predictions).most_common(1)[0][1]
            instability = 1.0 - most_common / len(predictions)
        mean_true_probability = sum(true_probs) / len(true_probs) if true_probs else None
        if metric == "mean_predicted_probability_of_true_class":
            score = 1.0 - mean_true_probability if mean_true_probability is not None else raw_errors[idx]
        elif metric == "prediction_instability":
            score = instability
        elif metric == "classwise_error_frequency":
            score = max(0.0, raw_errors[idx] - class_mean_error.get(labels[idx], 0.0))
        else:
            score = raw_errors[idx]
        scores.append(float(score))
        details[idx] = {
            "class": labels[idx],
            "n_evaluations": len(items),
            "misclassification_frequency": raw_errors[idx],
            "mean_predicted_probability_of_true_class": mean_true_probability,
            "prediction_instability": instability,
            "classwise_error_frequency": max(0.0, raw_errors[idx] - class_mean_error.get(labels[idx], 0.0)),
        }
    return scores, details


def _regression_resampling_scores(package: SpectralQCPackage, observations: dict[int, list[dict[str, Any]]], metric: str, *, y_std: float) -> tuple[list[float], dict[int, dict[str, Any]]]:
    scores: list[float] = []
    details: dict[int, dict[str, Any]] = {}
    for idx in range(package.n_samples):
        items = observations.get(idx, [])
        residuals = [float(item["residual"]) for item in items]
        mean_abs = sum(residuals) / len(residuals) if residuals else 0.0
        residual_std = _std(residuals)
        standardized = mean_abs / y_std
        score = residual_std if metric == "residual_std" else standardized if metric == "standardized_residual" else mean_abs
        scores.append(float(score))
        details[idx] = {
            "n_evaluations": len(items),
            "mean_absolute_residual": mean_abs,
            "standardized_residual": standardized,
            "residual_std": residual_std,
            "high_leverage_frequency": None,
        }
    return scores, details


def _mahalanobis_scores(matrix: list[list[float]], *, ridge: float) -> list[float]:
    try:
        import numpy as np
    except Exception as exc:
        raise QCMethodError("NUMPY_REQUIRED", "MD requires numpy.", method="MD") from exc
    arr = np.asarray(matrix, dtype=float)
    centered = arr - arr.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * ridge
    inv = np.linalg.pinv(cov)
    distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inv, centered), 0.0))
    return [float(value) for value in distances]


def _pca_distance_scores(matrix: list[list[float]], components: Any) -> list[float]:
    try:
        import numpy as np
    except Exception as exc:
        raise QCMethodError("NUMPY_REQUIRED", "PCA_DISTANCE requires numpy.", method="PCA_DISTANCE") from exc
    arr = np.asarray(matrix, dtype=float)
    centered = arr - arr.mean(axis=0)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if components == "auto":
        k = min(5, max(1, arr.shape[0] - 1), arr.shape[1])
    else:
        k = max(1, min(int(components), arr.shape[0] - 1, arr.shape[1]))
    loadings = vt[:k].T
    scores = centered @ loadings
    scale = np.std(scores, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    distances = np.sqrt(np.sum((scores / scale) ** 2, axis=1))
    return [float(value) for value in distances]


def _row_intensities(matrix: list[list[float]]) -> list[float]:
    return [sum(row) / len(row) if row else 0.0 for row in matrix]


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


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _robust_zscores(values: list[float]) -> list[float]:
    med = _median(values)
    mad = _median([abs(value - med) for value in values])
    scale = 1.4826 * mad
    if scale == 0:
        avg = sum(values) / len(values) if values else 0.0
        scale = (sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)) ** 0.5 or 1e-12
    return [(value - med) / scale for value in values]


def _threshold(scores: list[float], setting: Any) -> float:
    if isinstance(setting, (int, float)):
        return float(setting)
    text = str(setting)
    if text.startswith("quantile_"):
        return _quantile(scores, float(text.split("_", 1)[1]))
    if text.startswith("percentile_"):
        return _quantile(scores, float(text.split("_", 1)[1]) / 100.0)
    return float(text)


def _iqr_bounds(values: list[float], *, multiplier: float) -> tuple[float, float]:
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


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


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5

"""First-stage spectral classification and regression models."""

from __future__ import annotations

import itertools
import json
import math
import importlib
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from .estimators import PLSDAClassifier, SIMCAClassifier
from .io import ModelingPackage, SplitInfo
from .registry import CLASSIFICATION_MODELS, REGRESSION_MODELS, model_spec, normalize_model_name


class ModelingMethodError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


SUPPORTED_TASKS = {"classification", "regression"}

CLASSIFICATION_MODEL_SETS = {
    "compact": ["svm", "linear_svm", "pls_da"],
    "regular_fast": [
        "logistic_regression",
        "linear_svm",
        "svm",
        "lda",
        "knn_classifier",
        "random_forest_classifier",
        "extra_trees_classifier",
    ],
    "regular_full": [
        "logistic_regression",
        "linear_svm",
        "svm",
        "lda",
        "knn_classifier",
        "random_forest_classifier",
        "extra_trees_classifier",
        "gradient_boosting_classifier",
    ],
    "spectral_modeling": [
        "svm",
        "linear_svm",
        "pls_da",
        "lda",
        "qda",
        "knn_classifier",
        "random_forest_classifier",
    ],
}
CLASSIFICATION_MODEL_SETS["regular"] = CLASSIFICATION_MODEL_SETS["regular_full"]


def infer_task_type(package: ModelingPackage, task_type: str | None) -> str:
    if task_type:
        normalized = _normalize_task(task_type)
        if normalized == "multi_target_regression":
            raise ModelingMethodError("MULTI_TARGET_NOT_IMPLEMENTED", "This first modeling version supports single-target classification or regression only.")
        if normalized not in SUPPORTED_TASKS:
            raise ModelingMethodError("TASK_TYPE_UNSUPPORTED", "Unsupported modeling task type.", task_type=task_type)
        return normalized

    hint = str(package.contract.get("task_hint") or "").lower()
    if "class" in hint:
        return "classification"
    if "regression" in hint or "quant" in hint:
        return "regression"
    if not _all_float(package.y_values):
        return "classification"
    raise ModelingMethodError("TASK_TYPE_REQUIRED", "Please confirm task type: classification, regression, or multi_target_regression.")


def parse_models(models: str | list[str] | None, task_type: str) -> list[str]:
    if models is None:
        raise ModelingMethodError(
            "MODEL_TYPE_REQUIRED",
            "Please choose models. Classification: logistic_regression, linear_svm, svm, lda, random_forest_classifier, knn_classifier. Regression: plsr, ridge, svr, random_forest_regressor.",
            task_type=task_type,
        )
    raw = [item.strip() for item in models.split(",") if item.strip()] if isinstance(models, str) else [str(item).strip() for item in models if str(item).strip()]
    if not raw:
        raise ModelingMethodError("MODEL_TYPE_REQUIRED", "Please choose at least one model.", task_type=task_type)
    normalized: list[str] = []
    for item in raw:
        alias = _normalize_model_set_alias(item)
        if task_type == "classification" and alias in CLASSIFICATION_MODEL_SETS:
            normalized.extend(CLASSIFICATION_MODEL_SETS[alias])
        else:
            normalized.append(_normalize_model(item))
    supported = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    for model in normalized:
        if model not in supported:
            raise ModelingMethodError("MODEL_UNSUPPORTED_FOR_TASK", "Model is unsupported for the selected task.", model=model, task_type=task_type)
    return _dedupe_preserve_order(normalized)


def train_and_evaluate(
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    task_type: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    parameter_sources: dict[str, dict[str, str]] | None,
    cv_folds: int,
    random_seed: int,
    evaluation_mode: str = "final",
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    if split_info.split_type in {"cross_validation", "repeated_holdout"}:
        return _train_and_evaluate_partitions(
            package,
            split_info,
            task_type=task_type,
            models=models,
            model_parameters=model_parameters,
            parameter_sources=parameter_sources,
            cv_folds=cv_folds,
            random_seed=random_seed,
            param_search_enabled=param_search_enabled,
            model_params_source=model_params_source,
        )

    assignments = split_info.assignments
    train_idx = assignments.get("train", [])
    val_idx = assignments.get("val", [])
    test_idx = assignments.get("test", [])
    if evaluation_mode == "final" and not test_idx:
        raise ModelingMethodError("TEST_SPLIT_REQUIRED", "A test split is required for final independent model evaluation.")
    if evaluation_mode == "validation_only" and not val_idx:
        raise ModelingMethodError("VALIDATION_SPLIT_REQUIRED", "validation_only evaluation requires a non-empty validation split.")
    if not train_idx:
        raise ModelingMethodError("TRAIN_SPLIT_EMPTY", "Cannot train models without train samples.")
    evaluation_test_idx = test_idx if evaluation_mode == "final" else []

    X = np.asarray(package.X, dtype=float)
    y_raw = np.asarray(package.y_values, dtype=object)
    if task_type == "classification":
        result = _train_classification(
            X, y_raw, package, train_idx, val_idx, evaluation_test_idx,
            models=models, model_parameters=model_parameters, parameter_sources=parameter_sources,
            cv_folds=cv_folds, random_seed=random_seed,
            param_search_enabled=param_search_enabled,
            model_params_source=model_params_source,
        )
    else:
        result = _train_regression(
            X, y_raw, package, train_idx, val_idx, evaluation_test_idx,
            models=models, model_parameters=model_parameters, parameter_sources=parameter_sources,
            cv_folds=cv_folds, random_seed=random_seed,
            param_search_enabled=param_search_enabled,
            model_params_source=model_params_source,
        )
    result["evaluation_mode"] = evaluation_mode
    result["test_accessed"] = evaluation_mode == "final"
    return result


def train_and_evaluate_iteration_packages(
    packages: dict[str, ModelingPackage],
    split_info: SplitInfo,
    *,
    task_type: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    parameter_sources: dict[str, dict[str, str]] | None,
    cv_folds: int,
    random_seed: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    eval_role = "val" if split_info.split_type == "cross_validation" else "test"
    iteration_results = []
    all_predictions = []
    first_package = next(iter(packages.values()))
    y_raw = np.asarray(first_package.y_values, dtype=object)
    if task_type == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        if len(encoder.classes_) < 2:
            raise ModelingMethodError("CLASSIFICATION_NEEDS_TWO_CLASSES", "Classification requires at least two classes.")
        for partition in split_info.partitions or []:
            package = packages.get(partition.iteration_id)
            if package is None:
                raise ModelingMethodError("ITERATION_FEATURES_MISSING", "No iteration feature/preprocess matrix was loaded for this partition.", iteration_id=partition.iteration_id)
            item = _train_classification_iteration(
                np.asarray(package.X, dtype=float),
                y,
                y_raw.astype(str),
                package,
                partition,
                eval_role=eval_role,
                models=models,
                model_parameters=model_parameters,
                cv_folds=cv_folds,
                random_seed=random_seed,
                encoder=encoder,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
            )
            iteration_results.append(item)
            all_predictions.extend(item["predictions"])
        metric_summary = _metric_summary([item["eval_metrics"] for item in iteration_results])
        return _iterative_result_payload("classification", split_info, models, eval_role, iteration_results, all_predictions, metric_summary, encoder.classes_.tolist(), parameter_sources)

    if not _all_float(y_raw.tolist()):
        raise ModelingMethodError("REGRESSION_Y_NON_NUMERIC", "Regression requires numeric target values.")
    y = y_raw.astype(float)
    for partition in split_info.partitions or []:
        package = packages.get(partition.iteration_id)
        if package is None:
            raise ModelingMethodError("ITERATION_FEATURES_MISSING", "No iteration feature/preprocess matrix was loaded for this partition.", iteration_id=partition.iteration_id)
        item = _train_regression_iteration(
            np.asarray(package.X, dtype=float),
            y,
            package,
            partition,
            eval_role=eval_role,
            models=models,
            model_parameters=model_parameters,
            cv_folds=cv_folds,
            random_seed=random_seed,
            param_search_enabled=param_search_enabled,
            model_params_source=model_params_source,
        )
        iteration_results.append(item)
        all_predictions.extend(item["predictions"])
    metric_summary = _metric_summary([item["eval_metrics"] for item in iteration_results])
    return _iterative_result_payload("regression", split_info, models, eval_role, iteration_results, all_predictions, metric_summary, [], parameter_sources)


def _train_and_evaluate_partitions(
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    task_type: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    parameter_sources: dict[str, dict[str, str]] | None,
    cv_folds: int,
    random_seed: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    X = np.asarray(package.X, dtype=float)
    y_raw = np.asarray(package.y_values, dtype=object)
    eval_role = "val" if split_info.split_type == "cross_validation" else "test"
    iteration_results = []
    all_predictions = []
    if task_type == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        if len(encoder.classes_) < 2:
            raise ModelingMethodError("CLASSIFICATION_NEEDS_TWO_CLASSES", "Classification requires at least two classes.")
        for partition in split_info.partitions or []:
            item = _train_classification_iteration(
                X,
                y,
                y_raw.astype(str),
                package,
                partition,
                eval_role=eval_role,
                models=models,
                model_parameters=model_parameters,
                cv_folds=cv_folds,
                random_seed=random_seed,
                encoder=encoder,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
            )
            iteration_results.append(item)
            all_predictions.extend(item["predictions"])
        metric_summary = _metric_summary([item["eval_metrics"] for item in iteration_results])
        return _iterative_result_payload("classification", split_info, models, eval_role, iteration_results, all_predictions, metric_summary, encoder.classes_.tolist(), parameter_sources)

    if not _all_float(y_raw.tolist()):
        raise ModelingMethodError("REGRESSION_Y_NON_NUMERIC", "Regression requires numeric target values.")
    y = y_raw.astype(float)
    for partition in split_info.partitions or []:
        item = _train_regression_iteration(
            X,
            y,
            package,
            partition,
            eval_role=eval_role,
            models=models,
            model_parameters=model_parameters,
            cv_folds=cv_folds,
            random_seed=random_seed,
            param_search_enabled=param_search_enabled,
            model_params_source=model_params_source,
        )
        iteration_results.append(item)
        all_predictions.extend(item["predictions"])
    metric_summary = _metric_summary([item["eval_metrics"] for item in iteration_results])
    return _iterative_result_payload("regression", split_info, models, eval_role, iteration_results, all_predictions, metric_summary, [], parameter_sources)


def _iterative_result_payload(
    task_type: str,
    split_info: SplitInfo,
    models: list[str],
    eval_role: str,
    iteration_results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metric_summary: dict[str, Any],
    class_labels: list[str],
    parameter_sources: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "eval_role": eval_role,
        "model": None,
        "model_type": ",".join(models),
        "model_parameters": {},
        "model_families": {model: model_spec(model).family for model in models},
        "parameter_sources": parameter_sources or {},
        "selection": {
            "tuning_split": "outer_train_inner_cv" if eval_role == "val" else "partition_val_or_train_cv",
            "selection_metric": "macro_f1" if task_type == "classification" else "rmse",
            "candidate_count": len(models),
            "candidate_models": models,
            "test_used_for_selection": False,
            "hyperparameter_selection": _hyperparameter_selection_summary(eval_role=eval_role, cv_folds=None),
        },
        "metrics": {"summary": metric_summary},
        "metric_summary": metric_summary,
        "iteration_results": iteration_results,
        "predictions": predictions,
        "class_labels": class_labels,
    }


def _train_classification_iteration(
    X: np.ndarray,
    y: np.ndarray,
    y_raw: np.ndarray,
    package: ModelingPackage,
    partition: Any,
    *,
    eval_role: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    cv_folds: int,
    random_seed: int,
    encoder: LabelEncoder,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    train_idx = partition.train_indices
    val_idx = partition.val_indices
    test_idx = partition.test_indices
    eval_idx = val_idx if eval_role == "val" else test_idx
    if not eval_idx:
        raise ModelingMethodError("EVAL_SPLIT_EMPTY", "Each CV/repeated partition must have a non-empty evaluation split.", iteration_id=partition.iteration_id)
    candidates = []
    selection_idx = [] if eval_role == "val" else (val_idx if val_idx else [])
    for model_name in models:
        estimator, grid = _classification_estimator(
            model_name,
            random_seed=random_seed,
            n_train=len(train_idx),
            n_features=X.shape[1],
            parameters=(model_parameters or {}).get(model_name),
        )
        candidates.extend(
            _select_candidates(
                estimator,
                grid,
                model_name,
                X,
                y,
                train_idx,
                selection_idx,
                task_type="classification",
                cv_folds=cv_folds,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
                requested_parameters=(model_parameters or {}).get(model_name),
            )
        )
    best = max(candidates, key=lambda item: item["selection_score"])
    final_model = clone(best["estimator"])
    final_model.fit(X[train_idx], y[train_idx])
    predictions = _classification_predictions(final_model, X, y, y_raw, package.sample_ids, train_idx, val_idx, test_idx, encoder)
    for row in predictions:
        row["iteration_id"] = partition.iteration_id
        row["iteration_type"] = partition.iteration_type
    eval_metrics = _classification_metrics(y[eval_idx], final_model.predict(X[eval_idx]), final_model, X[eval_idx], encoder)
    return {
        "iteration_id": partition.iteration_id,
        "iteration_type": partition.iteration_type,
        "model": final_model,
        "model_type": best["model_type"],
        "model_parameters": best["parameters"],
        "model_family": model_spec(best["model_type"]).family,
        "selection": _selection_summary(best, candidates, used_val=bool(selection_idx), eval_role=eval_role),
        "train_metrics": _classification_metrics(y[train_idx], final_model.predict(X[train_idx]), final_model, X[train_idx], encoder),
        "eval_role": eval_role,
        "eval_metrics": eval_metrics,
        "predictions": predictions,
    }


def _train_regression_iteration(
    X: np.ndarray,
    y: np.ndarray,
    package: ModelingPackage,
    partition: Any,
    *,
    eval_role: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    cv_folds: int,
    random_seed: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    train_idx = partition.train_indices
    val_idx = partition.val_indices
    test_idx = partition.test_indices
    eval_idx = val_idx if eval_role == "val" else test_idx
    if not eval_idx:
        raise ModelingMethodError("EVAL_SPLIT_EMPTY", "Each CV/repeated partition must have a non-empty evaluation split.", iteration_id=partition.iteration_id)
    candidates = []
    selection_idx = [] if eval_role == "val" else (val_idx if val_idx else [])
    for model_name in models:
        estimator, grid = _regression_estimator(
            model_name,
            random_seed=random_seed,
            n_features=X.shape[1],
            n_train=len(train_idx),
            parameters=(model_parameters or {}).get(model_name),
        )
        candidates.extend(
            _select_candidates(
                estimator,
                grid,
                model_name,
                X,
                y,
                train_idx,
                selection_idx,
                task_type="regression",
                cv_folds=cv_folds,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
                requested_parameters=(model_parameters or {}).get(model_name),
            )
        )
    best = min(candidates, key=lambda item: item["selection_score"])
    final_model = clone(best["estimator"])
    final_model.fit(X[train_idx], y[train_idx])
    predictions = _regression_predictions(final_model, X, y, package.sample_ids, train_idx, val_idx, test_idx)
    for row in predictions:
        row["iteration_id"] = partition.iteration_id
        row["iteration_type"] = partition.iteration_type
    eval_metrics = _regression_metrics(y[eval_idx], final_model.predict(X[eval_idx]))
    return {
        "iteration_id": partition.iteration_id,
        "iteration_type": partition.iteration_type,
        "model": final_model,
        "model_type": best["model_type"],
        "model_parameters": best["parameters"],
        "model_family": model_spec(best["model_type"]).family,
        "selection": _selection_summary(best, candidates, used_val=bool(selection_idx), eval_role=eval_role),
        "train_metrics": _regression_metrics(y[train_idx], final_model.predict(X[train_idx])),
        "eval_role": eval_role,
        "eval_metrics": eval_metrics,
        "predictions": predictions,
    }


def _train_classification(
    X: np.ndarray,
    y_raw: np.ndarray,
    package: ModelingPackage,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    *,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    parameter_sources: dict[str, dict[str, str]] | None,
    cv_folds: int,
    random_seed: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    if len(encoder.classes_) < 2:
        raise ModelingMethodError("CLASSIFICATION_NEEDS_TWO_CLASSES", "Classification requires at least two classes.")

    candidates = []
    for model_name in models:
        estimator, grid = _classification_estimator(
            model_name,
            random_seed=random_seed,
            n_train=len(train_idx),
            n_features=X.shape[1],
            parameters=(model_parameters or {}).get(model_name),
        )
        candidates.extend(
            _select_candidates(
                estimator,
                grid,
                model_name,
                X,
                y,
                train_idx,
                val_idx,
                task_type="classification",
                cv_folds=cv_folds,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
                requested_parameters=(model_parameters or {}).get(model_name),
            )
        )
    best = max(candidates, key=lambda item: item["selection_score"])
    classifier_validation_summary = _classification_validation_summary_rows(
        candidates,
        X,
        y,
        train_idx,
        val_idx,
        test_idx,
        encoder,
        selected_model=best["model_type"],
        selected_parameters=best["parameters"],
    )
    final_model = clone(best["estimator"])
    final_model.fit(X[train_idx], y[train_idx])
    split_predictions = _classification_predictions(final_model, X, y, y_raw.astype(str), package.sample_ids, train_idx, val_idx, test_idx, encoder)
    metrics = {
        "train": _classification_metrics(y[train_idx], final_model.predict(X[train_idx]), final_model, X[train_idx], encoder),
        "val": _classification_metrics(y[val_idx], final_model.predict(X[val_idx]), final_model, X[val_idx], encoder) if val_idx else {},
        "test": _classification_metrics(y[test_idx], final_model.predict(X[test_idx]), final_model, X[test_idx], encoder) if test_idx else {},
    }
    return {
        "task_type": "classification",
        "model": final_model,
        "model_type": best["model_type"],
        "model_parameters": best["parameters"],
        "model_family": model_spec(best["model_type"]).family,
        "parameter_sources": (parameter_sources or {}).get(best["model_type"], {}),
        "selection": _selection_summary(best, candidates, used_val=bool(val_idx), eval_role="test"),
        "metrics": metrics,
        "classifier_validation_summary": classifier_validation_summary,
        "predictions": split_predictions,
        "class_labels": encoder.classes_.tolist(),
    }


def _train_regression(
    X: np.ndarray,
    y_raw: np.ndarray,
    package: ModelingPackage,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    *,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]] | None,
    parameter_sources: dict[str, dict[str, str]] | None,
    cv_folds: int,
    random_seed: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
) -> dict[str, Any]:
    if not _all_float(y_raw.tolist()):
        raise ModelingMethodError("REGRESSION_Y_NON_NUMERIC", "Regression requires numeric target values.")
    y = y_raw.astype(float)
    candidates = []
    for model_name in models:
        estimator, grid = _regression_estimator(
            model_name,
            random_seed=random_seed,
            n_features=X.shape[1],
            n_train=len(train_idx),
            parameters=(model_parameters or {}).get(model_name),
        )
        candidates.extend(
            _select_candidates(
                estimator,
                grid,
                model_name,
                X,
                y,
                train_idx,
                val_idx,
                task_type="regression",
                cv_folds=cv_folds,
                param_search_enabled=param_search_enabled,
                model_params_source=model_params_source,
                requested_parameters=(model_parameters or {}).get(model_name),
            )
        )
    best = min(candidates, key=lambda item: item["selection_score"])
    final_model = clone(best["estimator"])
    final_model.fit(X[train_idx], y[train_idx])
    split_predictions = _regression_predictions(final_model, X, y, package.sample_ids, train_idx, val_idx, test_idx)
    metrics = {
        "train": _regression_metrics(y[train_idx], final_model.predict(X[train_idx])),
        "val": _regression_metrics(y[val_idx], final_model.predict(X[val_idx])) if val_idx else {},
        "test": _regression_metrics(y[test_idx], final_model.predict(X[test_idx])) if test_idx else {},
    }
    return {
        "task_type": "regression",
        "model": final_model,
        "model_type": best["model_type"],
        "model_parameters": best["parameters"],
        "model_family": model_spec(best["model_type"]).family,
        "parameter_sources": (parameter_sources or {}).get(best["model_type"], {}),
        "selection": _selection_summary(best, candidates, used_val=bool(val_idx), eval_role="test"),
        "metrics": metrics,
        "predictions": split_predictions,
        "class_labels": [],
    }


def _select_candidates(
    estimator: Any,
    grid: dict[str, list[Any]],
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    *,
    task_type: str,
    cv_folds: int,
    param_search_enabled: bool = True,
    model_params_source: str = "modeling_internal_selection",
    requested_parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not param_search_enabled:
        locked_params = _locked_parameter_grid(model_name, grid, requested_parameters)
        candidate = clone(estimator).set_params(**locked_params)
        candidate.fit(X[train_idx], y[train_idx])
        eval_idx = val_idx if val_idx else train_idx
        pred = candidate.predict(X[eval_idx])
        score = float(f1_score(y[eval_idx], pred, average="macro", zero_division=0)) if task_type == "classification" else float(math.sqrt(mean_squared_error(y[eval_idx], pred)))
        return [
            {
                "model_type": model_name,
                "parameters": locked_params,
                "estimator": candidate,
                "selection_score": score,
                "selection_metric": "locked_val_macro_f1" if task_type == "classification" and val_idx else "locked_train_macro_f1" if task_type == "classification" else "locked_val_rmse" if val_idx else "locked_train_rmse",
                "selection_split": "locked_val" if val_idx else "locked_train",
                "selection_mode": "locked_from_optimizer_best" if model_params_source == "best_pipeline.json" else "fixed_parameters",
                "param_search_enabled": False,
                "model_params_source": model_params_source,
            }
        ]
    if val_idx:
        output = []
        for params in _param_product(grid):
            candidate = clone(estimator).set_params(**params)
            try:
                candidate.fit(X[train_idx], y[train_idx])
                pred = candidate.predict(X[val_idx])
            except Exception:
                continue
            score = float(f1_score(y[val_idx], pred, average="macro", zero_division=0)) if task_type == "classification" else float(math.sqrt(mean_squared_error(y[val_idx], pred)))
            output.append({"model_type": model_name, "parameters": params, "estimator": candidate, "selection_score": score, "selection_metric": "macro_f1" if task_type == "classification" else "rmse", "selection_split": "val", "selection_mode": "modeling_internal_selection", "param_search_enabled": True, "model_params_source": model_params_source})
        if not output:
            raise ModelingMethodError("MODEL_CANDIDATES_FAILED", "All candidate parameter settings failed during validation selection.", model=model_name)
        return output

    folds = _valid_cv_folds(y[train_idx], requested=cv_folds, task_type=task_type)
    search = GridSearchCV(
        clone(estimator),
        grid,
        cv=folds,
        scoring="f1_macro" if task_type == "classification" else "neg_root_mean_squared_error",
        error_score="raise",
    )
    search.fit(X[train_idx], y[train_idx])
    score = float(search.best_score_) if task_type == "classification" else float(-search.best_score_)
    return [{"model_type": model_name, "parameters": search.best_params_, "estimator": search.best_estimator_, "selection_score": score, "selection_metric": "cv_macro_f1" if task_type == "classification" else "cv_rmse", "selection_split": f"train_cv_{folds}fold", "selection_mode": "modeling_internal_selection", "param_search_enabled": True, "model_params_source": model_params_source}]


def _classification_validation_summary_rows(
    candidates: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    encoder: LabelEncoder,
    *,
    selected_model: str,
    selected_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name in _dedupe_preserve_order([str(item["model_type"]) for item in candidates]):
        model_candidates = [item for item in candidates if item["model_type"] == model_name]
        if not model_candidates:
            continue
        best = max(model_candidates, key=lambda item: item["selection_score"])
        estimator = best["estimator"]
        row: dict[str, Any] = {
            "model_method": model_name,
            "model_type": model_name,
            "model_parameters": dict(best.get("parameters") or {}),
            "model_parameters_json": json_safe_dumps(best.get("parameters") or {}),
            "selection_metric": best.get("selection_metric"),
            "selection_score": best.get("selection_score"),
            "selection_split": best.get("selection_split"),
            "model_selection_mode": best.get("selection_mode"),
            "param_search_enabled": bool(best.get("param_search_enabled", True)),
            "model_params_source": best.get("model_params_source"),
            "selected_model": model_name == selected_model and dict(best.get("parameters") or {}) == dict(selected_parameters or {}),
            "test_accessed": bool(test_idx),
        }
        for prefix, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if not indices:
                continue
            metrics = _classification_metrics(y[indices], estimator.predict(X[indices]), estimator, X[indices], encoder)
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        output.append(row)
    output.sort(key=lambda row: float(row.get("val_macro_f1", row.get("selection_score", -1.0)) or -1.0), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank_val_macro_f1"] = rank
    return output


def json_safe_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _locked_parameter_grid(model_name: str, grid: dict[str, list[Any]], parameters: dict[str, Any] | None) -> dict[str, Any]:
    parameters = dict(parameters or {})
    if not grid:
        return parameters
    if not parameters and len(_param_product(grid)) == 1:
        return _param_product(grid)[0]
    if not parameters:
        missing_key = next((key for key, values in grid.items() if len(values) > 1), next(iter(grid), None))
        raise ModelingMethodError(
            "LOCKED_MODEL_PARAMETER_MISSING",
            "Final evaluation cannot exactly reproduce optimizer-selected model params because no model parameters were supplied.",
            model=model_name,
            missing_parameter=missing_key,
            provided_parameters=parameters,
        )
    resolved: dict[str, Any] = {}
    missing = object()
    for grid_key, values in grid.items():
        aliases = _parameter_aliases(model_name, grid_key)
        value = missing
        for alias in aliases:
            if alias in parameters:
                value = parameters[alias]
                break
        if value is missing:
            if len(values) == 1:
                value = values[0]
            else:
                continue
        resolved[grid_key] = _coerce_to_grid_value(value, values)
    extra = sorted(set(parameters) - {alias for key in grid for alias in _parameter_aliases(model_name, key)})
    if extra:
        raise ModelingMethodError(
            "LOCKED_MODEL_PARAMETER_UNKNOWN",
            "Final evaluation cannot exactly reproduce optimizer-selected model params because unknown parameters were supplied.",
            model=model_name,
            unknown_parameters=extra,
            supported_parameters=sorted(grid),
        )
    return resolved


def _parameter_aliases(model_name: str, grid_key: str) -> list[str]:
    aliases = [grid_key]
    bare = grid_key.split("__")[-1]
    aliases.append(bare)
    aliases.append(f"{model_name}__{bare}")
    return list(dict.fromkeys(aliases))


def _coerce_to_grid_value(value: Any, allowed_values: list[Any]) -> Any:
    for allowed in allowed_values:
        if value == allowed or str(value) == str(allowed):
            return allowed
        try:
            if isinstance(allowed, float) and float(value) == allowed:
                return allowed
            if isinstance(allowed, int) and int(value) == allowed:
                return allowed
        except (TypeError, ValueError):
            pass
    return value


def _classification_estimator(
    model_name: str,
    *,
    random_seed: int,
    n_train: int,
    n_features: int,
    parameters: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, list[Any]]]:
    parameters = dict(parameters or {})
    if model_name == "logistic_regression":
        estimator = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=random_seed))])
        return estimator, {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"], "model__class_weight": [None, "balanced"]}
    if model_name == "svm":
        estimator = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=random_seed))])
        return estimator, {"model__C": [0.1, 1.0, 10.0, 100.0], "model__gamma": ["scale", 0.01, 0.1], "model__class_weight": [None, "balanced"]}
    if model_name == "linear_svm":
        estimator = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="linear", probability=True, random_state=random_seed))])
        return estimator, {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
    if model_name == "lda":
        estimator = Pipeline([("scale", StandardScaler()), ("model", LinearDiscriminantAnalysis(solver="lsqr"))])
        return estimator, {"model__shrinkage": [None, "auto", 0.1]}
    if model_name == "qda":
        estimator = Pipeline([("scale", StandardScaler()), ("model", QuadraticDiscriminantAnalysis())])
        return estimator, {"model__reg_param": [0.0, 0.1, 0.5, 0.9]}
    if model_name == "gaussian_nb":
        return GaussianNB(), {"var_smoothing": [1e-11, 1e-9, 1e-7]}
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(random_state=random_seed), {"n_estimators": [100, 300], "max_depth": [None, 5], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1]}
    if model_name == "extra_trees_classifier":
        return ExtraTreesClassifier(random_state=random_seed), {"n_estimators": [100, 300], "max_depth": [None, 5], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1]}
    if model_name == "gradient_boosting_classifier":
        return GradientBoostingClassifier(random_state=random_seed), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    if model_name == "knn_classifier":
        estimator = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])
        neighbors = [value for value in [1, 3, 5, 7, 9] if value <= n_train]
        return estimator, {"model__n_neighbors": neighbors or [1], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean"]}
    if model_name == "pls_da":
        max_components = max(1, min(10, n_features, n_train - 1))
        values = _component_grid(max_components, parameters.get("n_components"))
        return PLSDAClassifier(), {"n_components": values}
    if model_name == "simca":
        max_components = max(1, min(10, n_features, max(1, n_train - 1)))
        values = _component_grid(max_components, parameters.get("n_components"))
        return SIMCAClassifier(), {"n_components": values, "quantile": [float(parameters.get("quantile", 0.95))]}
    if model_name == "mlp_classifier":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", MLPClassifier(max_iter=int(parameters.get("max_iter", 500)), early_stopping=True, random_state=random_seed)),
            ]
        )
        return estimator, {"model__hidden_layer_sizes": [(32,), (64,)], "model__alpha": [0.0001, 0.001]}
    if model_name in {"xgboost_classifier", "lightgbm_classifier", "catboost_classifier"}:
        return _optional_classification_estimator(model_name, random_seed=random_seed, parameters=parameters)
    if model_name in {
        "spectral_dkl_gp_classifier",
        "proto_spectral_classifier",
        "cls_former_classifier",
        "cls_former_embedding_svm",
    }:
        return _experimental_estimator(model_name, "classification", parameters), {}
    raise ModelingMethodError("MODEL_UNSUPPORTED_FOR_TASK", "Unsupported classification model.", model=model_name)


def _regression_estimator(
    model_name: str,
    *,
    random_seed: int,
    n_features: int,
    n_train: int,
    parameters: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, list[Any]]]:
    parameters = dict(parameters or {})
    if model_name == "plsr":
        max_components = max(1, min(10, n_features, n_train - 1))
        return PLSRegression(), {"n_components": _component_grid(max_components, parameters.get("n_components"))}
    if model_name == "pcr":
        max_components = max(1, min(20, n_features, n_train - 1))
        estimator = Pipeline([("scale", StandardScaler()), ("pca", PCA()), ("model", LinearRegression())])
        return estimator, {"pca__n_components": _component_grid(max_components, parameters.get("n_components"))}
    if model_name == "linear_regression":
        return LinearRegression(), {"fit_intercept": [True]}
    if model_name == "ridge":
        estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
        return estimator, {"model__alpha": [0.1, 1.0, 10.0]}
    if model_name == "lasso":
        estimator = Pipeline([("scale", StandardScaler()), ("model", Lasso(max_iter=5000, random_state=random_seed))])
        return estimator, {"model__alpha": [0.001, 0.01, 0.1]}
    if model_name == "elastic_net":
        estimator = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(max_iter=5000, random_state=random_seed))])
        return estimator, {"model__alpha": [0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]}
    if model_name == "bayesian_ridge":
        estimator = Pipeline([("scale", StandardScaler()), ("model", BayesianRidge())])
        return estimator, {"model__alpha_1": [1e-6], "model__lambda_1": [1e-6]}
    if model_name == "svr":
        estimator = Pipeline([("scale", StandardScaler()), ("model", SVR())])
        return estimator, {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]}
    if model_name == "knn_regressor":
        estimator = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())])
        neighbors = [value for value in [1, 3, 5] if value <= n_train]
        return estimator, {"model__n_neighbors": neighbors or [1], "model__weights": ["uniform", "distance"]}
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(random_state=random_seed), {"n_estimators": [50, 100], "max_depth": [None, 5]}
    if model_name == "extra_trees_regressor":
        return ExtraTreesRegressor(random_state=random_seed), {"n_estimators": [100, 200], "max_depth": [None, 5]}
    if model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(random_state=random_seed), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    if model_name == "gpr":
        kernel_name = str(parameters.get("kernel", "rbf")).lower()
        base_kernel = Matern(length_scale=1.0, nu=1.5) if kernel_name == "matern" else RBF(length_scale=1.0)
        kernel = ConstantKernel(1.0) * base_kernel + WhiteKernel(noise_level=1e-3)
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_seed, n_restarts_optimizer=0)),
            ]
        )
        return estimator, {"model__alpha": [1e-8, 1e-6]}
    if model_name in {"xgboost_regressor", "lightgbm_regressor", "catboost_regressor"}:
        return _optional_regression_estimator(model_name, random_seed=random_seed, parameters=parameters)
    if model_name in {"spectral_dkl_gp_regressor", "proto_spectral_regressor", "cls_former_regressor"}:
        return _experimental_estimator(model_name, "regression", parameters), {}
    raise ModelingMethodError("MODEL_UNSUPPORTED_FOR_TASK", "Unsupported regression model.", model=model_name)


def _component_grid(max_components: int, requested: Any) -> list[int]:
    if requested is not None:
        return [max(1, min(int(requested), max_components))]
    candidates = sorted({1, min(2, max_components), min(5, max_components), min(10, max_components)})
    return [value for value in candidates if value >= 1]


def _optional_boosting_grid(model_name: str) -> dict[str, list[Any]]:
    """Return compact but non-trivial optional-boosting tuning grids.

    These grids intentionally include anti-overfitting controls. A grid that
    varies only tree count and depth is too weak for small high-dimensional
    spectral datasets, where optional boosting models can memorize the training
    split while underperforming on validation/test splits.
    """

    if model_name in {"xgboost_classifier", "xgboost_regressor"}:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "reg_lambda": [1.0, 5.0],
        }
    if model_name in {"lightgbm_classifier", "lightgbm_regressor"}:
        return {
            "n_estimators": [50, 100, 200],
            "num_leaves": [7, 15, 31],
            "learning_rate": [0.03, 0.1],
            "reg_lambda": [0.0, 1.0],
        }
    if model_name in {"catboost_classifier", "catboost_regressor"}:
        return {
            "iterations": [50, 100, 200],
            "depth": [3, 4, 6],
            "learning_rate": [0.03, 0.1],
            "l2_leaf_reg": [3.0, 10.0],
        }
    raise ModelingMethodError("MODEL_UNSUPPORTED_FOR_TASK", "Unsupported optional boosting model.", model=model_name)


def _optional_classification_estimator(model_name: str, *, random_seed: int, parameters: dict[str, Any]) -> tuple[Any, dict[str, list[Any]]]:
    if model_name == "xgboost_classifier":
        cls = getattr(importlib.import_module("xgboost"), "XGBClassifier")
        return cls(random_state=random_seed, eval_metric="logloss", n_jobs=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)
    if model_name == "lightgbm_classifier":
        cls = getattr(importlib.import_module("lightgbm"), "LGBMClassifier")
        return cls(random_state=random_seed, verbosity=-1, n_jobs=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)
    cls = getattr(importlib.import_module("catboost"), "CatBoostClassifier")
    return cls(random_seed=random_seed, verbose=False, thread_count=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)


def _optional_regression_estimator(model_name: str, *, random_seed: int, parameters: dict[str, Any]) -> tuple[Any, dict[str, list[Any]]]:
    if model_name == "xgboost_regressor":
        cls = getattr(importlib.import_module("xgboost"), "XGBRegressor")
        return cls(random_state=random_seed, n_jobs=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)
    if model_name == "lightgbm_regressor":
        cls = getattr(importlib.import_module("lightgbm"), "LGBMRegressor")
        return cls(random_state=random_seed, verbosity=-1, n_jobs=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)
    cls = getattr(importlib.import_module("catboost"), "CatBoostRegressor")
    return cls(random_seed=random_seed, verbose=False, thread_count=parameters.get("n_jobs", 1)), _optional_boosting_grid(model_name)


def _experimental_estimator(model_name: str, task_type: str, parameters: dict[str, Any]) -> Any:
    module = importlib.import_module("spectral_core.modeling.experimental_small_sample")
    return module.build_experimental_model(model_name, task_type, parameters)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, model: Any, X: np.ndarray, encoder: LabelEncoder) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    auc = _roc_auc(model, X, y_true, encoder)
    if auc is not None:
        metrics["roc_auc"] = auc
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    true = np.asarray(y_true, dtype=float).reshape(-1)
    rmse = math.sqrt(mean_squared_error(true, pred))
    std = float(np.std(true, ddof=1)) if len(true) > 1 else 0.0
    return {
        "r2": float(r2_score(true, pred)) if len(true) > 1 else float("nan"),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(true, pred)),
        "rpd": float(std / rmse) if rmse > 0 and std > 0 else float("nan"),
    }


def _classification_predictions(model: Any, X: np.ndarray, y: np.ndarray, y_raw: np.ndarray, sample_ids: list[str], train_idx: list[int], val_idx: list[int], test_idx: list[int], encoder: LabelEncoder) -> list[dict[str, Any]]:
    rows = []
    probability = _prediction_probability(model, X)
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not indices:
            continue
        pred = model.predict(X[indices])
        for local_idx, idx in enumerate(indices):
            label = str(encoder.inverse_transform([int(pred[local_idx])])[0])
            row = {"sample_id": sample_ids[idx], "split": split_name, "y_true": str(y_raw[idx]), "y_pred": label, "predicted_label": label}
            if probability is not None:
                p = np.asarray(probability[idx], dtype=float)
                row["probability"] = float(np.max(p))
                row["max_probability"] = float(np.max(p))
                row["predictive_entropy"] = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
            rows.append(row)
    return rows


def _regression_predictions(model: Any, X: np.ndarray, y: np.ndarray, sample_ids: list[str], train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> list[dict[str, Any]]:
    rows = []
    pred_all, std_all = _regression_prediction_with_std(model, X)
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for idx in indices:
            true = float(y[idx])
            pred = float(pred_all[idx])
            row = {"sample_id": sample_ids[idx], "split": split_name, "y_true": true, "y_pred": pred, "residual": true - pred, "absolute_error": abs(true - pred)}
            if std_all is not None:
                std = float(std_all[idx])
                row.update({"y_pred_mean": pred, "y_pred_std": std, "lower_95": pred - 1.96 * std, "upper_95": pred + 1.96 * std})
            rows.append(row)
    return rows


def _regression_prediction_with_std(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(model, "predict_with_std"):
        mean, std = model.predict_with_std(X)
        return np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        if isinstance(final, GaussianProcessRegressor):
            transformed = X
            for _, step in model.steps[:-1]:
                transformed = step.transform(transformed)
            mean, std = final.predict(transformed, return_std=True)
            return np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1), None


def _prediction_probability(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    if isinstance(model, Pipeline) and hasattr(model[-1], "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    return None


def _roc_auc(model: Any, X: np.ndarray, y_true: np.ndarray, encoder: LabelEncoder) -> float | None:
    probability = _prediction_probability(model, X)
    if probability is None or len(np.unique(y_true)) < 2:
        return None
    try:
        if len(encoder.classes_) == 2:
            return float(roc_auc_score(y_true, probability[:, 1]))
        return float(roc_auc_score(y_true, probability, multi_class="ovr", average="weighted"))
    except ValueError:
        return None


def _selection_summary(best: dict[str, Any], candidates: list[dict[str, Any]], *, used_val: bool, eval_role: str) -> dict[str, Any]:
    inner_cv_folds = _cv_folds_from_selection_split(str(best["selection_split"]))
    param_search_enabled = bool(best.get("param_search_enabled", True))
    mode = best.get("selection_mode") or "modeling_internal_selection"
    return {
        "tuning_split": "val" if used_val else best["selection_split"],
        "selection_metric": best["selection_metric"],
        "selected_score": best["selection_score"],
        "candidate_count": len(candidates),
        "test_used_for_selection": False,
        "model_selection_mode": mode,
        "param_search_enabled": param_search_enabled,
        "model_params_source": best.get("model_params_source") or "modeling_internal_selection",
        "outer_validation_used_for_selection": eval_role == "val" and used_val,
        "hyperparameter_selection": _hyperparameter_selection_summary(eval_role=eval_role, cv_folds=inner_cv_folds, used_val=used_val, param_search_enabled=param_search_enabled, model_selection_mode=mode),
        "candidates": [
            {"model_type": item["model_type"], "parameters": item["parameters"], "selection_score": item["selection_score"], "selection_metric": item["selection_metric"], "selection_split": item["selection_split"], "param_search_enabled": bool(item.get("param_search_enabled", True)), "model_params_source": item.get("model_params_source")}
            for item in candidates
        ],
    }


def _hyperparameter_selection_summary(*, eval_role: str, cv_folds: int | None, used_val: bool = False, param_search_enabled: bool = True, model_selection_mode: str = "modeling_internal_selection") -> dict[str, Any]:
    if not param_search_enabled:
        return {
            "strategy": model_selection_mode,
            "inner_cv_folds": None,
            "selection_scope": "locked_parameters",
            "outer_validation_used_for_selection": False,
            "param_search_enabled": False,
            "warning": None,
        }
    if eval_role == "val":
        return {
            "strategy": "inner_cv" if not used_val else "outer_validation_selection",
            "inner_cv_folds": cv_folds,
            "selection_scope": "outer_train_only" if not used_val else "outer_validation",
            "outer_validation_used_for_selection": used_val,
            "warning": None if not used_val else "CV metrics may be optimistic because fold validation was used for parameter selection.",
        }
    return {
        "strategy": "validation_split" if used_val else "inner_cv",
        "inner_cv_folds": cv_folds,
        "selection_scope": "partition_val" if used_val else "train_only",
        "outer_validation_used_for_selection": False,
        "warning": None,
    }


def _cv_folds_from_selection_split(value: str) -> int | None:
    prefix = "train_cv_"
    suffix = "fold"
    if value.startswith(prefix) and value.endswith(suffix):
        middle = value[len(prefix) : -len(suffix)]
        try:
            return int(middle)
        except ValueError:
            return None
    return None


def _metric_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in metrics for key, value in row.items() if isinstance(value, (int, float)) and math.isfinite(float(value))})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in metrics if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]
        if not values:
            continue
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"metrics_mean": means, "metrics_std": stds, "n_iterations": len(metrics)}


def _param_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _valid_cv_folds(y_train: np.ndarray, *, requested: int, task_type: str) -> int:
    if len(y_train) < 2:
        raise ModelingMethodError("CV_TRAIN_TOO_SMALL", "Internal CV requires at least two train samples.")
    folds = max(2, min(requested, len(y_train)))
    if task_type == "classification":
        _, counts = np.unique(y_train, return_counts=True)
        folds = min(folds, int(counts.min()))
        if folds < 2:
            raise ModelingMethodError("CV_CLASS_COUNT_TOO_SMALL", "Classification CV requires at least two train samples per class.")
    return folds


def _normalize_task(task_type: str) -> str:
    return task_type.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_model(model: str) -> str:
    return normalize_model_name(model)


def _normalize_model_set_alias(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _all_float(values: list[Any]) -> bool:
    try:
        for value in values:
            float(value)
    except (TypeError, ValueError):
        return False
    return True


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

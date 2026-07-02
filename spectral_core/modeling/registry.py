"""Model registry, aliases, dependency checks, and confirmation policy."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    task_type: str
    family: str
    dependency: str | None = None
    experimental: bool = False


CLASSIFICATION_MODELS = {
    "logistic_regression",
    "linear_svm",
    "svm",
    "lda",
    "qda",
    "gaussian_nb",
    "knn_classifier",
    "random_forest_classifier",
    "extra_trees_classifier",
    "gradient_boosting_classifier",
    "pls_da",
    "simca",
    "mlp_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
    "spectral_dkl_gp_classifier",
    "proto_spectral_classifier",
    "cls_former_classifier",
    "cls_former_embedding_svm",
}

REGRESSION_MODELS = {
    "plsr",
    "pcr",
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "bayesian_ridge",
    "svr",
    "knn_regressor",
    "random_forest_regressor",
    "extra_trees_regressor",
    "gradient_boosting_regressor",
    "gpr",
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
    "spectral_dkl_gp_regressor",
    "proto_spectral_regressor",
    "cls_former_regressor",
}

CHEMOMETRICS_MODELS = {"pls_da", "simca", "plsr", "pcr"}
OPTIONAL_DEPENDENCIES = {
    "xgboost_classifier": "xgboost",
    "xgboost_regressor": "xgboost",
    "lightgbm_classifier": "lightgbm",
    "lightgbm_regressor": "lightgbm",
    "catboost_classifier": "catboost",
    "catboost_regressor": "catboost",
}
EXPERIMENTAL_MODELS = {
    "spectral_dkl_gp_classifier",
    "spectral_dkl_gp_regressor",
    "proto_spectral_classifier",
    "proto_spectral_regressor",
    "cls_former_classifier",
    "cls_former_regressor",
    "cls_former_embedding_svm",
}

ALIASES = {
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "svc": "svm",
    "svm_classifier": "svm",
    "linear_svc": "linear_svm",
    "linear_svm_classifier": "linear_svm",
    "lda_classifier": "lda",
    "qda_classifier": "qda",
    "naive_bayes": "gaussian_nb",
    "gnb": "gaussian_nb",
    "rf": "random_forest_classifier",
    "random_forest": "random_forest_classifier",
    "extra_trees": "extra_trees_classifier",
    "gradient_boosting": "gradient_boosting_classifier",
    "knn": "knn_classifier",
    "plsda": "pls_da",
    "pls_discriminant_analysis": "pls_da",
    "pls": "plsr",
    "pls_regression": "plsr",
    "principal_component_regression": "pcr",
    "ols": "linear_regression",
    "ridge_regression": "ridge",
    "elasticnet": "elastic_net",
    "svm_regression": "svr",
    "rf_regressor": "random_forest_regressor",
    "extra_trees_reg": "extra_trees_regressor",
    "gradient_boosting_reg": "gradient_boosting_regressor",
    "gaussian_process_regression": "gpr",
    "xgb_classifier": "xgboost_classifier",
    "xgb_regressor": "xgboost_regressor",
    "lgbm_classifier": "lightgbm_classifier",
    "lgbm_regressor": "lightgbm_regressor",
    "dkl_gp_classifier": "spectral_dkl_gp_classifier",
    "dkl_gp_regressor": "spectral_dkl_gp_regressor",
    "proto_classifier": "proto_spectral_classifier",
    "proto_regressor": "proto_spectral_regressor",
    "clsformer": "cls_former_classifier",
    "cls_former": "cls_former_classifier",
    "cls_former_svm": "cls_former_embedding_svm",
}


# Deterministic parameters for optimizer validation-only comparisons. These
# values make each trial executable with model selection disabled; they are not
# a replacement for an explicitly confirmed parameter grid.
COMPARISON_FIXED_PARAMETERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0},
    "linear_svm": {"C": 1.0},
    "svm": {"C": 1.0, "gamma": "scale"},
    "lda": {"shrinkage": None},
    "qda": {"reg_param": 0.1},
    "gaussian_nb": {"var_smoothing": 1e-9},
    "knn_classifier": {"n_neighbors": 5},
    "random_forest_classifier": {"n_estimators": 100, "max_depth": 5},
    "extra_trees_classifier": {"n_estimators": 100, "max_depth": 5},
    "gradient_boosting_classifier": {"n_estimators": 100, "learning_rate": 0.1},
    "pls_da": {"n_components": 5},
    "simca": {"n_components": 5, "quantile": 0.95},
    "plsr": {"n_components": 5},
    "pcr": {"n_components": 5},
    "linear_regression": {"fit_intercept": True},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.01},
    "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
    "bayesian_ridge": {"alpha_1": 1e-6, "lambda_1": 1e-6},
    "svr": {"C": 1.0, "gamma": "scale"},
    "knn_regressor": {"n_neighbors": 5, "weights": "uniform"},
    "random_forest_regressor": {"n_estimators": 100, "max_depth": 5},
    "extra_trees_regressor": {"n_estimators": 100, "max_depth": 5},
    "gradient_boosting_regressor": {"n_estimators": 100, "learning_rate": 0.1},
    "gpr": {"alpha": 1e-6},
}

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "spectral_dkl_gp_classifier": {
        "preprojection": "pca",
        "n_components": 50,
        "embedding_dim": 32,
        "kernel": "rbf",
        "epochs": 100,
        "lr": 0.001,
        "random_state": 42,
        "device": "auto",
    },
    "spectral_dkl_gp_regressor": {
        "preprojection": "pca",
        "n_components": 50,
        "embedding_dim": 32,
        "kernel": "rbf",
        "epochs": 100,
        "lr": 0.001,
        "random_state": 42,
        "device": "auto",
    },
    "proto_spectral_classifier": {
        "encoder_type": "mlp",
        "embedding_dim": 32,
        "metric": "euclidean",
        "epochs": 100,
        "batch_size": 8,
        "lr": 0.001,
        "temperature": 0.1,
        "random_state": 42,
        "device": "auto",
    },
    "proto_spectral_regressor": {
        "encoder_type": "mlp",
        "embedding_dim": 32,
        "metric": "euclidean",
        "epochs": 100,
        "batch_size": 8,
        "lr": 0.001,
        "temperature": 0.1,
        "random_state": 42,
        "device": "auto",
    },
    "cls_former_classifier": {
        "epochs": 100,
        "batch_size": 8,
        "alpha": 0.5,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "feature_dim": 32,
        "dropout": 0.3,
        "lr": 0.0005,
        "weight_decay": 0.0005,
        "patience": 20,
        "random_state": 42,
        "device": "auto",
    },
    "cls_former_embedding_svm": {
        "epochs": 100,
        "batch_size": 8,
        "alpha": 0.5,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "feature_dim": 32,
        "dropout": 0.3,
        "lr": 0.0005,
        "weight_decay": 0.0005,
        "patience": 20,
        "random_state": 42,
        "device": "auto",
    },
    "cls_former_regressor": {
        "epochs": 100,
        "batch_size": 8,
        "alpha": 0.3,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "feature_dim": 32,
        "dropout": 0.3,
        "lr": 0.0005,
        "weight_decay": 0.0005,
        "patience": 20,
        "random_state": 42,
        "device": "auto",
    },
}

CRITICAL_PARAMETERS = {
    "spectral_dkl_gp_classifier": ("preprojection", "n_components", "embedding_dim", "kernel", "epochs", "lr", "random_state"),
    "spectral_dkl_gp_regressor": ("preprojection", "n_components", "embedding_dim", "kernel", "epochs", "lr", "random_state"),
    "proto_spectral_classifier": ("encoder_type", "embedding_dim", "metric", "epochs", "batch_size", "lr", "temperature", "random_state"),
    "proto_spectral_regressor": ("encoder_type", "embedding_dim", "metric", "epochs", "batch_size", "lr", "temperature", "random_state"),
    "cls_former_classifier": ("epochs", "batch_size", "alpha", "device"),
    "cls_former_embedding_svm": ("epochs", "batch_size", "alpha", "device"),
    "cls_former_regressor": ("epochs", "batch_size", "alpha", "device"),
}


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(normalized, normalized)


def comparison_fixed_parameters(name: str) -> dict[str, Any]:
    """Return a copy of the deterministic validation-only parameter policy."""

    return dict(COMPARISON_FIXED_PARAMETERS.get(normalize_model_name(name)) or {})


def model_spec(name: str) -> ModelSpec:
    canonical = normalize_model_name(name)
    if canonical in CLASSIFICATION_MODELS:
        task = "classification"
    elif canonical in REGRESSION_MODELS:
        task = "regression"
    else:
        raise KeyError(canonical)
    if canonical in EXPERIMENTAL_MODELS:
        family = "small_sample_experimental"
    elif canonical in OPTIONAL_DEPENDENCIES:
        family = "optional_boosting"
    elif canonical in CHEMOMETRICS_MODELS:
        family = "chemometrics"
    else:
        family = "traditional_ml"
    dependency = "torch" if canonical in EXPERIMENTAL_MODELS else OPTIONAL_DEPENDENCIES.get(canonical)
    return ModelSpec(task, family, dependency=dependency, experimental=canonical in EXPERIMENTAL_MODELS)


def load_model_config(value: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        payload = dict(value)
    else:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if "models" in payload and isinstance(payload["models"], dict):
        return {normalize_model_name(key): dict(params or {}) for key, params in payload["models"].items()}
    if "method" in payload:
        return {normalize_model_name(str(payload["method"])): dict(payload.get("params") or {})}
    return {normalize_model_name(key): dict(params or {}) for key, params in payload.items() if isinstance(params, dict)}


def prepare_model_parameters(
    models: list[str],
    config: dict[str, Any] | None,
    *,
    common_params: dict[str, Any] | None = None,
    auto_confirm_defaults: bool,
    random_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    config = config or {}
    common = {key: value for key, value in (common_params or {}).items() if value is not None}
    resolved: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    confirmations: list[dict[str, Any]] = []
    for model in models:
        params = dict(config.get(model) or {})
        params.update(common)
        defaults = dict(MODEL_DEFAULTS.get(model) or {})
        if "random_state" in defaults:
            defaults["random_state"] = random_seed if "random_state" not in params else params["random_state"]
        missing = [field for field in CRITICAL_PARAMETERS.get(model, ()) if params.get(field) is None]
        if missing and not auto_confirm_defaults:
            confirmations.append(
                {
                    "model": model,
                    "model_family": model_spec(model).family,
                    "fields": missing,
                    "recommended": {field: defaults.get(field) for field in missing},
                    "reason": "Experimental small-sample models require explicit confirmation of training-sensitive parameters.",
                }
            )
        merged = dict(defaults)
        merged.update(params)
        resolved[model] = merged
        sources[model] = {
            key: ("user_specified" if key in params else "defaulted")
            for key in merged
        }
    return resolved, sources, confirmations


def missing_dependencies(models: list[str]) -> list[dict[str, str]]:
    missing = []
    for model in models:
        dependency = model_spec(model).dependency
        if dependency and importlib.util.find_spec(dependency) is None:
            missing.append(
                {
                    "model": model,
                    "dependency": dependency,
                    "reason": f"{model} requires the optional Python package '{dependency}'.",
                }
            )
    return missing

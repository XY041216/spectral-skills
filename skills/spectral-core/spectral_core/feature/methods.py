"""Leakage-safe spectral feature extraction and variable selection methods."""

from __future__ import annotations

import math
import warnings as py_warnings
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import DictionaryLearning, FastICA, KernelPCA, NMF, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, f_regression
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from .io import FeaturePackage
from .parameter_policy import normalize_feature_method


class FeatureMethodError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


SUPPORTED_METHODS = {
    "none",
    "pca",
    "kernel_pca",
    "sparse_pca",
    "nmf",
    "ica_embedding",
    "lda_projection",
    "dct_features",
    "fft_features",
    "dictionary_learning",
    "umap_embedding",
    "isomap_embedding",
    "lle_embedding",
    "tsne_embedding",
    "variance_threshold",
    "select_by_band_range",
    "select_by_band_indices",
    "pls_latent_variables",
    "vip",
    "correlation_filter",
    "select_k_best",
    "anova_f",
    "f_regression",
    "interval_pls",
    "spa",
    "cars",
    "uve",
    "mcuve",
    "autoencoder_embedding",
    "denoising_autoencoder_embedding",
    "cnn_1d_embedding",
    "cls_former_embedding",
    "resnet1d_embedding",
    "masked_spectral_autoencoder_embedding",
    "contrastive_spectral_embedding",
}
DEEP_EMBEDDING_METHODS = {
    "autoencoder_embedding",
    "denoising_autoencoder_embedding",
    "cnn_1d_embedding",
    "cls_former_embedding",
    "resnet1d_embedding",
    "masked_spectral_autoencoder_embedding",
    "contrastive_spectral_embedding",
}
SUPERVISED_METHODS = {
    "pls_latent_variables",
    "vip",
    "correlation_filter",
    "select_k_best",
    "anova_f",
    "f_regression",
    "interval_pls",
    "lda_projection",
    "cars",
    "uve",
    "mcuve",
}
TRAIN_FIT_METHODS = {
    "pca",
    "kernel_pca",
    "sparse_pca",
    "nmf",
    "ica_embedding",
    "dictionary_learning",
    "umap_embedding",
    "isomap_embedding",
    "lle_embedding",
    "variance_threshold",
    "spa",
    *DEEP_EMBEDDING_METHODS,
    *SUPERVISED_METHODS,
}
VISUALIZATION_ONLY_METHODS = {"tsne_embedding"}
EXPERIMENTAL_GATED_METHODS = {
    "opls",
    "osc",
    "wavelet_features",
    "transformer_embedding",
    "attention_pooling",
    "self_supervised_spectral_embedding",
}
ADVANCED_METHODS = {
    "pls",
    "pls_da",
    "plsr",
    "mutual_information",
    "lasso",
    "elasticnet",
    "deep_embedding",
    "embedding",
    "auto_search",
    "optimizer",
}


def parse_method(method: str | None) -> str:
    if method is None or not str(method).strip():
        raise FeatureMethodError("FEATURE_METHOD_REQUIRED", "Please choose a supported feature extraction or selection method.")
    normalized = _normalize_method(str(method))
    if normalized in EXPERIMENTAL_GATED_METHODS:
        raise FeatureMethodError(
            "FEATURE_METHOD_EXPERIMENTAL_GATED",
            "This feature method is registered as an experimental spectral embedding and requires a dedicated training protocol, sample-size check, and explicit user confirmation before implementation.",
            method=normalized,
            recommended_route="Use spectral-report for visualization from existing embeddings, or create a dedicated feature training contract before modeling.",
        )
    if normalized in ADVANCED_METHODS:
        raise FeatureMethodError("FEATURE_METHOD_NOT_IMPLEMENTED", "This feature version does not implement the requested embedding, penalized selector, or optimizer search.", method=normalized)
    if normalized not in SUPPORTED_METHODS:
        raise FeatureMethodError("FEATURE_METHOD_UNSUPPORTED", "Unsupported feature method for this MVP.", method=normalized)
    return normalized


def requires_train_fit(method: str) -> bool:
    return method in TRAIN_FIT_METHODS


def requires_y(method: str) -> bool:
    return method in SUPERVISED_METHODS


def apply_feature_method(
    package: FeaturePackage,
    *,
    method: str,
    train_indices: list[int],
    n_components: int | None,
    explained_variance: float | None,
    variance_threshold: float | None,
    band_min: float | None,
    band_max: float | None,
    band_indices: str | None,
    feature_names: str | None,
    index_base: int,
    top_k: int | None = None,
    score_threshold: float | None = None,
    n_intervals: int | None = None,
    n_runs: int | None = None,
    sample_ratio: float | None = None,
    cv: int | None = None,
    random_state: int | None = None,
    task_type: str | None = None,
    correlation_method: str | None = None,
    interval_mode: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    noise_std: float | None = None,
    mask_ratio: float | None = None,
    temperature: float | None = None,
    patch_size: int | None = None,
    device: str | None = None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    if method == "none":
        state = {
            "method": "none",
            "parameters": {},
            "fitted": {"scope": "identity"},
            "fit_scope": "train_only",
            "transform_scope": "train_val_test",
        }
        return [list(row) for row in package.X], list(package.feature_names), [list(row) for row in package.band_axis_rows], state
    if method == "pca":
        return _apply_pca(package, train_indices, n_components=n_components, explained_variance=explained_variance)
    if method == "kernel_pca":
        return _apply_kernel_pca(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "sparse_pca":
        return _apply_sparse_pca(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "nmf":
        return _apply_nmf(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "ica_embedding":
        return _apply_ica(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "lda_projection":
        return _apply_lda_projection(package, train_indices, n_components=n_components, task_type=task_type)
    if method == "dct_features":
        return _apply_dct_features(package, n_components=n_components)
    if method == "fft_features":
        return _apply_fft_features(package, n_components=n_components)
    if method == "dictionary_learning":
        return _apply_dictionary_learning(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "umap_embedding":
        return _apply_umap_embedding(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "isomap_embedding":
        return _apply_isomap_embedding(package, train_indices, n_components=n_components)
    if method == "lle_embedding":
        return _apply_lle_embedding(package, train_indices, n_components=n_components, random_state=random_state)
    if method == "tsne_embedding":
        return _apply_tsne_embedding(package, train_indices, n_components=n_components, random_state=random_state)
    if method in DEEP_EMBEDDING_METHODS:
        return _apply_deep_embedding(
            package,
            train_indices,
            method=method,
            n_components=n_components,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            noise_std=noise_std,
            mask_ratio=mask_ratio,
            temperature=temperature,
            patch_size=patch_size,
            random_state=random_state,
            device=device,
        )
    if method == "variance_threshold":
        return _apply_variance_threshold(package, train_indices, threshold=variance_threshold)
    if method == "select_by_band_range":
        return _apply_band_range(package, band_min=band_min, band_max=band_max)
    if method == "select_by_band_indices":
        return _apply_band_indices(package, band_indices=band_indices, feature_names=feature_names, index_base=index_base)
    if method == "pls_latent_variables":
        return _apply_pls_latent_variables(package, train_indices, n_components=n_components, task_type=task_type)
    if method == "vip":
        return _apply_vip(
            package,
            train_indices,
            n_components=n_components,
            top_k=top_k,
            score_threshold=score_threshold,
            task_type=task_type,
        )
    if method == "correlation_filter":
        return _apply_correlation_filter(
            package,
            train_indices,
            top_k=top_k,
            score_threshold=score_threshold,
            task_type=task_type,
            correlation_method=correlation_method,
        )
    if method in {"select_k_best", "anova_f", "f_regression"}:
        return _apply_select_k_best(
            package,
            train_indices,
            method=method,
            top_k=top_k,
            task_type=task_type,
        )
    if method == "interval_pls":
        return _apply_interval_pls(
            package,
            train_indices,
            n_components=n_components,
            n_intervals=n_intervals,
            cv=cv,
            task_type=task_type,
            interval_mode=interval_mode,
            random_state=random_state,
        )
    if method == "spa":
        return _apply_spa(package, train_indices, top_k=top_k)
    if method == "cars":
        return _apply_cars(
            package,
            train_indices,
            n_components=n_components,
            top_k=top_k,
            n_runs=n_runs,
            sample_ratio=sample_ratio,
            cv=cv,
            random_state=random_state,
            task_type=task_type,
        )
    if method in {"uve", "mcuve"}:
        return _apply_uve(
            package,
            train_indices,
            method=method,
            n_components=n_components,
            top_k=top_k,
            score_threshold=score_threshold,
            n_runs=n_runs,
            sample_ratio=sample_ratio,
            random_state=random_state,
            task_type=task_type,
        )
    raise FeatureMethodError("FEATURE_METHOD_UNSUPPORTED", "Unsupported feature method.", method=method)


def _normalize_method(method: str) -> str:
    normalized = normalize_feature_method(method)
    aliases = {
        "identity": "none",
        "pass_through": "none",
        "passthrough": "none",
        "var_threshold": "variance_threshold",
        "low_variance": "variance_threshold",
        "kpca": "kernel_pca",
        "kernelpca": "kernel_pca",
        "sparsepca": "sparse_pca",
        "sparse_principal_components": "sparse_pca",
        "fastica": "ica_embedding",
        "ica": "ica_embedding",
        "independent_component_analysis": "ica_embedding",
        "lda": "lda_projection",
        "linear_discriminant_projection": "lda_projection",
        "linear_discriminant_analysis_projection": "lda_projection",
        "dct": "dct_features",
        "fft": "fft_features",
        "dictionary": "dictionary_learning",
        "dict_learning": "dictionary_learning",
        "umap": "umap_embedding",
        "tsne": "tsne_embedding",
        "t_sne": "tsne_embedding",
        "isomap": "isomap_embedding",
        "lle": "lle_embedding",
        "locally_linear_embedding": "lle_embedding",
        "opls_da": "opls",
        "orthogonal_signal_correction": "osc",
        "wavelet": "wavelet_features",
        "autoencoder": "autoencoder_embedding",
        "denoising_autoencoder": "denoising_autoencoder_embedding",
        "cls_former": "cls_former_embedding",
        "clsformer": "cls_former_embedding",
        "transformer": "cls_former_embedding",
        "transformer_embedding": "cls_former_embedding",
        "spectral_transformer": "cls_former_embedding",
        "spectral_transformer_embedding": "cls_former_embedding",
        "cnn": "cnn_1d_embedding",
        "cnn1d": "cnn_1d_embedding",
        "resnet1d": "resnet1d_embedding",
        "self_supervised_embedding": "self_supervised_spectral_embedding",
        "band_range": "select_by_band_range",
        "wavelength_range": "select_by_band_range",
        "wavenumber_range": "select_by_band_range",
        "band_indices": "select_by_band_indices",
        "feature_indices": "select_by_band_indices",
    }
    return aliases.get(normalized, normalized)


def _apply_pca(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    explained_variance: float | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    if n_components is not None and explained_variance is not None:
        raise FeatureMethodError("PCA_RETENTION_CONFLICT", "Use either n_components or explained_variance for PCA, not both.")
    if len(train_indices) < 2:
        raise FeatureMethodError("PCA_TRAIN_TOO_SMALL", "PCA requires at least two train samples.")

    X = np.asarray(package.X, dtype=float)
    train = X[train_indices, :]
    mean = train.mean(axis=0)
    centered_train = train - mean
    _, singular_values, vt = np.linalg.svd(centered_train, full_matrices=False)
    max_components = min(vt.shape[0], X.shape[1], len(train_indices) - 1)
    if max_components < 1:
        raise FeatureMethodError("PCA_NO_COMPONENTS", "PCA could not determine any components.")

    explained = (singular_values**2) / max(len(train_indices) - 1, 1)
    total_variance = float(explained.sum())
    ratios = explained / total_variance if total_variance > 0 else np.zeros_like(explained)
    if total_variance <= 0:
        raise FeatureMethodError("PCA_ZERO_TRAIN_VARIANCE", "PCA cannot fit a projection when train spectra have zero total variance.")

    method_warnings: list[dict[str, Any]] = []
    if n_components is not None or explained_variance is None:
        requested = 10 if n_components is None else int(n_components)
        if requested < 1:
            raise FeatureMethodError("PCA_N_COMPONENTS_INVALID", "n_components must be at least 1.", n_components=requested)
        keep = min(requested, max_components)
        if keep != requested:
            method_warnings.append(_warning("N_COMPONENTS_CLIPPED", "PCA n_components was clipped to the train-set legal maximum.", requested=requested, used=keep))
        retention = {"n_components": keep}
    else:
        assert explained_variance is not None
        if explained_variance <= 0 or explained_variance > 1:
            raise FeatureMethodError("PCA_EXPLAINED_VARIANCE_INVALID", "explained_variance must be in the interval (0, 1].", explained_variance=explained_variance)
        cumulative = np.cumsum(ratios)
        keep = int(np.searchsorted(cumulative, explained_variance, side="left") + 1)
        keep = min(max(keep, 1), max_components)
        retention = {"explained_variance": explained_variance}

    components = vt[:keep, :]
    transformed = (X - mean) @ components.T
    feature_names = [f"PC{idx}" for idx in range(1, keep + 1)]
    band_axis_rows = [[idx - 1, f"PC{idx}", "principal_component"] for idx in range(1, keep + 1)]
    state = {
        "method": "pca",
        "parameters": retention,
        "fitted": {
            "mean": mean.tolist(),
            "components": components.tolist(),
            "explained_variance": explained[:keep].tolist(),
            "explained_variance_ratio": ratios[:keep].tolist(),
            "cumulative_explained_variance_ratio": np.cumsum(ratios[:keep]).tolist(),
            "fit_sample_count": len(train_indices),
        },
        "fit_scope": "train_only",
        "transform_scope": "train_val_test",
        "warnings": method_warnings,
        "_fitted_transformer": {"method": "pca", "mean": mean.tolist(), "components": components.tolist()},
    }
    return transformed.tolist(), feature_names, band_axis_rows, state


def _apply_kernel_pca(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=2)
    model = KernelPCA(n_components=keep, kernel="rbf", gamma=None, fit_inverse_transform=False, random_state=random_state or 42)
    model.fit(X[train_indices])
    transformed = np.asarray(model.transform(X), dtype=float)
    return _projection_output(
        package,
        transformed,
        method="kernel_pca",
        method_family="kernel_projection",
        feature_mode="modeling_embedding",
        feature_prefix="KPCA",
        axis_kind="kernel_principal_component",
        params={"n_components": keep, "kernel": "rbf", "gamma": None, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "kernel": "rbf"},
        warnings=warnings,
        transformer=model,
    )


def _apply_sparse_pca(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=2)
    max_iter = 1000
    model = SparsePCA(n_components=keep, random_state=random_state or 42, max_iter=max_iter)
    convergence = _fit_with_convergence(
        model, X[train_indices], warnings, max_iter=max_iter, method="sparse_pca", random_seed=random_state or 42
    )
    transformed = np.asarray(model.transform(X), dtype=float)
    return _projection_output(
        package,
        transformed,
        method="sparse_pca",
        method_family="sparse_projection",
        feature_mode="modeling_embedding",
        feature_prefix="SPCA",
        axis_kind="sparse_principal_component",
        params={"n_components": keep, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "components": model.components_.tolist(), "convergence": convergence},
        warnings=warnings,
        transformer=model,
    )


def _apply_nmf(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    train = X[train_indices]
    keep, warnings = _resolve_n_components(n_components, train, default=2)
    input_min = float(np.nanmin(X))
    if input_min < 0:
        raise FeatureMethodError(
            "NMF_NONNEGATIVE_REQUIRED",
            "NMF requires non-negative input. Current X contains negative values after preprocessing.",
            method="nmf",
            requires_nonnegative_X=True,
            input_min_value=input_min,
            nonnegative_check="failed",
        )
    max_iter = 1000
    model = NMF(n_components=keep, init="nndsvda", random_state=random_state or 42, max_iter=max_iter)
    convergence = _fit_with_convergence(
        model, X[train_indices], warnings, max_iter=max_iter, method="nmf", random_seed=random_state or 42
    )
    transformed = np.asarray(model.transform(X), dtype=float)
    return _projection_output(
        package,
        transformed,
        method="nmf",
        method_family="nonnegative_projection",
        feature_mode="modeling_embedding",
        feature_prefix="NMF",
        axis_kind="nmf_component",
        params={"n_components": keep, "random_state": random_state or 42},
        fitted={
            "fit_sample_count": len(train_indices),
            "components": model.components_.tolist(),
            "requires_nonnegative_X": True,
            "input_min_value": input_min,
            "nonnegative_check": "passed",
            "convergence": convergence,
        },
        warnings=warnings,
        transformer=model,
    )


def _apply_ica(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=2)
    max_iter = 1000
    model = FastICA(n_components=keep, random_state=random_state or 42, max_iter=max_iter, whiten="unit-variance")
    convergence = _fit_with_convergence(
        model, X[train_indices], warnings, max_iter=max_iter, method="ica_embedding", random_seed=random_state or 42
    )
    transformed = np.asarray(model.transform(X), dtype=float)
    return _projection_output(
        package,
        transformed,
        method="ica_embedding",
        method_family="independent_component_projection",
        feature_mode="modeling_embedding",
        feature_prefix="ICA",
        axis_kind="independent_component",
        params={"n_components": keep, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "components": model.components_.tolist(), "convergence": convergence},
        warnings=warnings,
        transformer=model,
    )


def _apply_lda_projection(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    if (task_type or package.contract.get("task_hint") or "classification") != "classification":
        raise FeatureMethodError("LDA_PROJECTION_TASK_MISMATCH", "lda_projection is only valid for classification tasks.")
    if package.y_rows is None:
        raise FeatureMethodError("Y_REQUIRED", "lda_projection requires y.csv.")
    raw = [str(row[0]).strip() for row in package.y_rows]
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw)
    train_y = y[train_indices]
    classes = len(set(train_y.tolist()))
    if classes < 2:
        raise FeatureMethodError("LDA_CLASS_COUNT_INVALID", "lda_projection requires at least two classes in train.")
    X = np.asarray(package.X, dtype=float)
    max_components = min(classes - 1, X.shape[1])
    requested = max_components if n_components is None else int(n_components)
    if requested < 1:
        raise FeatureMethodError("N_COMPONENTS_INVALID", "n_components must be at least 1.", n_components=requested)
    keep = min(requested, max_components)
    warnings = []
    if keep != requested:
        warnings.append(
            _warning(
                "N_COMPONENTS_CLIPPED",
                "LDA projection n_components was clipped to the n_classes - 1 limit.",
                requested_n_components=requested,
                effective_n_components=keep,
                reason="n_classes - 1 limit",
            )
        )
    model = LinearDiscriminantAnalysis(n_components=keep)
    model.fit(X[train_indices], train_y)
    transformed = np.asarray(model.transform(X), dtype=float)
    names, axis = _derived_axis("LDA", transformed.shape[1], "lda_discriminant")
    artifacts = {
        "components.csv": _matrix_artifact(
            ["component", *package.feature_names],
            [[names[idx], *np.asarray(model.scalings_)[:, idx].tolist()] for idx in range(transformed.shape[1])],
        )
    }
    state = _state(
        "lda_projection",
        "supervised_projection",
        "classification",
        {
            "n_components": transformed.shape[1],
            "requested_n_components": requested,
            "effective_n_components": transformed.shape[1],
            "n_classes": classes,
            "component_limit": "n_classes - 1",
        },
        package.n_features,
        transformed.shape[1],
        "supervised_modeling_embedding",
        fitted={
            "fit_sample_count": len(train_indices),
            "classes": encoder.classes_.tolist(),
            "supervised_y_used": True,
            "val_test_y_used_for_fit": False,
        },
        artifacts=artifacts,
        warnings=warnings,
    )
    state["_fitted_transformer"] = model
    return transformed.tolist(), names, axis, state


def _apply_dct_features(
    package: FeaturePackage,
    *,
    n_components: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    try:
        from scipy.fft import dct
    except Exception as exc:  # pragma: no cover - scipy is expected with sklearn but keep the error explicit.
        raise FeatureMethodError("OPTIONAL_DEPENDENCY_MISSING", "dct_features requires scipy.fft.dct.") from exc
    X = np.asarray(package.X, dtype=float)
    keep = _validate_component_count(n_components, X.shape[1], default=min(10, X.shape[1]))
    transformed = np.asarray(dct(X, type=2, norm="ortho", axis=1)[:, :keep], dtype=float)
    names, axis = _derived_axis("DCT", keep, "dct_coefficient")
    state = _deterministic_signal_state("dct_features", {"n_components": keep}, package.n_features, keep)
    return transformed.tolist(), names, axis, state


def _apply_fft_features(
    package: FeaturePackage,
    *,
    n_components: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    spectrum = np.abs(np.fft.rfft(X, axis=1))
    keep = _validate_component_count(n_components, spectrum.shape[1], default=min(10, spectrum.shape[1]))
    transformed = spectrum[:, :keep]
    names, axis = _derived_axis("FFT_MAG", keep, "fft_magnitude")
    state = _deterministic_signal_state("fft_features", {"n_components": keep, "transform": "rfft_magnitude"}, package.n_features, keep)
    return transformed.tolist(), names, axis, state


def _apply_dictionary_learning(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=2)
    max_iter = 500
    model = DictionaryLearning(n_components=keep, random_state=random_state or 42, max_iter=max_iter, transform_algorithm="lasso_lars")
    convergence = _fit_with_convergence(
        model, X[train_indices], warnings, max_iter=max_iter, method="dictionary_learning", random_seed=random_state or 42
    )
    transformed = np.asarray(model.transform(X), dtype=float)
    return _projection_output(
        package,
        transformed,
        method="dictionary_learning",
        method_family="dictionary_projection",
        feature_mode="modeling_embedding",
        feature_prefix="DICT",
        axis_kind="dictionary_atom_score",
        params={"n_components": keep, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "components": model.components_.tolist(), "convergence": convergence},
        warnings=warnings,
        transformer=model,
    )


def _apply_umap_embedding(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    try:
        from umap import UMAP  # type: ignore
    except Exception as exc:
        raise FeatureMethodError(
            "OPTIONAL_DEPENDENCY_MISSING",
            "umap_embedding requires the optional umap-learn package. Use isomap_embedding, lle_embedding, or tsne_embedding for visualization when UMAP is unavailable.",
            dependency="umap-learn",
        ) from exc
    X = np.asarray(package.X, dtype=float)
    keep = _validate_component_count(n_components, X.shape[1], default=2)
    neighbors = max(2, min(15, len(train_indices) - 1))
    model = UMAP(n_components=keep, n_neighbors=neighbors, random_state=random_state or 42)
    model.fit(X[train_indices])
    transformed = np.asarray(model.transform(X), dtype=float)
    return _embedding_output(
        package,
        transformed,
        method="umap_embedding",
        feature_mode="visualization_embedding",
        method_family="visualization_embedding",
        params={"n_components": keep, "n_neighbors": neighbors, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "transform_available_for_new_samples": True},
        warnings=[_warning("DISCOVERY_EMBEDDING", "UMAP is primarily a discovery visualization; visual separation is not model performance evidence.")],
        transformer=model,
    )


def _apply_isomap_embedding(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep = _validate_component_count(n_components, X.shape[1], default=2)
    neighbors = max(2, min(10, len(train_indices) - 1))
    model = Isomap(n_components=keep, n_neighbors=neighbors)
    model.fit(X[train_indices])
    transformed = np.asarray(model.transform(X), dtype=float)
    return _embedding_output(
        package,
        transformed,
        method="isomap_embedding",
        feature_mode="manifold_embedding",
        method_family="manifold_embedding",
        params={"n_components": keep, "n_neighbors": neighbors},
        fitted={"fit_sample_count": len(train_indices), "transform_available_for_new_samples": True},
        warnings=[_warning("DISCOVERY_EMBEDDING", "Isomap is primarily a discovery visualization; visual separation is not model performance evidence.")],
        transformer=model,
    )


def _apply_lle_embedding(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X = np.asarray(package.X, dtype=float)
    keep = _validate_component_count(n_components, X.shape[1], default=2)
    neighbors = max(2, min(10, len(train_indices) - 1))
    model = LocallyLinearEmbedding(n_components=keep, n_neighbors=neighbors, random_state=random_state or 42)
    model.fit(X[train_indices])
    transformed = np.asarray(model.transform(X), dtype=float)
    return _embedding_output(
        package,
        transformed,
        method="lle_embedding",
        feature_mode="manifold_embedding",
        method_family="manifold_embedding",
        params={"n_components": keep, "n_neighbors": neighbors, "random_state": random_state or 42},
        fitted={"fit_sample_count": len(train_indices), "transform_available_for_new_samples": True},
        warnings=[_warning("DISCOVERY_EMBEDDING", "LLE is primarily a discovery visualization; visual separation is not model performance evidence.")],
        transformer=model,
    )


def _apply_tsne_embedding(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    if sorted(train_indices) != list(range(package.n_samples)):
        raise FeatureMethodError(
            "TSNE_VISUALIZATION_ONLY_REQUIRES_UNSPLIT",
            "tsne_embedding has no stable transform for validation/test samples. Run it only as an explicitly confirmed all-sample discovery visualization for spectral-report, not as a modeling feature package.",
        )
    X = np.asarray(package.X, dtype=float)
    keep = _validate_component_count(n_components, X.shape[1], default=2)
    perplexity = max(2, min(30, (package.n_samples - 1) // 3))
    model = TSNE(n_components=keep, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state or 42)
    transformed = np.asarray(model.fit_transform(X), dtype=float)
    names, axis = _derived_axis("TSNE", transformed.shape[1], "tsne_embedding")
    state = {
        "method": "tsne_embedding",
        "canonical_method": "tsne_embedding",
        "method_family": "visualization_embedding",
        "requires_y": False,
        "task_type": package.contract.get("task_hint"),
        "fit_scope": "all_samples_confirmed",
        "transform_scope": "all_samples",
        "parameters": {"n_components": keep, "perplexity": perplexity, "random_state": random_state or 42},
        "params": {"n_components": keep, "perplexity": perplexity, "random_state": random_state or 42},
        "input_features": {"n_features": package.n_features},
        "output_features": {"n_features": transformed.shape[1], "feature_mode": "visualization_embedding"},
        "fitted": {"fit_sample_count": package.n_samples, "transform_available_for_new_samples": False},
        **_method_semantics("tsne_embedding", "visualization_embedding"),
        "leakage_check": {"split_contract_used": False, "fit_on_train_only": False, "y_used": False, "test_used_in_fit": False},
        "warnings": [
            _warning("VISUALIZATION_ONLY", "t-SNE is for discovery visualization only; do not feed this embedding to model selection or claim performance evidence."),
        ],
    }
    return transformed.tolist(), names, axis, state


def _apply_deep_embedding(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    method: str,
    n_components: int | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    weight_decay: float | None,
    noise_std: float | None,
    mask_ratio: float | None,
    temperature: float | None,
    patch_size: int | None,
    random_state: int | None,
    device: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    try:
        from .deep_embeddings import train_deep_embedding
    except ImportError as exc:
        raise FeatureMethodError(
            "OPTIONAL_DEPENDENCY_MISSING",
            "Deep spectral embedding methods require PyTorch.",
            dependency="torch",
            method=method,
        ) from exc
    values = {
        "embedding_dim": n_components,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-5 if weight_decay is None else weight_decay,
        "noise_std": 0.05 if noise_std is None else noise_std,
        "mask_ratio": 0.15 if mask_ratio is None else mask_ratio,
        "temperature": 0.2 if temperature is None else temperature,
        "patch_size": 16 if patch_size is None else patch_size,
        "random_state": random_state,
        "device": device or "cpu",
    }
    missing = [key for key in ["embedding_dim", "epochs", "batch_size", "learning_rate", "random_state"] if values[key] is None]
    if missing:
        raise FeatureMethodError(
            "DEEP_EMBEDDING_PARAMETERS_REQUIRED",
            "Deep embedding training requires an explicit embedding dimension, epochs, batch size, learning rate, and random seed.",
            method=method,
            missing_parameters=missing,
        )
    try:
        result = train_deep_embedding(
            np.asarray(package.X, dtype=float),
            train_indices,
            method=method,
            embedding_dim=int(values["embedding_dim"]),
            epochs=int(values["epochs"]),
            batch_size=int(values["batch_size"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            noise_std=float(values["noise_std"]),
            mask_ratio=float(values["mask_ratio"]),
            temperature=float(values["temperature"]),
            patch_size=int(values["patch_size"]),
            random_state=int(values["random_state"]),
            device=str(values["device"]),
        )
    except (ValueError, RuntimeError) as exc:
        raise FeatureMethodError(
            "DEEP_EMBEDDING_TRAINING_FAILED",
            str(exc),
            method=method,
        ) from exc
    prefix = {
        "autoencoder_embedding": "AE",
        "denoising_autoencoder_embedding": "DAE",
        "cnn_1d_embedding": "CNN1D",
        "cls_former_embedding": "CLS",
        "resnet1d_embedding": "RESNET1D",
        "masked_spectral_autoencoder_embedding": "MAE",
        "contrastive_spectral_embedding": "CONTRAST",
    }[method]
    names, axis = _derived_axis(prefix, result.transformed.shape[1], method)
    warnings = [
        _warning(
            "DEEP_EMBEDDING_FIXED_EPOCH_TRAINING",
            "Fixed-epoch training completed; numerical convergence is not claimed.",
            epochs=result.training_audit["epochs_completed"],
        )
    ]
    if len(train_indices) < 100:
        warnings.append(
            _warning(
                "DEEP_EMBEDDING_SMALL_SAMPLE_RISK",
                "Deep embedding was trained on fewer than 100 spectra; treat it as experimental and compare against simpler baselines.",
                train_sample_count=len(train_indices),
            )
        )
    state = _state(
        method,
        "deep_self_supervised_embedding",
        str(package.contract.get("task_hint") or "unspecified"),
        result.parameters,
        package.n_features,
        result.transformed.shape[1],
        "modeling_embedding",
        fitted={
            "fit_sample_count": len(train_indices),
            "training_audit": result.training_audit,
            "convergence": result.training_audit,
            "standardization": "train_mean_and_scale",
            "transform_available_for_new_samples": True,
        },
        artifacts={
            "training_trace.csv": {
                "header": ["epoch", "loss"],
                "rows": result.trace_rows,
            }
        },
        warnings=warnings,
    )
    state["deep_training_confirmation"] = {
        "status": "confirmed",
        "protocol": "train_only_self_supervised_embedding",
    }
    state["training_audit"] = result.training_audit
    state["_fitted_transformer"] = result.transformer
    return result.transformed.tolist(), names, axis, state


def _apply_variance_threshold(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    threshold: float | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    cutoff = 0.0 if threshold is None else float(threshold)
    if cutoff < 0 or not math.isfinite(cutoff):
        raise FeatureMethodError("VARIANCE_THRESHOLD_INVALID", "variance_threshold must be a non-negative finite number.", threshold=threshold)
    X = np.asarray(package.X, dtype=float)
    train = X[train_indices, :]
    variances = train.var(axis=0)
    selected = [idx for idx, value in enumerate(variances.tolist()) if value > cutoff]
    _assert_selected(selected, "VARIANCE_THRESHOLD_EMPTY", "variance_threshold selected zero features.")
    return _select_columns(
        package,
        selected,
        method="variance_threshold",
        parameters={"threshold": cutoff},
        fitted={"variances": variances.tolist(), "fit_sample_count": len(train_indices)},
    )


def _apply_band_range(
    package: FeaturePackage,
    *,
    band_min: float | None,
    band_max: float | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    if band_min is None or band_max is None:
        raise FeatureMethodError("BAND_RANGE_REQUIRED", "Please provide band_min and band_max for select_by_band_range.")
    lo = float(min(band_min, band_max))
    hi = float(max(band_min, band_max))
    values = _band_values(package)
    selected = [idx for idx, value in enumerate(values) if lo <= value <= hi]
    _assert_selected(selected, "BAND_RANGE_EMPTY", "select_by_band_range selected zero features.")
    return _select_columns(
        package,
        selected,
        method="select_by_band_range",
        parameters={"band_min": lo, "band_max": hi},
        fitted={"selected_band_values": [values[idx] for idx in selected]},
    )


def _apply_band_indices(
    package: FeaturePackage,
    *,
    band_indices: str | None,
    feature_names: str | None,
    index_base: int,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    selected: list[int] = []
    if band_indices:
        selected.extend(_parse_index_spec(band_indices, n_features=package.n_features, index_base=index_base))
    if feature_names:
        name_to_index = {name: idx for idx, name in enumerate(package.feature_names)}
        for name in [item.strip() for item in feature_names.split(",") if item.strip()]:
            if name not in name_to_index:
                raise FeatureMethodError("FEATURE_NAME_UNKNOWN", "select_by_band_indices references an unknown feature name.", feature_name=name)
            selected.append(name_to_index[name])
    selected = _dedupe_preserve_order(selected)
    _assert_selected(selected, "BAND_INDICES_EMPTY", "select_by_band_indices selected zero features.")
    return _select_columns(
        package,
        selected,
        method="select_by_band_indices",
        parameters={"band_indices": band_indices, "feature_names": feature_names, "index_base": index_base},
        fitted={"selected_band_indices": selected},
    )


def _apply_pls_latent_variables(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=10)
    model = PLSRegression(n_components=keep, scale=True)
    model.fit(X[train_indices], y[train_indices])
    transformed = np.asarray(model.transform(X), dtype=float)
    names = [f"PLS_LV_{idx:03d}" for idx in range(1, transformed.shape[1] + 1)]
    axis = [[idx - 1, name, "pls_latent_variable"] for idx, name in enumerate(names, start=1)]
    artifacts = {
        "components.csv": _matrix_artifact(
            ["component", *package.feature_names],
            [[names[idx], *model.x_loadings_[:, idx].tolist()] for idx in range(model.x_loadings_.shape[1])],
        ),
        "loadings.csv": _matrix_artifact(
            ["feature_index", "feature_name", *names],
            [[idx, package.feature_names[idx], *model.x_loadings_[idx, :].tolist()] for idx in range(package.n_features)],
        ),
    }
    state = _state(
        "pls_latent_variables",
        "supervised_projection",
        selected_task,
        {"n_components": keep},
        package.n_features,
        transformed.shape[1],
        "projection",
        fitted={"fit_sample_count": len(train_indices), "x_rotations": model.x_rotations_.tolist()},
        artifacts=artifacts,
        warnings=warnings,
    )
    state["_fitted_transformer"] = model
    return transformed.tolist(), names, axis, state


def _apply_vip(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    top_k: int | None,
    score_threshold: float | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    keep, warnings = _resolve_n_components(n_components, X[train_indices], default=10)
    model = PLSRegression(n_components=keep, scale=True)
    model.fit(X[train_indices], y[train_indices])
    scores = _vip_scores(model)
    selected, params = _select_from_scores(
        scores,
        package.n_features,
        top_k=top_k,
        score_threshold=1.0 if top_k is None and score_threshold is None else score_threshold,
        default_top_k=None,
    )
    warnings.extend(_selection_warnings(selected, package.n_features, method="vip"))
    return _selection_result(
        package,
        selected,
        scores,
        method="vip",
        family="supervised_selector",
        task_type=selected_task,
        params={"n_components": keep, **params},
        warnings=warnings,
    )


def _apply_correlation_filter(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    top_k: int | None,
    score_threshold: float | None,
    task_type: str | None,
    correlation_method: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    method = (correlation_method or "pearson").strip().lower()
    if method not in {"pearson", "spearman"}:
        raise FeatureMethodError("CORRELATION_METHOD_INVALID", "correlation_method must be pearson or spearman.", correlation_method=method)
    if selected_task == "classification" and y.shape[1] != 1:
        raise FeatureMethodError(
            "CORRELATION_MULTICLASS_UNSUPPORTED",
            "correlation_filter is not recommended for multiclass labels; use anova_f or select_k_best.",
        )
    train_X = X[train_indices]
    target = y[train_indices, 0]
    if method == "spearman":
        train_X = np.apply_along_axis(_rank_values, 0, train_X)
        target = _rank_values(target)
    scores = np.asarray([_safe_abs_corr(train_X[:, idx], target) for idx in range(train_X.shape[1])], dtype=float)
    default_k = min(50, max(5, int(math.ceil(0.05 * package.n_features))))
    selected, params = _select_from_scores(
        scores,
        package.n_features,
        top_k=top_k,
        score_threshold=score_threshold,
        default_top_k=default_k,
    )
    warnings = _selection_warnings(selected, package.n_features, method="correlation_filter")
    return _selection_result(
        package,
        selected,
        scores,
        method="correlation_filter",
        family="supervised_selector",
        task_type=selected_task,
        params={"correlation_method": method, **params},
        warnings=warnings,
    )


def _apply_select_k_best(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    method: str,
    top_k: int | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    if method == "anova_f" and selected_task != "classification":
        raise FeatureMethodError("ANOVA_TASK_MISMATCH", "anova_f requires a classification task.", task_type=selected_task)
    if method == "f_regression" and selected_task != "regression":
        raise FeatureMethodError("F_REGRESSION_TASK_MISMATCH", "f_regression requires a regression task.", task_type=selected_task)
    score_name = "f_classif" if method == "anova_f" or (method == "select_k_best" and selected_task == "classification") else "f_regression"
    default_k = min(50, max(5, int(math.ceil(0.05 * package.n_features))))
    selected_k = _validate_top_k(top_k if top_k is not None else default_k, package.n_features)
    train_y = y[train_indices, 0]
    if score_name == "f_classif":
        scores, _ = f_classif(X[train_indices], train_y.astype(int))
    else:
        scores, _ = f_regression(X[train_indices], train_y)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    selected = _top_indices(scores, selected_k)
    warnings = _selection_warnings(selected, package.n_features, method=method)
    return _selection_result(
        package,
        selected,
        scores,
        method=method,
        family="supervised_selector",
        task_type=selected_task,
        params={"score_func": score_name, "top_k": selected_k},
        warnings=warnings,
    )


def _apply_interval_pls(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    n_intervals: int | None,
    cv: int | None,
    task_type: str | None,
    interval_mode: str | None,
    random_state: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    mode = (interval_mode or "ipls").strip().lower()
    if mode != "ipls":
        raise FeatureMethodError("INTERVAL_PLS_MODE_NOT_IMPLEMENTED", "The first interval_pls version supports mode=ipls.", mode=mode)
    interval_count = n_intervals if n_intervals is not None else (10 if package.n_features < 200 else 20)
    if interval_count < 2 or interval_count > package.n_features:
        raise FeatureMethodError("N_INTERVALS_INVALID", "n_intervals must be between 2 and n_features.", n_intervals=interval_count, n_features=package.n_features)
    folds = _resolve_cv(cv, len(train_indices))
    intervals = [chunk.tolist() for chunk in np.array_split(np.arange(package.n_features), interval_count) if len(chunk)]
    interval_scores: list[float] = []
    rows: list[list[Any]] = []
    for interval_id, selected in enumerate(intervals, start=1):
        components, _ = _resolve_n_components(n_components, X[np.ix_(train_indices, selected)], default=10)
        score = _pls_cv_rmse(
            X[np.ix_(train_indices, selected)],
            y[train_indices],
            n_components=components,
            cv=folds,
            random_state=random_state or 42,
        )
        interval_scores.append(score)
        values = _band_values(package)
        rows.append([interval_id, selected[0], selected[-1], min(values[idx] for idx in selected), max(values[idx] for idx in selected), score, False])
    best_idx = int(np.argmin(interval_scores))
    selected = intervals[best_idx]
    rows[best_idx][-1] = True
    warnings = _selection_warnings(selected, package.n_features, method="interval_pls")
    if len(selected) >= int(0.8 * package.n_features):
        warnings.append(_warning("INTERVAL_PLS_WIDE_SELECTION", "The selected interval covers most of the spectrum."))
    artifacts = {
        "selected_intervals.csv": _matrix_artifact(
            ["interval_id", "start_index", "end_index", "band_min", "band_max", "score", "selected"],
            rows,
        ),
        "interval_scores.csv": _matrix_artifact(
            ["interval_id", "score"],
            [[idx, score] for idx, score in enumerate(interval_scores, start=1)],
        ),
    }
    return _selection_result(
        package,
        selected,
        np.asarray([0.0] * package.n_features),
        method="interval_pls",
        family="interval_selector",
        task_type=selected_task,
        params={"mode": mode, "n_intervals": interval_count, "n_components": n_components, "cv": folds, "random_state": random_state or 42},
        warnings=warnings,
        artifacts=artifacts,
        feature_mode="interval_band_subset",
    )


def _apply_spa(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    top_k: int | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    default_k = min(30, max(5, int(math.ceil(0.03 * package.n_features))))
    selected_k = _validate_top_k(top_k if top_k is not None else default_k, package.n_features)
    train = np.asarray(package.X, dtype=float)[train_indices]
    centered = train - train.mean(axis=0)
    scale = centered.std(axis=0)
    scaled = centered / np.where(scale > 1e-12, scale, 1.0)
    residual = scaled.copy()
    selected: list[int] = []
    scores = np.zeros(package.n_features, dtype=float)
    for order in range(selected_k):
        norms = np.linalg.norm(residual, axis=0)
        if selected:
            norms[selected] = -np.inf
        idx = int(np.argmax(norms))
        if not np.isfinite(norms[idx]) or norms[idx] <= 1e-12:
            break
        selected.append(idx)
        scores[idx] = float(selected_k - order)
        vector = residual[:, idx]
        denominator = float(vector @ vector)
        if denominator > 1e-12:
            residual = residual - np.outer(vector, (vector @ residual) / denominator)
    _assert_selected(selected, "SPA_EMPTY", "SPA selected zero features.")
    warnings = _selection_warnings(selected, package.n_features, method="spa")
    if len(selected) < selected_k:
        warnings.append(_warning("SPA_EARLY_STOP", "SPA stopped early because remaining variables were numerically redundant.", requested=selected_k, selected=len(selected)))
    return _selection_result(
        package,
        selected,
        scores,
        method="spa",
        family="projection_selector",
        task_type="unsupervised",
        params={"top_k": selected_k, "mode": "unsupervised"},
        warnings=warnings,
        selected_order=selected,
    )


def _apply_cars(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    n_components: int | None,
    top_k: int | None,
    n_runs: int | None,
    sample_ratio: float | None,
    cv: int | None,
    random_state: int | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    runs = 50 if n_runs is None else int(n_runs)
    _validate_runs(runs)
    ratio = _validate_sample_ratio(0.8 if sample_ratio is None else sample_ratio)
    folds = _resolve_cv(cv, len(train_indices))
    rng = np.random.default_rng(random_state or 42)
    train_X = X[train_indices]
    train_y = y[train_indices]
    min_variables = _validate_top_k(top_k, package.n_features) if top_k is not None else min(package.n_features, max(2, int(math.ceil(0.05 * package.n_features))))
    active = np.arange(package.n_features)
    best_selected = active.copy()
    best_score = math.inf
    frequency = np.zeros(package.n_features, dtype=float)
    trace_rows: list[list[Any]] = []
    for run in range(runs):
        sample_count = max(3, min(len(train_indices), int(math.ceil(ratio * len(train_indices)))))
        sampled = rng.choice(len(train_indices), size=sample_count, replace=False)
        components, _ = _resolve_n_components(n_components, train_X[sampled][:, active], default=10)
        model = PLSRegression(n_components=components, scale=True)
        model.fit(train_X[sampled][:, active], train_y[sampled])
        coefficients = _coefficient_importance(model, len(active))
        target_count = max(min_variables, int(round(package.n_features * ((min_variables / package.n_features) ** ((run + 1) / runs)))))
        keep_local = np.argsort(coefficients)[::-1][: min(target_count, len(active))]
        active = active[keep_local]
        frequency[active] += 1
        components, _ = _resolve_n_components(n_components, train_X[:, active], default=10)
        score = _pls_cv_rmse(train_X[:, active], train_y, n_components=components, cv=folds, random_state=(random_state or 42) + run)
        trace_rows.append([run + 1, len(active), score, ";".join(str(int(idx)) for idx in active)])
        if score < best_score:
            best_score = score
            best_selected = active.copy()
    selected = sorted(int(idx) for idx in best_selected)
    scores = frequency / runs
    warnings = _selection_warnings(selected, package.n_features, method="cars")
    if runs > 100:
        warnings.append(_warning("CARS_MANY_RUNS", "CARS n_runs is above 100 and may be slow.", n_runs=runs))
    artifacts = {
        "selection_trace.csv": _matrix_artifact(["iteration", "n_variables", "cv_score", "selected_indices"], trace_rows),
    }
    return _selection_result(
        package,
        selected,
        scores,
        method="cars",
        family="monte_carlo_selector",
        task_type=selected_task,
        params={"n_components": n_components, "n_runs": runs, "sample_ratio": ratio, "cv": folds, "random_state": random_state or 42},
        warnings=warnings,
        artifacts=artifacts,
    )


def _apply_uve(
    package: FeaturePackage,
    train_indices: list[int],
    *,
    method: str,
    n_components: int | None,
    top_k: int | None,
    score_threshold: float | None,
    n_runs: int | None,
    sample_ratio: float | None,
    random_state: int | None,
    task_type: str | None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    X, y, selected_task = _supervised_arrays(package, train_indices, task_type)
    default_runs = 100 if method == "mcuve" else 50
    runs = default_runs if n_runs is None else int(n_runs)
    _validate_runs(runs)
    ratio = _validate_sample_ratio(0.8 if sample_ratio is None else sample_ratio)
    rng = np.random.default_rng(random_state or 42)
    train_X = X[train_indices]
    train_y = y[train_indices]
    coefficients = np.zeros((runs, package.n_features), dtype=float)
    noise_coefficients = np.zeros((runs, package.n_features), dtype=float)
    feature_scale = np.std(train_X, axis=0)
    for run in range(runs):
        sample_count = max(3, min(len(train_indices), int(math.ceil(ratio * len(train_indices)))))
        sampled = rng.choice(len(train_indices), size=sample_count, replace=False)
        noise = rng.normal(0.0, np.where(feature_scale > 1e-12, feature_scale, 1.0), size=train_X[sampled].shape)
        augmented = np.hstack([train_X[sampled], noise])
        components, _ = _resolve_n_components(n_components, augmented, default=10)
        model = PLSRegression(n_components=components, scale=True)
        model.fit(augmented, train_y[sampled])
        coef = _signed_coefficient_mean(model, augmented.shape[1])
        coefficients[run] = coef[: package.n_features]
        noise_coefficients[run] = coef[package.n_features :]
    stability = np.abs(coefficients.mean(axis=0)) / (coefficients.std(axis=0, ddof=1) + 1e-12)
    noise_stability = np.abs(noise_coefficients.mean(axis=0)) / (noise_coefficients.std(axis=0, ddof=1) + 1e-12)
    threshold = float(score_threshold) if score_threshold is not None else float(np.nanmax(noise_stability))
    if top_k is not None:
        selected = _top_indices(stability, _validate_top_k(top_k, package.n_features))
        selection_params = {"top_k": top_k, "score_threshold": None}
    else:
        selected = [idx for idx, score in enumerate(stability.tolist()) if score > threshold]
        selection_params = {"top_k": None, "score_threshold": threshold}
    _assert_selected(selected, "UVE_EMPTY", "UVE selected zero features; lower the threshold or provide top_k.")
    warnings = _selection_warnings(selected, package.n_features, method=method)
    frequency = (np.abs(coefficients) > np.median(np.abs(coefficients), axis=1, keepdims=True)).mean(axis=0)
    instability = float(np.mean(frequency * (1.0 - frequency)))
    if instability > 0.2:
        warnings.append(_warning("UVE_SELECTION_UNSTABLE", "Variable stability is low across repeated fits.", instability=instability))
    artifacts = {
        "stability_scores.csv": _matrix_artifact(
            ["feature_index", "feature_name", "stability_score", "selection_frequency", "selected"],
            [[idx, package.feature_names[idx], stability[idx], frequency[idx], idx in set(selected)] for idx in range(package.n_features)],
        ),
        "selection_trace.csv": _matrix_artifact(
            ["iteration", "mean_abs_coefficient"],
            [[run + 1, float(np.mean(np.abs(coefficients[run])))] for run in range(runs)],
        ),
    }
    return _selection_result(
        package,
        selected,
        stability,
        method=method,
        family="monte_carlo_selector",
        task_type=selected_task,
        params={"n_components": n_components, "n_runs": runs, "sample_ratio": ratio, "random_state": random_state or 42, **selection_params},
        warnings=warnings,
        artifacts=artifacts,
    )


def _select_columns(
    package: FeaturePackage,
    selected: list[int],
    *,
    method: str,
    parameters: dict[str, Any],
    fitted: dict[str, Any],
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    selected_X = [[row[idx] for idx in selected] for row in package.X]
    selected_names = [package.feature_names[idx] for idx in selected]
    selected_band_rows = [list(package.band_axis_rows[idx]) for idx in selected]
    fitted = dict(fitted)
    fitted.setdefault("selected_band_indices", selected)
    fitted.setdefault("selected_feature_names", selected_names)
    state = {
        "method": method,
        "parameters": parameters,
        "fitted": fitted,
        "fit_scope": "train_only",
        "transform_scope": "train_val_test",
    }
    return selected_X, selected_names, selected_band_rows, state


def _selection_result(
    package: FeaturePackage,
    selected: list[int],
    scores: np.ndarray,
    *,
    method: str,
    family: str,
    task_type: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
    artifacts: dict[str, Any] | None = None,
    feature_mode: str = "original_band_subset",
    selected_order: list[int] | None = None,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    _assert_selected(selected, "FEATURE_SELECTION_EMPTY", f"{method} selected zero features.")
    selected_order = selected_order or sorted(selected, key=lambda idx: float(scores[idx]), reverse=True)
    ranks = {idx: rank for rank, idx in enumerate(np.argsort(np.asarray(scores))[::-1].tolist(), start=1)}
    values = _optional_band_values(package)
    selected_set = set(selected)
    score_rows = [
        [idx, package.feature_names[idx], values[idx], float(scores[idx]), ranks[idx], idx in selected_set]
        for idx in range(package.n_features)
    ]
    selected_rows = [
        [idx, package.feature_names[idx], values[idx], order]
        for order, idx in enumerate(selected_order, start=1)
        if idx in selected_set
    ]
    merged_artifacts = {
        "selected_features.csv": _matrix_artifact(["feature_index", "feature_name", "band_value", "selected_order"], selected_rows),
        "feature_scores.csv": _matrix_artifact(["feature_index", "feature_name", "band_value", "score", "rank", "selected"], score_rows),
        **(artifacts or {}),
    }
    selected_X = [[row[idx] for idx in selected] for row in package.X]
    selected_names = [package.feature_names[idx] for idx in selected]
    selected_axis = [list(package.band_axis_rows[idx]) for idx in selected]
    state = _state(
        method,
        family,
        task_type,
        params,
        package.n_features,
        len(selected),
        feature_mode,
        fitted={"selected_band_indices": selected, "selected_feature_names": selected_names},
        artifacts=merged_artifacts,
        warnings=warnings,
    )
    return selected_X, selected_names, selected_axis, state


def _state(
    method: str,
    family: str,
    task_type: str,
    params: dict[str, Any],
    input_features: int,
    output_features: int,
    feature_mode: str,
    *,
    fitted: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    semantics = _method_semantics(method, feature_mode)
    return {
        "method": method,
        "canonical_method": method,
        "method_family": family,
        "requires_y": method in SUPERVISED_METHODS,
        "task_type": task_type,
        "fit_scope": "train_only",
        "transform_scope": "train_val_test",
        "parameters": params,
        "params": params,
        "input_features": {"n_features": input_features},
        "output_features": {"n_features": output_features, "feature_mode": feature_mode},
        "fitted": fitted,
        "artifacts": artifacts or {},
        **semantics,
        "leakage_check": {
            "split_contract_used": True,
            "fit_on_train_only": True,
            "y_used": method in SUPERVISED_METHODS,
            "test_used_in_fit": False,
        },
        "warnings": warnings or [],
    }


def _projection_output(
    package: FeaturePackage,
    transformed: np.ndarray,
    *,
    method: str,
    method_family: str,
    feature_mode: str = "projection",
    feature_prefix: str,
    axis_kind: str,
    params: dict[str, Any],
    fitted: dict[str, Any],
    warnings: list[dict[str, Any]],
    transformer: Any,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    names, axis = _derived_axis(feature_prefix, transformed.shape[1], axis_kind)
    state = _state(
        method,
        method_family,
        str(package.contract.get("task_hint") or "unspecified"),
        params,
        package.n_features,
        transformed.shape[1],
        feature_mode,
        fitted=fitted,
        warnings=warnings,
    )
    state["_fitted_transformer"] = transformer
    return transformed.tolist(), names, axis, state


def _embedding_output(
    package: FeaturePackage,
    transformed: np.ndarray,
    *,
    method: str,
    feature_mode: str,
    method_family: str,
    params: dict[str, Any],
    fitted: dict[str, Any],
    warnings: list[dict[str, Any]],
    transformer: Any,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    prefix = method.replace("_embedding", "").upper()
    names, axis = _derived_axis(prefix, transformed.shape[1], method)
    state = _state(
        method,
        method_family,
        str(package.contract.get("task_hint") or "unspecified"),
        params,
        package.n_features,
        transformed.shape[1],
        feature_mode,
        fitted={**fitted, "visualization_only": feature_mode == "visualization_embedding"},
        warnings=warnings,
    )
    state["report_hint"] = {
        "preferred_consumer": "spectral-report",
        "figure_role": "discovery",
        "not_performance_evidence": True,
    }
    state["_fitted_transformer"] = transformer
    return transformed.tolist(), names, axis, state


def _deterministic_signal_state(method: str, params: dict[str, Any], input_features: int, output_features: int) -> dict[str, Any]:
    return {
        "method": method,
        "canonical_method": method,
        "method_family": "deterministic_signal_transform",
        "requires_y": False,
        "task_type": "unspecified",
        "fit_scope": "per_sample_no_fit",
        "transform_scope": "all_samples_independent",
        "parameters": params,
        "params": params,
        "input_features": {"n_features": input_features},
        "output_features": {"n_features": output_features, "feature_mode": "signal_transform_features"},
        "fitted": {"scope": "per_sample_no_fit"},
        **_method_semantics(method, "signal_transform_features"),
        "leakage_check": {
            "split_contract_used": False,
            "fit_on_train_only": True,
            "y_used": False,
            "test_used_in_fit": False,
            "per_sample_deterministic": True,
        },
        "warnings": [],
    }


def _derived_axis(prefix: str, count: int, axis_kind: str) -> tuple[list[str], list[list[Any]]]:
    names = [f"{prefix}_{idx:03d}" for idx in range(1, count + 1)]
    return names, [[idx - 1, name, axis_kind] for idx, name in enumerate(names, start=1)]


def _validate_component_count(n_components: int | None, max_components: int, *, default: int) -> int:
    requested = default if n_components is None else int(n_components)
    if requested < 1:
        raise FeatureMethodError("N_COMPONENTS_INVALID", "n_components must be at least 1.", n_components=requested)
    return min(requested, max(1, int(max_components)))


def _method_semantics(method: str, feature_mode: str) -> dict[str, Any]:
    if method == "tsne_embedding":
        return {
            "intended_use": "visualization",
            "out_of_sample_transform": "unsupported",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": False,
            "modeling_requires_confirmation": True,
        }
    if method == "umap_embedding":
        return {
            "intended_use": "visualization_or_confirmed_modeling",
            "out_of_sample_transform": "supported",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": True,
            "modeling_requires_confirmation": True,
        }
    if feature_mode == "manifold_embedding":
        return {
            "intended_use": "exploratory_or_cautious_modeling",
            "out_of_sample_transform": "limited",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": True,
            "modeling_requires_confirmation": True,
        }
    if feature_mode == "supervised_modeling_embedding":
        return {
            "intended_use": "modeling",
            "out_of_sample_transform": "supported",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": True,
            "supervised_y_used": True,
        }
    if feature_mode == "modeling_embedding":
        return {
            "intended_use": "modeling",
            "out_of_sample_transform": "supported",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": True,
        }
    if feature_mode == "signal_transform_features":
        return {
            "intended_use": "modeling",
            "out_of_sample_transform": "per_sample_deterministic",
            "allowed_for_optimizer_default": False,
            "transform_available_for_new_samples": True,
        }
    return {
        "intended_use": "modeling",
        "out_of_sample_transform": "supported" if feature_mode != "unchanged" else "not_applicable",
        "allowed_for_optimizer_default": method in {"none", "pca", "pls_latent_variables", "vip", "select_k_best", "spa"},
    }


def _fit_with_convergence(
    model: Any,
    X: np.ndarray,
    warnings: list[dict[str, Any]],
    *,
    max_iter: int,
    method: str,
    random_seed: int,
) -> dict[str, Any]:
    with py_warnings.catch_warnings(record=True) as caught:
        py_warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X)
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    payload = _convergence_payload(
        model,
        max_iter=max_iter,
        warning_count=len(convergence_warnings),
        random_seed=random_seed,
    )
    if payload["converged"] is False:
        warnings.append(
            _warning(
                "FEATURE_METHOD_NOT_CONVERGED",
                f"{method} did not clearly converge; inspect feature_state.json before modeling.",
                n_iter=payload.get("n_iter"),
                max_iter=max_iter,
                warning_count=len(convergence_warnings),
            )
        )
    return payload


def _convergence_payload(model: Any, *, max_iter: int, warning_count: int, random_seed: int) -> dict[str, Any]:
    raw_n_iter = getattr(model, "n_iter_", None)
    if isinstance(raw_n_iter, (list, tuple, np.ndarray)):
        n_iter = int(max(raw_n_iter)) if len(raw_n_iter) else None
    elif raw_n_iter is None:
        n_iter = None
    else:
        n_iter = int(raw_n_iter)
    converged = None if n_iter is None else bool(n_iter < max_iter)
    if warning_count:
        converged = False
    return {
        "converged": converged,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "random_seed": random_seed,
        "warning_count": warning_count,
        "warning": "ConvergenceWarning captured during fit" if warning_count else None,
    }


def _supervised_arrays(
    package: FeaturePackage,
    train_indices: list[int],
    task_type: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if package.y_rows is None:
        raise FeatureMethodError("Y_REQUIRED", "This supervised feature method requires y.csv.")
    if any(len(row) != 1 or str(row[0]).strip() == "" for row in package.y_rows):
        raise FeatureMethodError("Y_INVALID", "Supervised feature methods require one complete y column.")
    selected_task = _infer_task_type(package, task_type)
    raw = [str(row[0]).strip() for row in package.y_rows]
    if selected_task == "classification":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(raw)
        classes = len(encoder.classes_)
        if classes < 2:
            raise FeatureMethodError("Y_CLASS_COUNT_INVALID", "Classification feature selection requires at least two classes.")
        y = encoded.astype(float).reshape(-1, 1) if classes == 2 else np.eye(classes, dtype=float)[encoded]
    else:
        try:
            y = np.asarray([float(value) for value in raw], dtype=float).reshape(-1, 1)
        except ValueError as exc:
            raise FeatureMethodError("Y_NON_NUMERIC", "Regression feature selection requires numeric y values.") from exc
    if len(train_indices) < 3:
        raise FeatureMethodError("FEATURE_TRAIN_TOO_SMALL", "Supervised feature methods require at least three train samples.")
    return np.asarray(package.X, dtype=float), y, selected_task


def _infer_task_type(package: FeaturePackage, task_type: str | None) -> str:
    selected = str(task_type or package.contract.get("task_hint") or package.contract.get("task_type") or "").strip().lower()
    if selected in {"classification", "class", "categorical"}:
        return "classification"
    if selected in {"regression", "continuous"}:
        return "regression"
    raise FeatureMethodError("TASK_TYPE_REQUIRED", "Please confirm task_type=classification or regression for this supervised feature method.")


def _resolve_n_components(n_components: int | None, train_X: np.ndarray, *, default: int) -> tuple[int, list[dict[str, Any]]]:
    max_components = max(1, min(train_X.shape[0] - 1, train_X.shape[1]))
    requested = default if n_components is None else int(n_components)
    if requested < 1:
        raise FeatureMethodError("N_COMPONENTS_INVALID", "n_components must be at least 1.", n_components=requested)
    keep = min(requested, max_components)
    warnings = []
    if keep != requested:
        warnings.append(_warning("N_COMPONENTS_CLIPPED", "n_components was clipped to the train-set legal maximum.", requested=requested, used=keep))
    return keep, warnings


def _vip_scores(model: PLSRegression) -> np.ndarray:
    t = np.asarray(model.x_scores_, dtype=float)
    w = np.asarray(model.x_weights_, dtype=float)
    q = np.asarray(model.y_loadings_, dtype=float)
    p = w.shape[0]
    explained = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    total = float(np.sum(explained))
    if total <= 0:
        return np.zeros(p, dtype=float)
    weight_norm = np.sum(w**2, axis=0)
    weight_norm = np.where(weight_norm > 1e-12, weight_norm, 1.0)
    return np.sqrt(p * np.sum((w**2 / weight_norm) * explained, axis=1) / total)


def _coefficient_importance(model: PLSRegression, n_features: int) -> np.ndarray:
    coefficients = np.abs(np.asarray(model.coef_, dtype=float))
    if coefficients.ndim == 1:
        return coefficients
    if coefficients.shape[1] == n_features:
        return coefficients.mean(axis=0)
    if coefficients.shape[0] == n_features:
        return coefficients.mean(axis=1)
    raise FeatureMethodError("PLS_COEFFICIENT_SHAPE_INVALID", "Unexpected PLS coefficient shape.", shape=list(coefficients.shape), n_features=n_features)


def _signed_coefficient_mean(model: PLSRegression, n_features: int) -> np.ndarray:
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        return coefficients
    if coefficients.shape[1] == n_features:
        return coefficients.mean(axis=0)
    if coefficients.shape[0] == n_features:
        return coefficients.mean(axis=1)
    raise FeatureMethodError("PLS_COEFFICIENT_SHAPE_INVALID", "Unexpected PLS coefficient shape.", shape=list(coefficients.shape), n_features=n_features)


def _select_from_scores(
    scores: np.ndarray,
    n_features: int,
    *,
    top_k: int | None,
    score_threshold: float | None,
    default_top_k: int | None,
) -> tuple[list[int], dict[str, Any]]:
    if top_k is not None:
        value = _validate_top_k(top_k, n_features)
        return _top_indices(scores, value), {"top_k": value, "score_threshold": None}
    if score_threshold is not None:
        threshold = float(score_threshold)
        if not math.isfinite(threshold):
            raise FeatureMethodError("SCORE_THRESHOLD_INVALID", "score_threshold must be finite.", score_threshold=threshold)
        selected = [idx for idx, score in enumerate(scores.tolist()) if score >= threshold]
        _assert_selected(selected, "FEATURE_SELECTION_EMPTY", "The score threshold selected zero features.")
        return selected, {"top_k": None, "score_threshold": threshold}
    assert default_top_k is not None
    value = _validate_top_k(default_top_k, n_features)
    return _top_indices(scores, value), {"top_k": value, "score_threshold": None}


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    return sorted(int(idx) for idx in np.argsort(np.asarray(scores, dtype=float))[::-1][:top_k])


def _validate_top_k(top_k: int | None, n_features: int) -> int:
    if top_k is None:
        raise FeatureMethodError("TOP_K_REQUIRED", "top_k is required.")
    value = int(top_k)
    if value < 1 or value > n_features:
        raise FeatureMethodError("TOP_K_INVALID", "top_k must be between 1 and n_features.", top_k=value, n_features=n_features)
    return value


def _resolve_cv(cv: int | None, n_train: int) -> int:
    requested = 5 if cv is None else int(cv)
    if n_train < 4:
        raise FeatureMethodError("CV_TRAIN_TOO_SMALL", "At least four train samples are required for internal CV.")
    return max(2, min(requested, 3 if n_train < 10 else n_train - 1))


def _pls_cv_rmse(X: np.ndarray, y: np.ndarray, *, n_components: int, cv: int, random_state: int) -> float:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    predictions = cross_val_predict(PLSRegression(n_components=n_components, scale=True), X, y, cv=splitter)
    return float(np.sqrt(np.mean((np.asarray(predictions) - y) ** 2)))


def _validate_runs(n_runs: int) -> None:
    if n_runs < 2:
        raise FeatureMethodError("N_RUNS_INVALID", "n_runs must be at least 2.", n_runs=n_runs)
    if n_runs > 500:
        raise FeatureMethodError("N_RUNS_CONFIRMATION_REQUIRED", "n_runs above 500 requires explicit reduction or confirmation.", n_runs=n_runs)


def _validate_sample_ratio(sample_ratio: float) -> float:
    value = float(sample_ratio)
    if value <= 0 or value > 1:
        raise FeatureMethodError("SAMPLE_RATIO_INVALID", "sample_ratio must be in the interval (0, 1].", sample_ratio=value)
    return value


def _selection_warnings(selected: list[int], n_features: int, *, method: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if len(selected) == n_features:
        warnings.append(_warning("FEATURE_SELECTION_ALL", f"{method} retained all input features."))
    elif len(selected) > 0.8 * n_features:
        warnings.append(_warning("FEATURE_SELECTION_DENSE", f"{method} retained more than 80% of input features.", selected=len(selected), n_features=n_features))
    elif len(selected) < 2:
        warnings.append(_warning("FEATURE_SELECTION_SPARSE", f"{method} retained fewer than two features.", selected=len(selected)))
    return warnings


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return abs(value) if math.isfinite(value) else 0.0


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _optional_band_values(package: FeaturePackage) -> list[Any]:
    try:
        return _band_values(package)
    except FeatureMethodError:
        return [None] * package.n_features


def _matrix_artifact(header: list[Any], rows: list[list[Any]]) -> dict[str, Any]:
    return {"header": header, "rows": rows}


def _warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "severity": "warning", "details": details}


def _band_values(package: FeaturePackage) -> list[float]:
    header = [str(item).strip().lower() for item in package.band_axis_header]
    preferred = ["value", "wavelength", "wavenumber", "band", "feature"]
    value_idx = next((header.index(name) for name in preferred if name in header), None)
    if value_idx is None:
        value_idx = 1 if package.band_axis_header and len(package.band_axis_header) > 1 else 0
    values: list[float] = []
    for row_idx, row in enumerate(package.band_axis_rows):
        if value_idx >= len(row):
            raise FeatureMethodError("BAND_AXIS_VALUE_MISSING", "band_axis row lacks a numeric band value.", row=row_idx)
        try:
            values.append(float(str(row[value_idx]).strip()))
        except ValueError as exc:
            raise FeatureMethodError("BAND_AXIS_VALUE_NON_NUMERIC", "select_by_band_range requires numeric band_axis values.", value=row[value_idx], row=row_idx) from exc
    return values


def _parse_index_spec(spec: str, *, n_features: int, index_base: int) -> list[int]:
    if index_base not in {0, 1}:
        raise FeatureMethodError("INDEX_BASE_INVALID", "index_base must be 0 or 1.", index_base=index_base)
    selected: list[int] = []
    for raw in [item.strip() for item in spec.split(",") if item.strip()]:
        if ":" in raw or "-" in raw:
            sep = ":" if ":" in raw else "-"
            start_text, end_text = [part.strip() for part in raw.split(sep, 1)]
            start = int(start_text) - index_base
            end = int(end_text) - index_base
            step = 1 if end >= start else -1
            selected.extend(range(start, end + step, step))
        else:
            selected.append(int(raw) - index_base)
    for idx in selected:
        if idx < 0 or idx >= n_features:
            raise FeatureMethodError("BAND_INDEX_OUT_OF_RANGE", "select_by_band_indices references an out-of-range feature index.", index=idx, n_features=n_features, index_base=index_base)
    return selected


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _assert_selected(selected: list[int], code: str, message: str) -> None:
    if not selected:
        raise FeatureMethodError(code, message)

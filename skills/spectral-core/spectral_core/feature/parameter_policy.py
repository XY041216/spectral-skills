"""Shared critical-parameter policy for spectral feature methods."""

from __future__ import annotations

from typing import Any


ALIASES = {
    "pls_lv": "pls_latent_variables",
    "pls_scores": "pls_latent_variables",
    "kpca": "kernel_pca",
    "kernelpca": "kernel_pca",
    "sparsepca": "sparse_pca",
    "fastica": "ica_embedding",
    "ica": "ica_embedding",
    "lda": "lda_projection",
    "dct": "dct_features",
    "fft": "fft_features",
    "dictionary": "dictionary_learning",
    "dict_learning": "dictionary_learning",
    "umap": "umap_embedding",
    "tsne": "tsne_embedding",
    "t_sne": "tsne_embedding",
    "isomap": "isomap_embedding",
    "lle": "lle_embedding",
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
    "pls_vip": "vip",
    "variable_importance_projection": "vip",
    "corr": "correlation_filter",
    "pearson_corr": "correlation_filter",
    "spearman_corr": "correlation_filter",
    "kbest": "select_k_best",
    "skb": "select_k_best",
    "anova": "anova_f",
    "f_classif": "anova_f",
    "ipls": "interval_pls",
    "bipls": "interval_pls",
    "sipls": "interval_pls",
    "moving_window_pls": "interval_pls",
    "cars_pls": "cars",
    "uve_pls": "uve",
    "mc_uve": "mcuve",
}


CRITICAL_PARAMETERS = {
    "pls_latent_variables": ("n_components",),
    "kernel_pca": ("n_components",),
    "sparse_pca": ("n_components",),
    "nmf": ("n_components",),
    "ica_embedding": ("n_components",),
    "lda_projection": ("n_components",),
    "dct_features": ("n_components",),
    "fft_features": ("n_components",),
    "dictionary_learning": ("n_components",),
    "umap_embedding": ("n_components",),
    "isomap_embedding": ("n_components",),
    "lle_embedding": ("n_components",),
    "tsne_embedding": ("n_components",),
    "select_k_best": ("top_k",),
    "anova_f": ("top_k",),
    "f_regression": ("top_k",),
    "interval_pls": ("n_intervals", "n_components", "cv"),
    "spa": ("top_k",),
    "cars": ("n_components", "n_runs", "sample_ratio", "cv", "random_state"),
    "uve": ("n_components", "n_runs", "selection_rule", "random_state"),
    "mcuve": ("n_components", "n_runs", "selection_rule", "random_state"),
    "vip": ("selection_rule", "n_components"),
    "correlation_filter": ("selection_rule",),
    "autoencoder_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "random_state"),
    "denoising_autoencoder_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "noise_std", "random_state"),
    "cnn_1d_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "random_state"),
    "cls_former_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "patch_size", "random_state"),
    "resnet1d_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "random_state"),
    "masked_spectral_autoencoder_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "mask_ratio", "patch_size", "random_state"),
    "contrastive_spectral_embedding": ("n_components", "epochs", "batch_size", "learning_rate", "noise_std", "mask_ratio", "temperature", "random_state"),
}


def normalize_feature_method(method: str | None) -> str:
    normalized = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(normalized, normalized)


def missing_critical_parameters(method: str | None, values: dict[str, Any]) -> list[str]:
    normalized = normalize_feature_method(method)
    missing: list[str] = []
    for field in CRITICAL_PARAMETERS.get(normalized, ()):
        if field == "selection_rule":
            if values.get("top_k") is None and values.get("score_threshold") is None:
                missing.append(field)
        elif values.get(field) is None:
            missing.append(field)
    return missing

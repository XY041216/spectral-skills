"""Deployable spectral pipeline artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpectralPipelineArtifact:
    """Minimal callable artifact for raw spectra -> model predictions."""

    model: Any
    model_type: str | None
    model_parameters: dict[str, Any] | None
    task_type: str | None
    class_labels: list[str] | None
    input_schema: dict[str, Any]
    pipeline_steps: dict[str, Any]
    contracts: dict[str, Any]
    preprocess_methods: list[str]
    feature_transformer: Any | None = None

    def transform_raw(self, X: Any) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        expected = self.input_schema.get("raw_n_features") or self.input_schema.get("n_features")
        if expected is not None and data.shape[1] != int(expected):
            raise ValueError(f"Expected {expected} raw features, got {data.shape[1]}.")
        transformed = self._apply_preprocess(data)
        transformed = self._apply_feature(transformed)
        return transformed

    def predict_raw(self, X: Any) -> np.ndarray:
        predictions = self.model.predict(self.transform_raw(X))
        if self.task_type == "classification" and self.class_labels:
            labels = list(self.class_labels)
            mapped = []
            for value in predictions:
                try:
                    mapped.append(labels[int(value)])
                except (ValueError, TypeError, IndexError):
                    mapped.append(str(value))
            return np.asarray(mapped, dtype=object)
        return predictions

    def predict_proba_raw(self, X: Any) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The fitted model does not expose predict_proba.")
        return self.model.predict_proba(self.transform_raw(X))

    def _apply_preprocess(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        for method in self.preprocess_methods:
            canonical = str(method).strip().lower()
            if canonical in {"", "none", "skip"}:
                continue
            if canonical == "snv":
                mean = data.mean(axis=1, keepdims=True)
                std = data.std(axis=1, ddof=1, keepdims=True)
                std = np.where(std > 1e-12, std, 1.0)
                data = (data - mean) / std
                continue
            raise NotImplementedError(f"Raw prediction does not yet support preprocess method '{method}'.")
        return data

    def _apply_feature(self, X: np.ndarray) -> np.ndarray:
        transformer = self.feature_transformer
        if transformer is None:
            return X
        if isinstance(transformer, dict) and transformer.get("method") == "pca":
            mean = np.asarray(transformer["mean"], dtype=float)
            components = np.asarray(transformer["components"], dtype=float)
            return (X - mean) @ components.T
        if hasattr(transformer, "transform"):
            return np.asarray(transformer.transform(X), dtype=float)
        raise NotImplementedError("Raw prediction does not support the stored feature transformer.")

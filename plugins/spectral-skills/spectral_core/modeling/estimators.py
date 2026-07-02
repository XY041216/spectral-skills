"""Additional sklearn-compatible chemometric estimators."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS regression on one-hot class targets with discriminant decoding."""

    def __init__(self, n_components: int = 2, scale: bool = True, max_iter: int = 500):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter

    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[1] == 1:
            Y = np.column_stack([1.0 - Y[:, 0], Y[:, 0]])
        self.classes_ = self.label_binarizer_.classes_
        n_components = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))
        self.model_ = PLSRegression(n_components=n_components, scale=self.scale, max_iter=self.max_iter)
        self.model_.fit(X, Y)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        scores = np.asarray(self.model_.predict(X), dtype=float)
        return scores if scores.ndim == 2 else scores.reshape(-1, 1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        probability = np.exp(scores)
        return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X):
        indices = np.argmax(self.predict_proba(X), axis=1)
        return np.asarray(self.classes_)[indices]


class SIMCAClassifier(BaseEstimator, ClassifierMixin):
    """Soft independent modeling of class analogy using per-class PCA distance."""

    def __init__(self, n_components: int = 3, quantile: float = 0.95):
        self.n_components = n_components
        self.quantile = quantile

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_models_ = {}
        for label in self.classes_:
            block = X[y == label]
            center = block.mean(axis=0)
            centered = block - center
            max_components = max(1, min(int(self.n_components), centered.shape[1], max(1, centered.shape[0] - 1)))
            pca = PCA(n_components=max_components)
            scores = pca.fit_transform(centered)
            reconstructed = pca.inverse_transform(scores)
            residual = np.sum((centered - reconstructed) ** 2, axis=1)
            score_scale = np.maximum(np.var(scores, axis=0, ddof=1), 1e-12)
            score_distance = np.sum((scores**2) / score_scale, axis=1)
            residual_scale = max(float(np.median(residual)), 1e-12)
            score_distance_scale = max(float(np.median(score_distance)), 1e-12)
            combined = residual / residual_scale + score_distance / score_distance_scale
            self.class_models_[label] = {
                "center": center,
                "pca": pca,
                "score_scale": score_scale,
                "residual_scale": residual_scale,
                "score_distance_scale": score_distance_scale,
                "threshold": float(np.quantile(combined, self.quantile)),
            }
        return self

    def decision_function(self, X):
        check_is_fitted(self, "class_models_")
        X = np.asarray(X, dtype=float)
        distances = []
        for label in self.classes_:
            item = self.class_models_[label]
            centered = X - item["center"]
            scores = item["pca"].transform(centered)
            reconstructed = item["pca"].inverse_transform(scores)
            residual = np.sum((centered - reconstructed) ** 2, axis=1)
            score_distance = np.sum((scores**2) / item["score_scale"], axis=1)
            distance = residual / item["residual_scale"] + score_distance / item["score_distance_scale"]
            distances.append(distance / max(item["threshold"], 1e-12))
        return -np.column_stack(distances)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        probability = np.exp(scores)
        return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X):
        indices = np.argmax(self.decision_function(X), axis=1)
        return np.asarray(self.classes_)[indices]

# spectral_core/modeling/experimental_small_sample.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 通用工具
# ============================================================

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, task_type: str):
        self.x = torch.FloatTensor(x)
        self.task_type = task_type
        if task_type == "classification":
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    task_type: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ArrayDataset(x, y, task_type),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# ============================================================
# 1. 损失函数
# ============================================================

class RobustSupConLoss(nn.Module):
    """
    监督对比损失。
    分类任务：labels 为类别标签。
    与 CLS-Former 文档中的 RobustSupConLoss 思路一致：
    L2 normalize -> cosine similarity -> temperature scaling -> mask positives.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        n = features.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = F.normalize(features, dim=1, eps=self.eps)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = torch.clamp(sim, -50.0, 50.0)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0.0)

        positive_mask = mask * logits_mask
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        pos_count = positive_mask.sum(1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (pos_count + self.eps)
        loss = -mean_log_prob_pos[valid].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss


class ContinuousContrastiveLoss(nn.Module):
    """
    回归任务用的软监督对比损失。
    y 越接近，样本对权重越大。
    """

    def __init__(self, temperature: float = 0.1, y_temperature: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.y_temperature = y_temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = features.device
        n = features.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = F.normalize(features, dim=1, eps=self.eps)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = torch.clamp(sim, -50.0, 50.0)

        y = y.view(-1, 1)
        y_dist = torch.abs(y - y.T)
        soft_pos = torch.exp(-y_dist / max(self.y_temperature, self.eps)).to(device)

        logits_mask = torch.ones_like(soft_pos)
        logits_mask.fill_diagonal_(0.0)
        soft_pos = soft_pos * logits_mask

        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        weight_sum = soft_pos.sum(1)
        valid = weight_sum > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        weighted_log_prob = (soft_pos * log_prob).sum(1) / (weight_sum + self.eps)
        loss = -weighted_log_prob[valid].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss


# ============================================================
# 2. 通用 Encoder
# ============================================================

class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1DSpectralEncoder(nn.Module):
    """
    简单 1D-CNN 光谱 encoder。
    输入仍是 tabular spectrum: [batch, n_bands]
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, embedding_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.conv(x))


def build_encoder(
    encoder_type: str,
    input_dim: int,
    embedding_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> nn.Module:
    if encoder_type == "mlp":
        return MLPEncoder(input_dim, embedding_dim, hidden_dim, dropout)
    if encoder_type == "1d_cnn":
        return Conv1DSpectralEncoder(input_dim, embedding_dim, hidden_dim // 4, dropout)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


def extract_embeddings(
    encoder: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    encoder.eval()
    xs = torch.FloatTensor(x)
    loader = DataLoader(xs, batch_size=batch_size, shuffle=False)
    blocks = []
    with torch.no_grad():
        for xb in loader:
            emb = encoder(xb.to(device)).detach().cpu().numpy()
            blocks.append(emb)
    return np.vstack(blocks)


# ============================================================
# 3. 训练 Encoder 的通用函数
# ============================================================

def train_classifier_encoder(
    encoder: nn.Module,
    classifier_head: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    *,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    alpha: float = 0.5,
    temperature: float = 0.1,
    patience: int = 20,
    random_state: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    set_global_seed(random_state)
    dev = resolve_device(device)
    encoder.to(dev)
    classifier_head.to(dev)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    ce = nn.CrossEntropyLoss()
    supcon = RobustSupConLoss(temperature=temperature)

    train_loader = make_loader(x_train, y_train, "classification", batch_size, True)

    best_state = None
    best_val = -np.inf
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier_head.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            opt.zero_grad()
            emb = encoder(xb)
            logits = classifier_head(emb)
            loss = alpha * supcon(emb, yb) + (1.0 - alpha) * ce(logits, yb)
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier_head.parameters()),
                    max_norm=1.0,
                )
                opt.step()
                total_loss += loss.item()

        metric = None
        if x_val is not None and y_val is not None and len(x_val) > 0:
            encoder.eval()
            classifier_head.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(x_val).to(dev)
                val_logits = classifier_head(encoder(val_x))
                val_pred = val_logits.argmax(1).detach().cpu().numpy()
            metric = accuracy_score(y_val, val_pred)

            if metric > best_val:
                best_val = metric
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in classifier_head.state_dict().items()},
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_accuracy": metric,
        })

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        classifier_head.load_state_dict(best_state["head"])

    return {
        "device": str(dev),
        "history": history,
        "best_val_accuracy": None if best_val == -np.inf else float(best_val),
        "epochs_ran": len(history),
    }


def train_regression_encoder(
    encoder: nn.Module,
    reg_head: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    *,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    alpha: float = 0.3,
    temperature: float = 0.1,
    y_temperature: float = 1.0,
    patience: int = 20,
    random_state: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    set_global_seed(random_state)
    dev = resolve_device(device)
    encoder.to(dev)
    reg_head.to(dev)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(reg_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    huber = nn.HuberLoss()
    ccon = ContinuousContrastiveLoss(temperature=temperature, y_temperature=y_temperature)

    train_loader = make_loader(x_train, y_train, "regression", batch_size, True)

    best_state = None
    best_val = np.inf
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        reg_head.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            opt.zero_grad()
            emb = encoder(xb)
            pred = reg_head(emb)
            loss = alpha * ccon(emb, yb) + (1.0 - alpha) * huber(pred, yb)
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(reg_head.parameters()),
                    max_norm=1.0,
                )
                opt.step()
                total_loss += loss.item()

        metric = None
        if x_val is not None and y_val is not None and len(x_val) > 0:
            encoder.eval()
            reg_head.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(x_val).to(dev)
                val_pred = reg_head(encoder(val_x)).detach().cpu().numpy().ravel()
            metric = float(np.sqrt(np.mean((val_pred - y_val.ravel()) ** 2)))

            if metric < best_val:
                best_val = metric
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in reg_head.state_dict().items()},
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_rmse": metric,
        })

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        reg_head.load_state_dict(best_state["head"])

    return {
        "device": str(dev),
        "history": history,
        "best_val_rmse": None if best_val == np.inf else float(best_val),
        "epochs_ran": len(history),
    }


# ============================================================
# 4. Spectral-DKL-GP
# 轻量实现：Torch encoder + sklearn GP head
# ============================================================

class SpectralDKLClassifier(BaseEstimator, ClassifierMixin):
    """
    小样本分类：
    StandardScaler/PCA -> neural embedding -> GaussianProcessClassifier.
    注意：这不是 gpytorch 的端到端 DKL，而是低依赖的 DKL-style 实现。
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        preprojection: str = "pca",
        n_components: int = 50,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.5,
        temperature: float = 0.1,
        patience: int = 20,
        kernel: str = "rbf",
        random_state: int = 42,
        device: str = "auto",
    ):
        self.encoder_type = encoder_type
        self.preprojection = preprojection
        self.n_components = n_components
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = temperature
        self.patience = patience
        self.kernel = kernel
        self.random_state = random_state
        self.device = device

    def _prepare_x_fit(self, x: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler()
        x2 = self.scaler_.fit_transform(x)

        self.projector_ = None
        if self.preprojection == "pca":
            n_comp = min(self.n_components, x2.shape[0] - 1, x2.shape[1])
            n_comp = max(1, n_comp)
            self.projector_ = PCA(n_components=n_comp, random_state=self.random_state)
            x2 = self.projector_.fit_transform(x2)
        elif self.preprojection in ("none", None):
            pass
        else:
            raise ValueError(f"Unsupported preprojection: {self.preprojection}")
        return as_float32(x2)

    def _prepare_x_transform(self, x: np.ndarray) -> np.ndarray:
        x2 = self.scaler_.transform(x)
        if self.projector_ is not None:
            x2 = self.projector_.transform(x2)
        return as_float32(x2)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        set_global_seed(self.random_state)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        x_train = self._prepare_x_fit(x)
        x_valid = self._prepare_x_transform(x_val) if x_val is not None else None
        y_valid = self.label_encoder_.transform(y_val) if y_val is not None else None

        input_dim = x_train.shape[1]
        n_classes = len(self.classes_)

        self.encoder_ = build_encoder(
            self.encoder_type,
            input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.classifier_head_ = nn.Linear(self.embedding_dim, n_classes)

        self.training_info_ = train_classifier_encoder(
            self.encoder_,
            self.classifier_head_,
            x_train,
            y_enc,
            x_valid,
            y_valid,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            temperature=self.temperature,
            patience=self.patience,
            random_state=self.random_state,
            device=self.device,
        )

        dev = resolve_device(self.device)
        z_train = extract_embeddings(self.encoder_, x_train, dev)

        if self.kernel == "rbf":
            gp_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif self.kernel == "matern":
            gp_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        self.gp_ = GaussianProcessClassifier(
            kernel=gp_kernel,
            random_state=self.random_state,
            n_restarts_optimizer=0,
        )
        self.gp_.fit(z_train, y_enc)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        x2 = self._prepare_x_transform(x)
        dev = resolve_device(self.device)
        return extract_embeddings(self.encoder_, x2, dev)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        pred_enc = self.gp_.predict(z)
        return self.label_encoder_.inverse_transform(pred_enc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        return self.gp_.predict_proba(z)


class SpectralDKLRegressor(BaseEstimator, RegressorMixin):
    """
    小样本回归：
    StandardScaler/PCA -> neural embedding -> GaussianProcessRegressor.
    可输出不确定性。
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        preprojection: str = "pca",
        n_components: int = 50,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.3,
        temperature: float = 0.1,
        y_temperature: float = 1.0,
        patience: int = 20,
        kernel: str = "rbf",
        random_state: int = 42,
        device: str = "auto",
    ):
        self.encoder_type = encoder_type
        self.preprojection = preprojection
        self.n_components = n_components
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = temperature
        self.y_temperature = y_temperature
        self.patience = patience
        self.kernel = kernel
        self.random_state = random_state
        self.device = device

    def _prepare_x_fit(self, x: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler()
        x2 = self.scaler_.fit_transform(x)

        self.projector_ = None
        if self.preprojection == "pca":
            n_comp = min(self.n_components, x2.shape[0] - 1, x2.shape[1])
            n_comp = max(1, n_comp)
            self.projector_ = PCA(n_components=n_comp, random_state=self.random_state)
            x2 = self.projector_.fit_transform(x2)
        elif self.preprojection in ("none", None):
            pass
        else:
            raise ValueError(f"Unsupported preprojection: {self.preprojection}")
        return as_float32(x2)

    def _prepare_x_transform(self, x: np.ndarray) -> np.ndarray:
        x2 = self.scaler_.transform(x)
        if self.projector_ is not None:
            x2 = self.projector_.transform(x2)
        return as_float32(x2)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        set_global_seed(self.random_state)

        y = np.asarray(y, dtype=float).ravel()
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()

        x_train = self._prepare_x_fit(x)
        x_valid = self._prepare_x_transform(x_val) if x_val is not None else None
        y_valid = None
        if y_val is not None:
            y_valid = self.y_scaler_.transform(np.asarray(y_val).reshape(-1, 1)).ravel()

        input_dim = x_train.shape[1]
        self.encoder_ = build_encoder(
            self.encoder_type,
            input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.reg_head_ = nn.Linear(self.embedding_dim, 1)

        self.training_info_ = train_regression_encoder(
            self.encoder_,
            self.reg_head_,
            x_train,
            y_scaled,
            x_valid,
            y_valid,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            temperature=self.temperature,
            y_temperature=self.y_temperature,
            patience=self.patience,
            random_state=self.random_state,
            device=self.device,
        )

        dev = resolve_device(self.device)
        z_train = extract_embeddings(self.encoder_, x_train, dev)

        if self.kernel == "rbf":
            gp_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        elif self.kernel == "matern":
            gp_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        self.gp_ = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=1e-6,
            normalize_y=False,
            random_state=self.random_state,
            n_restarts_optimizer=0,
        )
        self.gp_.fit(z_train, y_scaled)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        x2 = self._prepare_x_transform(x)
        dev = resolve_device(self.device)
        return extract_embeddings(self.encoder_, x2, dev)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        pred_scaled = self.gp_.predict(z)
        return self.y_scaler_.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    def predict_with_std(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.transform(x)
        mean_scaled, std_scaled = self.gp_.predict(z, return_std=True)
        mean = self.y_scaler_.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std = std_scaled * float(self.y_scaler_.scale_[0])
        return mean, std


# ============================================================
# 5. Proto-Spectral
# ============================================================

class ProtoSpectralClassifier(BaseEstimator, ClassifierMixin):
    """
    Prototype 小样本分类。
    Encoder 训练后，计算每类 prototype，预测时按最近 prototype 分类。
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        metric: str = "euclidean",
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.7,
        temperature: float = 0.1,
        patience: int = 20,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = temperature
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        set_global_seed(self.random_state)

        self.scaler_ = StandardScaler()
        x_train = as_float32(self.scaler_.fit_transform(x))
        x_valid = as_float32(self.scaler_.transform(x_val)) if x_val is not None else None

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        y_valid = self.label_encoder_.transform(y_val) if y_val is not None else None
        self.classes_ = self.label_encoder_.classes_

        input_dim = x_train.shape[1]
        n_classes = len(self.classes_)

        self.encoder_ = build_encoder(
            self.encoder_type,
            input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.classifier_head_ = nn.Linear(self.embedding_dim, n_classes)

        self.training_info_ = train_classifier_encoder(
            self.encoder_,
            self.classifier_head_,
            x_train,
            y_enc,
            x_valid,
            y_valid,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            temperature=self.temperature,
            patience=self.patience,
            random_state=self.random_state,
            device=self.device,
        )

        dev = resolve_device(self.device)
        z_train = extract_embeddings(self.encoder_, x_train, dev)
        self.prototypes_ = []
        for c in range(n_classes):
            self.prototypes_.append(z_train[y_enc == c].mean(axis=0))
        self.prototypes_ = np.vstack(self.prototypes_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        x2 = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        return extract_embeddings(self.encoder_, x2, dev)

    def _distances(self, z: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return ((z[:, None, :] - self.prototypes_[None, :, :]) ** 2).sum(axis=2)
        if self.metric == "cosine":
            z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            p_norm = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-12)
            return 1.0 - np.matmul(z_norm, p_norm.T)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        d = self._distances(z)
        pred_enc = np.argmin(d, axis=1)
        return self.label_encoder_.inverse_transform(pred_enc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        d = self._distances(z)
        logits = -d
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        return p


class ProtoSpectralRegressor(BaseEstimator, RegressorMixin):
    """
    Prototype-style 小样本回归。
    训练回归 encoder，然后在 embedding 空间构建 y 分箱 prototype。
    预测时用距离权重对 prototype 的 y 均值加权。
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        n_prototypes: int = 5,
        metric: str = "euclidean",
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.3,
        temperature: float = 0.1,
        y_temperature: float = 1.0,
        patience: int = 20,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_prototypes = n_prototypes
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = temperature
        self.y_temperature = y_temperature
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        set_global_seed(self.random_state)

        y = np.asarray(y, dtype=float).ravel()

        self.scaler_ = StandardScaler()
        x_train = as_float32(self.scaler_.fit_transform(x))
        x_valid = as_float32(self.scaler_.transform(x_val)) if x_val is not None else None

        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        y_valid_scaled = None
        if y_val is not None:
            y_valid_scaled = self.y_scaler_.transform(np.asarray(y_val).reshape(-1, 1)).ravel()

        input_dim = x_train.shape[1]
        self.encoder_ = build_encoder(
            self.encoder_type,
            input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.reg_head_ = nn.Linear(self.embedding_dim, 1)

        self.training_info_ = train_regression_encoder(
            self.encoder_,
            self.reg_head_,
            x_train,
            y_scaled,
            x_valid,
            y_valid_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            temperature=self.temperature,
            y_temperature=self.y_temperature,
            patience=self.patience,
            random_state=self.random_state,
            device=self.device,
        )

        dev = resolve_device(self.device)
        z_train = extract_embeddings(self.encoder_, x_train, dev)

        # 按 y 分位数构造 prototype
        n_proto = min(self.n_prototypes, max(2, len(y) // 2))
        quantiles = np.linspace(0, 1, n_proto + 1)
        edges = np.quantile(y, quantiles)
        edges = np.unique(edges)

        proto_z = []
        proto_y = []
        for i in range(len(edges) - 1):
            if i == len(edges) - 2:
                mask = (y >= edges[i]) & (y <= edges[i + 1])
            else:
                mask = (y >= edges[i]) & (y < edges[i + 1])
            if mask.sum() == 0:
                continue
            proto_z.append(z_train[mask].mean(axis=0))
            proto_y.append(y[mask].mean())

        if len(proto_z) == 0:
            proto_z = [z_train.mean(axis=0)]
            proto_y = [y.mean()]

        self.prototypes_ = np.vstack(proto_z)
        self.prototype_y_ = np.asarray(proto_y, dtype=float)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        x2 = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        return extract_embeddings(self.encoder_, x2, dev)

    def _distances(self, z: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return ((z[:, None, :] - self.prototypes_[None, :, :]) ** 2).sum(axis=2)
        if self.metric == "cosine":
            z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            p_norm = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-12)
            return 1.0 - np.matmul(z_norm, p_norm.T)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        d = self._distances(z)
        w = np.exp(-d)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        return np.matmul(w, self.prototype_y_)


# ============================================================
# 6. CLS-Former
# ============================================================

class CLSFormerNetwork(nn.Module):
    """
    Contract-aware CLS-Former 主干。
    保留原始文档中的核心：
    input_projection + learnable position + TransformerEncoder
    + deep feature layer + projection head + classifier/regression head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.task_type = task_type

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.feature_linear1 = nn.Linear(d_model, dim_feedforward)
        self.feature_act1 = nn.ReLU()
        self.feature_drop1 = nn.Dropout(dropout)
        self.feature_linear2 = nn.Linear(dim_feedforward, feature_dim)

        self.proj_linear1 = nn.Linear(feature_dim, feature_dim)
        self.proj_act1 = nn.ReLU()
        self.proj_linear2 = nn.Linear(feature_dim, feature_dim)

        self.output_head = nn.Linear(feature_dim, output_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        h = self.feature_linear1(x)
        h = self.feature_act1(h)
        h = self.feature_drop1(h)
        deep = self.feature_linear2(h)

        if return_features:
            return deep

        proj = self.proj_linear1(deep)
        proj = self.proj_act1(proj)
        proj = self.proj_linear2(proj)
        proj = F.normalize(proj, dim=1, eps=1e-8)

        out = self.output_head(deep)
        return out, proj, deep


class CLSFormerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        feature_dim: int = 32,
        epochs: int = 100,
        batch_size: int = 8,
        alpha: float = 0.5,
        temperature: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        patience: int = 20,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        set_global_seed(self.random_state)

        self.scaler_ = StandardScaler()
        x_train = as_float32(self.scaler_.fit_transform(x))
        x_valid = as_float32(self.scaler_.transform(x_val)) if x_val is not None else None

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        y_valid = self.label_encoder_.transform(y_val) if y_val is not None else None
        self.classes_ = self.label_encoder_.classes_

        self.model_ = CLSFormerNetwork(
            input_dim=x_train.shape[1],
            output_dim=len(self.classes_),
            task_type="classification",
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            feature_dim=self.feature_dim,
        )

        self.training_info_ = self._train_classifier(
            x_train, y_enc, x_valid, y_valid
        )
        return self

    def _train_classifier(self, x_train, y_train, x_val, y_val):
        dev = resolve_device(self.device)
        self.model_.to(dev)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss()
        supcon = RobustSupConLoss(temperature=self.temperature)

        train_loader = make_loader(x_train, y_train, "classification", self.batch_size, True)

        best_state = None
        best_val = -np.inf
        no_improve = 0
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)

                opt.zero_grad()
                logits, proj, _ = self.model_(xb)
                loss = self.alpha * supcon(proj, yb) + (1.0 - self.alpha) * ce(logits, yb)

                if not torch.isnan(loss) and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                    opt.step()
                    total_loss += loss.item()

            val_acc = None
            if x_val is not None and y_val is not None and len(x_val) > 0:
                pred = self._predict_encoded(x_val, dev)
                val_acc = accuracy_score(y_val, pred)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break

            history.append({
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "val_accuracy": val_acc,
            })

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return {
            "device": str(dev),
            "history": history,
            "best_val_accuracy": None if best_val == -np.inf else float(best_val),
            "epochs_ran": len(history),
        }

    def _predict_encoded(self, x_scaled: np.ndarray, dev: torch.device) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits, _, _ = self.model_(torch.FloatTensor(x_scaled).to(dev))
            return logits.argmax(1).detach().cpu().numpy()

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        x_scaled = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        self.model_.to(dev)
        self.model_.eval()

        loader = DataLoader(torch.FloatTensor(x_scaled), batch_size=128, shuffle=False)
        blocks = []
        with torch.no_grad():
            for xb in loader:
                feat = self.model_(xb.to(dev), return_features=True)
                blocks.append(feat.detach().cpu().numpy())
        return np.vstack(blocks)

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        x_scaled = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        self.model_.to(dev)
        pred_enc = self._predict_encoded(x_scaled, dev)
        return self.label_encoder_.inverse_transform(pred_enc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        x_scaled = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        self.model_.to(dev)
        self.model_.eval()
        with torch.no_grad():
            logits, _, _ = self.model_(torch.FloatTensor(x_scaled).to(dev))
            p = F.softmax(logits, dim=1).detach().cpu().numpy()
        return p


class CLSFormerEmbeddingSVM(BaseEstimator, ClassifierMixin):
    """
    CLS-Former 训练 embedding，然后用 SVM 分类。
    """

    def __init__(
        self,
        svm_C: float = 1.0,
        svm_gamma: str = "scale",
        **cls_former_params,
    ):
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.cls_former_params = cls_former_params

    def fit(self, x, y, x_val=None, y_val=None):
        self.encoder_model_ = CLSFormerClassifier(**self.cls_former_params)
        self.encoder_model_.fit(x, y, x_val=x_val, y_val=y_val)

        z_train = self.encoder_model_.transform(x)
        self.svm_ = SVC(
            C=self.svm_C,
            gamma=self.svm_gamma,
            kernel="rbf",
            probability=True,
            random_state=self.cls_former_params.get("random_state", 42),
        )
        self.svm_.fit(z_train, y)
        self.classes_ = self.svm_.classes_
        return self

    def transform(self, x):
        return self.encoder_model_.transform(x)

    def predict(self, x):
        z = self.transform(x)
        return self.svm_.predict(z)

    def predict_proba(self, x):
        z = self.transform(x)
        return self.svm_.predict_proba(z)


class CLSFormerRegressor(BaseEstimator, RegressorMixin):
    """
    CLS-Former 回归版：
    Transformer feature + regression head + continuous contrastive loss + HuberLoss.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        feature_dim: int = 32,
        epochs: int = 100,
        batch_size: int = 8,
        alpha: float = 0.3,
        temperature: float = 0.1,
        y_temperature: float = 1.0,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        patience: int = 20,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.temperature = temperature
        self.y_temperature = y_temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(self, x, y, x_val=None, y_val=None):
        set_global_seed(self.random_state)

        y = np.asarray(y, dtype=float).ravel()
        self.scaler_ = StandardScaler()
        x_train = as_float32(self.scaler_.fit_transform(x))
        x_valid = as_float32(self.scaler_.transform(x_val)) if x_val is not None else None

        self.y_scaler_ = StandardScaler()
        y_train = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        y_valid = None
        if y_val is not None:
            y_valid = self.y_scaler_.transform(np.asarray(y_val).reshape(-1, 1)).ravel()

        self.model_ = CLSFormerNetwork(
            input_dim=x_train.shape[1],
            output_dim=1,
            task_type="regression",
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            feature_dim=self.feature_dim,
        )

        self.training_info_ = self._train_regressor(x_train, y_train, x_valid, y_valid)
        return self

    def _train_regressor(self, x_train, y_train, x_val, y_val):
        dev = resolve_device(self.device)
        self.model_.to(dev)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        huber = nn.HuberLoss()
        ccon = ContinuousContrastiveLoss(
            temperature=self.temperature,
            y_temperature=self.y_temperature,
        )

        train_loader = make_loader(x_train, y_train, "regression", self.batch_size, True)

        best_state = None
        best_val = np.inf
        no_improve = 0
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)

                opt.zero_grad()
                pred, proj, _ = self.model_(xb)
                loss = self.alpha * ccon(proj, yb) + (1.0 - self.alpha) * huber(pred, yb)

                if not torch.isnan(loss) and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                    opt.step()
                    total_loss += loss.item()

            val_rmse = None
            if x_val is not None and y_val is not None and len(x_val) > 0:
                pred_val = self._predict_scaled(x_val, dev)
                val_rmse = float(np.sqrt(np.mean((pred_val - y_val.ravel()) ** 2)))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break

            history.append({
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "val_rmse": val_rmse,
            })

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return {
            "device": str(dev),
            "history": history,
            "best_val_rmse": None if best_val == np.inf else float(best_val),
            "epochs_ran": len(history),
        }

    def _predict_scaled(self, x_scaled, dev):
        self.model_.eval()
        with torch.no_grad():
            pred, _, _ = self.model_(torch.FloatTensor(x_scaled).to(dev))
            return pred.detach().cpu().numpy().ravel()

    def transform(self, x):
        check_is_fitted(self, "model_")
        x_scaled = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        self.model_.to(dev)
        self.model_.eval()

        loader = DataLoader(torch.FloatTensor(x_scaled), batch_size=128, shuffle=False)
        blocks = []
        with torch.no_grad():
            for xb in loader:
                feat = self.model_(xb.to(dev), return_features=True)
                blocks.append(feat.detach().cpu().numpy())
        return np.vstack(blocks)

    def predict(self, x):
        check_is_fitted(self, "model_")
        x_scaled = as_float32(self.scaler_.transform(x))
        dev = resolve_device(self.device)
        self.model_.to(dev)
        pred_scaled = self._predict_scaled(x_scaled, dev)
        return self.y_scaler_.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


# ============================================================
# 7. registry adapter
# ============================================================

EXPERIMENTAL_SMALL_SAMPLE_MODELS = {
    "spectral_dkl_gp_classifier": SpectralDKLClassifier,
    "spectral_dkl_gp_regressor": SpectralDKLRegressor,

    "proto_spectral_classifier": ProtoSpectralClassifier,
    "proto_spectral_regressor": ProtoSpectralRegressor,

    "cls_former_classifier": CLSFormerClassifier,
    "cls_former_embedding_svm": CLSFormerEmbeddingSVM,
    "cls_former_regressor": CLSFormerRegressor,
}


EXPERIMENTAL_ALIASES = {
    "dkl_gp_classifier": "spectral_dkl_gp_classifier",
    "dkl_gp_regressor": "spectral_dkl_gp_regressor",
    "proto_classifier": "proto_spectral_classifier",
    "proto_regressor": "proto_spectral_regressor",
    "clsformer": "cls_former_classifier",
    "cls_former": "cls_former_classifier",
    "cls_former_svm": "cls_former_embedding_svm",
}


def normalize_experimental_model_name(name: str) -> str:
    key = name.strip().lower()
    return EXPERIMENTAL_ALIASES.get(key, key)


def build_experimental_model(
    name: str,
    task_type: str,
    params: Optional[Dict[str, Any]] = None,
):
    params = dict(params or {})
    canonical = normalize_experimental_model_name(name)

    if canonical not in EXPERIMENTAL_SMALL_SAMPLE_MODELS:
        raise ValueError(f"Unknown experimental small-sample model: {name}")

    if task_type == "classification" and canonical.endswith("_regressor"):
        raise ValueError(f"{canonical} is a regression model, but task_type=classification")

    if task_type == "regression" and canonical.endswith("_classifier"):
        raise ValueError(f"{canonical} is a classification model, but task_type=regression")

    cls = EXPERIMENTAL_SMALL_SAMPLE_MODELS[canonical]
    return cls(**params)
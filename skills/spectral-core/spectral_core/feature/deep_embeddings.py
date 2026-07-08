"""Train-only PyTorch spectral embedding implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils import data


@dataclass
class DeepEmbeddingResult:
    transformed: np.ndarray
    transformer: Any
    parameters: dict[str, Any]
    training_audit: dict[str, Any]
    trace_rows: list[list[Any]]


def train_deep_embedding(
    X: np.ndarray,
    train_indices: list[int],
    *,
    method: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    noise_std: float,
    mask_ratio: float,
    temperature: float,
    patch_size: int,
    random_state: int,
    device: str,
) -> DeepEmbeddingResult:
    torch, nn, data = _torch_modules()
    _validate_training_parameters(
        X,
        train_indices,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        noise_std=noise_std,
        mask_ratio=mask_ratio,
        temperature=temperature,
        patch_size=patch_size,
    )
    resolved_device = _resolve_device(torch, device)
    _set_determinism(torch, random_state)

    X = np.asarray(X, dtype=np.float32)
    train_X = X[train_indices]
    mean = train_X.mean(axis=0).astype(np.float32)
    scale = train_X.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    normalized = ((X - mean) / scale).astype(np.float32)
    train_tensor = torch.from_numpy(normalized[train_indices])
    dataset = data.TensorDataset(train_tensor)
    generator = torch.Generator().manual_seed(random_state)
    loader = data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    architecture = _architecture_for_method(method)
    model = _DeepEmbeddingModel(
        architecture=architecture,
        input_dim=X.shape[1],
        embedding_dim=embedding_dim,
        patch_size=patch_size,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trace_rows: list[list[Any]] = []
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            if method == "contrastive_spectral_embedding":
                if clean.shape[0] < 2:
                    continue
                view_a = _augment(torch, clean, noise_std=noise_std, mask_ratio=mask_ratio)
                view_b = _augment(torch, clean, noise_std=noise_std, mask_ratio=mask_ratio)
                loss = _contrastive_loss(torch, model.encode(view_a), model.encode(view_b), temperature)
            else:
                model_input, loss_mask = _training_input(
                    torch,
                    clean,
                    method=method,
                    noise_std=noise_std,
                    mask_ratio=mask_ratio,
                )
                _, reconstruction = model(model_input)
                squared = (reconstruction - clean).pow(2)
                loss = (squared * loss_mask).sum() / loss_mask.sum().clamp_min(1.0) if loss_mask is not None else squared.mean()
            if not torch.isfinite(loss):
                raise ValueError("Deep embedding training produced a non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise ValueError("Deep embedding training produced no valid optimization batches.")
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        trace_rows.append([epoch, epoch_loss])

    model = model.to("cpu").eval()
    transformer = DeepEmbeddingTransformer(
        model=model,
        mean=mean,
        scale=scale,
        method=method,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )
    transformed = transformer.transform(X)
    training_audit = {
        "status": "completed",
        "objective": "nt_xent" if method == "contrastive_spectral_embedding" else "masked_reconstruction" if method == "masked_spectral_autoencoder_embedding" else "reconstruction",
        "epochs_requested": epochs,
        "epochs_completed": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_finite": bool(all(np.isfinite(losses))),
        "converged": None,
        "n_iter": len(losses),
        "max_iter": epochs,
        "random_seed": random_state,
        "warning": "Fixed-epoch training completed; numerical convergence is not claimed.",
        "device": str(resolved_device),
    }
    parameters = {
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "noise_std": noise_std,
        "mask_ratio": mask_ratio,
        "temperature": temperature,
        "patch_size": patch_size,
        "random_state": random_state,
        "device": str(resolved_device),
        "architecture": architecture,
    }
    return DeepEmbeddingResult(
        transformed=transformed,
        transformer=transformer,
        parameters=parameters,
        training_audit=training_audit,
        trace_rows=trace_rows,
    )


class DeepEmbeddingTransformer:
    """Pickle-friendly train-fitted encoder used by downstream pipeline bundles."""

    def __init__(self, *, model: Any, mean: np.ndarray, scale: np.ndarray, method: str, embedding_dim: int, batch_size: int) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.method = method
        self.embedding_dim = int(embedding_dim)
        self.batch_size = int(batch_size)

    def transform(self, X: Any) -> np.ndarray:
        torch, _, _ = _torch_modules()
        array = np.asarray(X, dtype=np.float32)
        normalized = ((array - self.mean) / self.scale).astype(np.float32)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = torch.from_numpy(normalized[start : start + self.batch_size])
                outputs.append(self.model.encode(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(float)


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(32, min(256, max(embedding_dim * 4, input_dim // 2)))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim))

    def forward(self, x: Any) -> Any:
        return self.net(x)


class _CNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, embedding_dim)

    def forward(self, x: Any) -> Any:
        return self.head(self.net(x.unsqueeze(1)).squeeze(-1))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.act = nn.GELU()

    def forward(self, x: Any) -> Any:
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class _ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 24, kernel_size=7, padding=3), nn.GELU())
        self.blocks = nn.Sequential(_ResidualBlock(24), _ResidualBlock(24))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(24, embedding_dim)

    def forward(self, x: Any) -> Any:
        hidden = self.blocks(self.stem(x.unsqueeze(1)))
        return self.head(self.pool(hidden).squeeze(-1))


class _TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = int(np.ceil(input_dim / patch_size))
        model_dim = max(16, min(64, embedding_dim * 4))
        nhead = 4 if model_dim % 4 == 0 else 2 if model_dim % 2 == 0 else 1
        self.patch_projection = nn.Linear(patch_size, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, self.n_patches + 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(model_dim, embedding_dim)

    def forward(self, x: Any) -> Any:
        pad = self.n_patches * self.patch_size - x.shape[1]
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        patches = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        tokens = self.patch_projection(patches)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1) + self.position)
        return self.head(encoded[:, 0])


class _DeepEmbeddingModel(nn.Module):
    def __init__(self, *, architecture: str, input_dim: int, embedding_dim: int, patch_size: int) -> None:
        super().__init__()
        if architecture == "mlp":
            self.encoder = _MLPEncoder(input_dim, embedding_dim)
        elif architecture == "cnn1d":
            self.encoder = _CNNEncoder(embedding_dim)
        elif architecture == "resnet1d":
            self.encoder = _ResNetEncoder(embedding_dim)
        elif architecture == "cls_transformer":
            self.encoder = _TransformerEncoder(input_dim, embedding_dim, patch_size)
        else:
            raise ValueError(f"Unsupported deep embedding architecture: {architecture}")
        hidden = max(32, min(256, input_dim // 2 or 32))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, input_dim))

    def encode(self, x: Any) -> Any:
        return self.encoder(x)

    def forward(self, x: Any) -> tuple[Any, Any]:
        embedding = self.encode(x)
        return embedding, self.decoder(embedding)


def _training_input(torch: Any, clean: Any, *, method: str, noise_std: float, mask_ratio: float) -> tuple[Any, Any | None]:
    if method == "denoising_autoencoder_embedding":
        return clean + torch.randn_like(clean) * noise_std, None
    if method == "masked_spectral_autoencoder_embedding":
        mask = torch.rand_like(clean) < mask_ratio
        if not bool(mask.any()):
            mask[:, 0] = True
        return clean.masked_fill(mask, 0.0), mask.to(clean.dtype)
    return clean, None


def _augment(torch: Any, clean: Any, *, noise_std: float, mask_ratio: float) -> Any:
    augmented = clean + torch.randn_like(clean) * noise_std
    mask = torch.rand_like(clean) < mask_ratio
    return augmented.masked_fill(mask, 0.0)


def _contrastive_loss(torch: Any, first: Any, second: Any, temperature: float) -> Any:
    z = torch.cat(
        [torch.nn.functional.normalize(first, dim=1), torch.nn.functional.normalize(second, dim=1)],
        dim=0,
    )
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    size = first.shape[0]
    targets = (torch.arange(size * 2, device=z.device) + size) % (size * 2)
    return torch.nn.functional.cross_entropy(logits, targets)


def _architecture_for_method(method: str) -> str:
    if method in {"autoencoder_embedding", "denoising_autoencoder_embedding", "contrastive_spectral_embedding"}:
        return "mlp"
    if method == "cnn_1d_embedding":
        return "cnn1d"
    if method == "resnet1d_embedding":
        return "resnet1d"
    if method in {"cls_former_embedding", "masked_spectral_autoencoder_embedding"}:
        return "cls_transformer"
    raise ValueError(f"Unsupported deep embedding method: {method}")


def _validate_training_parameters(X: np.ndarray, train_indices: list[int], **params: Any) -> None:
    if len(train_indices) < 4:
        raise ValueError("Deep embedding methods require at least four training samples.")
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("Deep embedding methods require a two-dimensional feature matrix with at least two features.")
    if not 1 <= int(params["embedding_dim"]) <= X.shape[1]:
        raise ValueError("embedding_dim must be between 1 and the input feature count.")
    if int(params["epochs"]) < 1 or int(params["batch_size"]) < 2:
        raise ValueError("epochs must be positive and batch_size must be at least 2.")
    if float(params["learning_rate"]) <= 0 or float(params["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative.")
    if float(params["noise_std"]) < 0 or not 0 <= float(params["mask_ratio"]) < 1:
        raise ValueError("noise_std must be non-negative and mask_ratio must be in [0, 1).")
    if float(params["temperature"]) <= 0 or int(params["patch_size"]) < 1:
        raise ValueError("temperature and patch_size must be positive.")


def _resolve_device(torch: Any, requested: str) -> Any:
    normalized = str(requested or "cpu").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if normalized != "cpu" and not normalized.startswith("cuda"):
        raise ValueError("device must be cpu, auto, cuda, or cuda:<index>.")
    return torch.device(normalized)


def _set_determinism(torch: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _torch_modules() -> tuple[Any, Any, Any]:
    return torch, nn, data

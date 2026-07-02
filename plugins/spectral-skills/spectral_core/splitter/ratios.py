"""Ratio parsing and deterministic split counts."""

from __future__ import annotations

import math
from typing import Any


class SplitRatioError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def resolve_ratios(
    *,
    ratio: str | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    confirm_incomplete_ratio: bool = False,
) -> dict[str, float]:
    if ratio:
        parsed = _parse_ratio_string(ratio, confirm_incomplete_ratio=confirm_incomplete_ratio)
    else:
        if train_ratio is None or test_ratio is None:
            raise SplitRatioError("SPLIT_RATIO_REQUIRED", "Please confirm a split ratio such as 8:2, 7:3, or 6:2:2.")
        parsed = {
            "train": float(train_ratio),
            "val": float(val_ratio or 0.0),
            "test": float(test_ratio),
        }
    _validate_ratios(parsed)
    return parsed


def split_counts(n_samples: int, ratios: dict[str, float]) -> dict[str, int]:
    requested = [(name, ratios[name]) for name in ["train", "val", "test"] if ratios[name] > 0]
    if n_samples < len(requested):
        raise SplitRatioError(
            "SPLIT_TOO_FEW_SAMPLES",
            "Not enough samples to create all requested non-empty splits.",
            n_samples=n_samples,
            requested_splits=[name for name, _ in requested],
        )

    raw = {name: n_samples * value for name, value in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    for name, value in ratios.items():
        if value > 0 and counts[name] == 0:
            counts[name] = 1

    while sum(counts.values()) > n_samples:
        candidates = [name for name, value in ratios.items() if value > 0 and counts[name] > 1]
        if not candidates:
            raise SplitRatioError("SPLIT_COUNT_INVALID", "Split ratios cannot produce safe non-empty splits.", n_samples=n_samples, ratios=ratios)
        name = max(candidates, key=lambda item: (counts[item] - raw[item], counts[item]))
        counts[name] -= 1

    while sum(counts.values()) < n_samples:
        candidates = [name for name, value in ratios.items() if value > 0]
        name = max(candidates, key=lambda item: (raw[item] - counts[item], ratios[item]))
        counts[name] += 1

    if ratios["val"] == 0:
        counts["val"] = 0
    for name, value in ratios.items():
        if value > 0 and counts[name] <= 0:
            raise SplitRatioError("SPLIT_EMPTY_PARTITION", f"{name} split would be empty under the requested ratio.", split=name, n_samples=n_samples, ratios=ratios)
    return counts


def _parse_ratio_string(value: str, *, confirm_incomplete_ratio: bool = False) -> dict[str, float]:
    normalized = _normalize_ratio_text(value)
    raw_parts = [part.strip() for part in normalized.split(":")]
    if len(raw_parts) >= 3 and any(part == "" for part in raw_parts):
        if not confirm_incomplete_ratio:
            raise SplitRatioError(
                "SPLIT_RATIO_CONFIRMATION_REQUIRED",
                "Split ratio is incomplete; confirm the intended train/validation/test ratio before splitting.",
                ratio=value,
                recommended_interpretation="train:val:test = 6:2:2" if normalized.startswith("6:2:") else None,
            )
        if len(raw_parts) == 3 and raw_parts[2] == "":
            raw_parts[2] = raw_parts[1]
    parts = [part for part in raw_parts if part]
    if len(parts) not in {2, 3}:
        raise SplitRatioError("SPLIT_RATIO_INVALID", "Split ratio must contain two or three parts, such as 8:2 or 6:2:2.", ratio=value)
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise SplitRatioError("SPLIT_RATIO_INVALID", "Split ratio parts must be numeric.", ratio=value) from exc
    if len(numbers) == 2:
        train, test = numbers
        val = 0.0
    else:
        train, val, test = numbers
    total = train + val + test
    if total <= 0:
        raise SplitRatioError("SPLIT_RATIO_INVALID", "Split ratio total must be positive.", ratio=value)
    return {"train": train / total, "val": val / total, "test": test / total}


def _normalize_ratio_text(value: str) -> str:
    text = value.strip().lower()
    for old in ("\uff1a", "/", ","):
        text = text.replace(old, ":")
    return text


def _validate_ratios(ratios: dict[str, Any]) -> None:
    for name in ["train", "val", "test"]:
        value = float(ratios.get(name, 0.0))
        if value < 0:
            raise SplitRatioError("SPLIT_RATIO_INVALID", "Split ratios cannot be negative.", split=name, value=value)
        ratios[name] = value
    if ratios["train"] <= 0 or ratios["test"] <= 0:
        raise SplitRatioError("SPLIT_RATIO_INVALID", "train and test ratios must be greater than 0.", ratios=ratios)
    total = ratios["train"] + ratios["val"] + ratios["test"]
    if abs(total - 1.0) > 1e-6:
        raise SplitRatioError("SPLIT_RATIO_SUM_INVALID", "train/val/test ratios must sum to 1.", total=total, ratios=ratios)

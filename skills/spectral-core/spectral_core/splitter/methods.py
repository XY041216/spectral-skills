"""Spectral split methods for holdout, CV, repeated holdout, and representative sampling."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

from .ratios import SplitRatioError, split_counts


class SplitMethodError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


HOLDOUT_METHODS = {
    "random",
    "stratified",
    "predefined_split",
    "kennard_stone",
    "spxy",
    "duplex",
    "regression_stratified",
    "y_binned_stratified",
    "group",
    "group_aware",
    "stratified_group",
}
CV_METHODS = {"kfold", "stratified_kfold", "leave_one_out"}
REPEATED_METHODS = {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"}
SUPPORTED_METHODS = HOLDOUT_METHODS | CV_METHODS | REPEATED_METHODS
UNSUPPORTED_METHODS = {"time-series", "time_series", "nested_cv", "repeated_cv"}


def normalize_method(method: str | None) -> str:
    if method is None or not method.strip():
        return "auto"
    normalized = method.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "random_split": "random",
        "stratified_split": "stratified",
        "class_stratified": "stratified",
        "classification_stratified": "stratified",
        "predefined": "predefined_split",
        "external_split": "predefined_split",
        "split_column": "predefined_split",
        "ks": "kennard_stone",
        "kennardstone": "kennard_stone",
        "mc_cv": "monte_carlo_cv",
        "mccv": "monte_carlo_cv",
        "repeated_holdout": "repeated_random_split",
        "repeated_random": "repeated_random_split",
        "stratified_mccv": "stratified_monte_carlo_cv",
        "loocv": "leave_one_out",
        "loo": "leave_one_out",
        "group_aware_split": "group_aware",
        "group_split": "group_aware",
        "stratified_group_split": "stratified_group",
        "regression_binned": "regression_stratified",
        "y_binned": "y_binned_stratified",
    }
    return aliases.get(normalized, normalized)


def split_type_for_method(method: str) -> str:
    if method in CV_METHODS:
        return "cross_validation"
    if method in REPEATED_METHODS:
        return "repeated_holdout"
    return "holdout"


def choose_method(
    *,
    requested_method: str | None,
    task_hint: str,
    has_y: bool,
    confirm_stratified: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    method = normalize_method(requested_method)
    warnings: list[dict[str, Any]] = []
    if method in UNSUPPORTED_METHODS:
        raise SplitMethodError(
            "SPLIT_METHOD_NOT_IMPLEMENTED",
            "This splitter does not implement the requested experimental design yet.",
            requested_method=requested_method,
        )
    if method == "auto":
        if task_hint == "classification" and has_y:
            if not confirm_stratified:
                raise SplitMethodError(
                    "STRATIFIED_CONFIRMATION_REQUIRED",
                    "Classification labels are present. Please confirm stratified split or explicitly request random split.",
                    recommended_method="stratified",
                )
            return "stratified", warnings
        return "random", warnings
    if method not in SUPPORTED_METHODS:
        raise SplitMethodError("SPLIT_METHOD_UNSUPPORTED", "Unsupported split method.", method=requested_method)
    if method == "stratified" and not _is_classification(task_hint):
        raise SplitMethodError(
            "REGRESSION_STRATIFICATION_NOT_IMPLEMENTED",
            "Use regression_stratified or y_binned_stratified for continuous targets.",
            task_hint=task_hint,
        )
    if method in {"stratified", "stratified_kfold", "stratified_monte_carlo_cv", "stratified_group"} and not has_y:
        raise SplitMethodError("STRATIFIED_LABELS_REQUIRED", "Stratified split requires y.csv with class labels.")
    if method == "random" and task_hint == "classification" and has_y:
        warnings.append(
            {
                "code": "CLASSIFICATION_RANDOM_SPLIT",
                "message": "Classification labels are present; stratified split is usually safer for preserving class ratios.",
                "severity": "warning",
                "details": {"recommended_method": "stratified"},
            }
        )
    if method == "spxy" and not has_y:
        raise SplitMethodError("SPXY_REQUIRES_Y", "SPXY requires numeric y values.")
    if method == "spxy" and _is_classification(task_hint):
        raise SplitMethodError("SPXY_REGRESSION_ONLY", "SPXY uses y-distance and is intended for regression tasks, not classification.")
    return method, warnings


def random_split_indices(n_samples: int, ratios: dict[str, float], *, random_seed: int) -> dict[str, list[int]]:
    counts = split_counts(n_samples, ratios)
    indices = list(range(n_samples))
    rng = random.Random(random_seed)
    rng.shuffle(indices)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": sorted(indices[:train_end]),
        "val": sorted(indices[train_end:val_end]),
        "test": sorted(indices[val_end:]),
    }


def stratified_split_indices(labels: list[str], ratios: dict[str, float], *, random_seed: int) -> dict[str, list[int]]:
    global_counts = split_counts(len(labels), ratios)
    requested_splits = [name for name, value in ratios.items() if value > 0]
    class_counts = Counter(labels)
    too_small = {label: count for label, count in class_counts.items() if count < len(requested_splits)}
    if too_small:
        raise SplitMethodError(
            "STRATIFIED_CLASS_TOO_SMALL",
            "Some classes have too few samples to appear in every requested split.",
            class_counts=dict(too_small),
            requested_splits=requested_splits,
        )

    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    rng = random.Random(random_seed)
    assignments = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        class_indices = list(by_label[label])
        rng.shuffle(class_indices)
        local_counts = split_counts(len(class_indices), ratios)
        train_end = local_counts["train"]
        val_end = train_end + local_counts["val"]
        assignments["train"].extend(class_indices[:train_end])
        assignments["val"].extend(class_indices[train_end:val_end])
        assignments["test"].extend(class_indices[val_end:])

    _rebalance_to_global_counts(assignments, labels, global_counts, random_seed=random_seed)
    return {name: sorted(values) for name, values in assignments.items()}


def predefined_split_indices(sample_ids: list[str], split_values: list[str]) -> dict[str, list[int]]:
    if len(split_values) != len(sample_ids):
        raise SplitMethodError("PREDEFINED_SPLIT_LENGTH_MISMATCH", "Predefined split values must match sample count.")
    aliases = {"validation": "val", "valid": "val", "dev": "val", "train": "train", "test": "test", "val": "val"}
    assignments = {"train": [], "val": [], "test": []}
    unknown: list[str] = []
    for idx, value in enumerate(split_values):
        normalized = aliases.get(str(value).strip().lower())
        if normalized is None:
            unknown.append(str(value))
            continue
        assignments[normalized].append(idx)
    if unknown:
        raise SplitMethodError("PREDEFINED_SPLIT_UNKNOWN_LABEL", "Predefined split column contains unsupported split labels.", values=sorted(set(unknown)))
    if not assignments["train"] or not assignments["test"]:
        raise SplitMethodError("PREDEFINED_SPLIT_EMPTY_TRAIN_OR_TEST", "Predefined split must contain non-empty train and test partitions.")
    return assignments


def kfold_indices(n_samples: int, *, n_splits: int, random_seed: int, shuffle: bool = True) -> list[dict[str, Any]]:
    if n_splits < 2 or n_splits > n_samples:
        raise SplitMethodError("KFOLD_INVALID_N_SPLITS", "n_splits must be between 2 and n_samples.", n_splits=n_splits, n_samples=n_samples)
    indices = list(range(n_samples))
    if shuffle:
        random.Random(random_seed).shuffle(indices)
    fold_chunks = _round_robin_chunks(indices, n_splits)
    return _folds_from_chunks(fold_chunks)


def stratified_kfold_indices(labels: list[str], *, n_splits: int, random_seed: int, shuffle: bool = True) -> list[dict[str, Any]]:
    counts = Counter(labels)
    too_small = {label: count for label, count in counts.items() if count < n_splits}
    if too_small:
        raise SplitMethodError("STRATIFIED_KFOLD_CLASS_TOO_SMALL", "Each class must have at least n_splits samples.", class_counts=too_small, n_splits=n_splits)
    rng = random.Random(random_seed)
    fold_chunks = [[] for _ in range(n_splits)]
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    for label in sorted(by_label):
        values = list(by_label[label])
        if shuffle:
            rng.shuffle(values)
        for offset, idx in enumerate(values):
            fold_chunks[offset % n_splits].append(idx)
    return _folds_from_chunks(fold_chunks)


def leave_one_out_indices(n_samples: int) -> list[dict[str, Any]]:
    return [
        {"fold_id": idx + 1, "train_indices": [item for item in range(n_samples) if item != idx], "val_indices": [idx]}
        for idx in range(n_samples)
    ]


def monte_carlo_split_indices(
    n_samples: int,
    ratios: dict[str, float],
    *,
    n_repeats: int,
    random_seed: int,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    repeats = []
    for repeat_id in range(1, n_repeats + 1):
        seed = random_seed + repeat_id - 1
        if labels is not None:
            assignments = stratified_split_indices(labels, ratios, random_seed=seed)
        else:
            assignments = random_split_indices(n_samples, ratios, random_seed=seed)
        repeats.append(
            {
                "repeat_id": repeat_id,
                "train_indices": assignments["train"],
                "val_indices": assignments["val"],
                "test_indices": assignments["test"],
            }
        )
    return repeats


def representative_split_indices(
    x: list[list[float]],
    ratios: dict[str, float],
    *,
    random_seed: int,
    method: str,
    y_values: list[float] | None = None,
    scale: str = "standardize",
) -> dict[str, list[int]]:
    counts = split_counts(len(x), ratios)
    train_count = counts["train"]
    matrix = _standardized_matrix(x) if scale == "standardize" else x
    if method == "spxy":
        if y_values is None:
            raise SplitMethodError("SPXY_REQUIRES_NUMERIC_Y", "SPXY requires numeric y values.")
        y_scaled = _minmax_vector(y_values)
        distance_matrix = _combined_distance_matrix(matrix, y_scaled)
    else:
        distance_matrix = _distance_matrix(matrix)
    if method == "duplex":
        assignments = _duplex_assign(distance_matrix, counts, random_seed=random_seed)
    else:
        train = _kennard_stone_select(distance_matrix, train_count)
        remaining = [idx for idx in range(len(x)) if idx not in set(train)]
        rng = random.Random(random_seed)
        rng.shuffle(remaining)
        val = sorted(remaining[: counts["val"]])
        test = sorted(remaining[counts["val"] :])
        assignments = {"train": sorted(train), "val": val, "test": test}
    return assignments


def regression_stratified_split_indices(y_values: list[float], ratios: dict[str, float], *, random_seed: int, n_bins: int | None = None) -> dict[str, list[int]]:
    bins = regression_bins(y_values, n_bins=n_bins, ratios=ratios)
    target_counts = split_counts(len(y_values), ratios)
    by_bin: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(bins):
        by_bin[label].append(idx)
    rng = random.Random(random_seed)
    assignments = {"train": [], "val": [], "test": []}
    for label in sorted(by_bin):
        values = list(by_bin[label])
        rng.shuffle(values)
        local_counts = split_counts(len(values), ratios)
        train_end = local_counts["train"]
        val_end = train_end + local_counts["val"]
        assignments["train"].extend(values[:train_end])
        assignments["val"].extend(values[train_end:val_end])
        assignments["test"].extend(values[val_end:])
    _rebalance_counts_any(assignments, target_counts, random_seed=random_seed)
    return {name: sorted(values) for name, values in assignments.items()}


def regression_bins(y_values: list[float], *, n_bins: int | None = None, ratios: dict[str, float] | None = None) -> list[str]:
    n = len(y_values)
    if n == 0:
        return []
    requested_splits = sum(1 for value in (ratios or {"train": 1.0, "test": 1.0}).values() if value > 0)
    max_safe_bins = max(1, n // max(1, requested_splits))
    requested_bins = n_bins or min(10, int(math.sqrt(n)) or 2)
    bins = max(1, min(requested_bins, max_safe_bins, n))
    sorted_pairs = sorted((value, idx) for idx, value in enumerate(y_values))
    labels = [""] * n
    for rank, (_, idx) in enumerate(sorted_pairs):
        labels[idx] = f"bin_{min(bins - 1, int(rank * bins / n)) + 1}"
    return labels


def group_split_indices(groups: list[str], ratios: dict[str, float], *, random_seed: int) -> dict[str, list[int]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[str(group)].append(idx)
    group_names = sorted(group_to_indices)
    group_assignments = random_split_indices(len(group_names), ratios, random_seed=random_seed)
    assignments = {"train": [], "val": [], "test": []}
    for split_name, group_positions in group_assignments.items():
        for pos in group_positions:
            assignments[split_name].extend(group_to_indices[group_names[pos]])
    return {name: sorted(values) for name, values in assignments.items()}


def stratified_group_split_indices(labels: list[str], groups: list[str], ratios: dict[str, float], *, random_seed: int) -> dict[str, list[int]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[str(group)].append(idx)
    group_names = sorted(group_to_indices)
    rng = random.Random(random_seed)
    rng.shuffle(group_names)
    target_counts = split_counts(len(labels), ratios)
    assignments = {"train": [], "val": [], "test": []}
    for group in sorted(group_names, key=lambda item: len(group_to_indices[item]), reverse=True):
        best_split = min(["train", "val", "test"], key=lambda name: (len(assignments[name]) - target_counts[name], len(assignments[name])))
        assignments[best_split].extend(group_to_indices[group])
    return {name: sorted(values) for name, values in assignments.items()}


def label_distribution(labels: list[str], assignments: dict[str, list[int]]) -> dict[str, Any]:
    before = dict(Counter(labels))
    after: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        after[split_name] = dict(Counter(labels[idx] for idx in assignments.get(split_name, [])))
    return {"enabled": True, "before": before, "after": after}


def fold_label_distribution(labels: list[str], folds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "enabled": True,
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "train": dict(Counter(labels[idx] for idx in fold.get("train_indices", []))),
                "val": dict(Counter(labels[idx] for idx in fold.get("val_indices", []))),
            }
            for fold in folds
        ],
    }


def group_leakage_check(groups: list[str] | None, assignments: dict[str, list[int]] | None = None) -> dict[str, Any]:
    if not groups or assignments is None:
        return {"enabled": False}
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name, indices in assignments.items():
        for idx in indices:
            seen[str(groups[idx])].add(split_name)
    leaked = {group: sorted(splits) for group, splits in seen.items() if len(splits) > 1}
    return {"enabled": True, "leakage_group_count": len(leaked), "leakage_groups": leaked}


def x_space_coverage(x: list[list[float]], assignments: dict[str, list[int]]) -> dict[str, Any]:
    if not x:
        return {}
    center = [sum(row[col] for row in x) / len(x) for col in range(len(x[0]))]
    return {
        split_name: {
            "n_samples": len(indices),
            "mean_distance_to_global_center": sum(_euclidean(x[idx], center) for idx in indices) / len(indices) if indices else 0.0,
        }
        for split_name, indices in assignments.items()
    }


def fold_size_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    val_sizes = [len(fold.get("val_indices", [])) for fold in folds]
    train_sizes = [len(fold.get("train_indices", [])) for fold in folds]
    return {
        "fold_count": len(folds),
        "train_min": min(train_sizes) if train_sizes else 0,
        "train_max": max(train_sizes) if train_sizes else 0,
        "val_min": min(val_sizes) if val_sizes else 0,
        "val_max": max(val_sizes) if val_sizes else 0,
    }


def repeat_size_summary(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "repeat_count": len(repeats),
        "train_size": len(repeats[0].get("train_indices", [])) if repeats else 0,
        "val_size": len(repeats[0].get("val_indices", [])) if repeats else 0,
        "test_size": len(repeats[0].get("test_indices", [])) if repeats else 0,
    }


def _is_classification(task_hint: str) -> bool:
    return task_hint in {"classification", "class", "categorical"}


def _rebalance_to_global_counts(assignments: dict[str, list[int]], labels: list[str], target_counts: dict[str, int], *, random_seed: int) -> None:
    rng = random.Random(random_seed + 1009)
    for split_name in ["train", "val", "test"]:
        while len(assignments[split_name]) > target_counts[split_name]:
            receiver = min(["train", "val", "test"], key=lambda name: len(assignments[name]) - target_counts[name])
            if len(assignments[receiver]) >= target_counts[receiver]:
                break
            movable = _movable_indices(assignments, split_name, labels)
            if not movable:
                break
            idx = rng.choice(movable)
            assignments[split_name].remove(idx)
            assignments[receiver].append(idx)
    for split_name in ["train", "val", "test"]:
        if len(assignments[split_name]) != target_counts[split_name]:
            raise SplitRatioError(
                "STRATIFIED_REBALANCE_FAILED",
                "Stratified split could not satisfy requested global split counts safely.",
                observed={name: len(values) for name, values in assignments.items()},
                expected=target_counts,
            )


def _rebalance_counts_any(assignments: dict[str, list[int]], target_counts: dict[str, int], *, random_seed: int) -> None:
    rng = random.Random(random_seed + 2003)
    for split_name in ["train", "val", "test"]:
        while len(assignments[split_name]) > target_counts[split_name]:
            receivers = [name for name in ["train", "val", "test"] if len(assignments[name]) < target_counts[name]]
            if not receivers:
                break
            receiver = min(receivers, key=lambda name: len(assignments[name]) - target_counts[name])
            idx = rng.choice(assignments[split_name])
            assignments[split_name].remove(idx)
            assignments[receiver].append(idx)
    for split_name in ["train", "val", "test"]:
        if len(assignments[split_name]) != target_counts[split_name]:
            raise SplitRatioError(
                "REGRESSION_STRATIFIED_REBALANCE_FAILED",
                "Regression-stratified split could not satisfy requested global counts.",
                observed={name: len(values) for name, values in assignments.items()},
                expected=target_counts,
            )


def _movable_indices(assignments: dict[str, list[int]], split_name: str, labels: list[str]) -> list[int]:
    counts = Counter(labels[idx] for idx in assignments[split_name])
    return [idx for idx in assignments[split_name] if counts[labels[idx]] > 1]


def _round_robin_chunks(indices: list[int], n_splits: int) -> list[list[int]]:
    chunks = [[] for _ in range(n_splits)]
    for offset, idx in enumerate(indices):
        chunks[offset % n_splits].append(idx)
    return chunks


def _folds_from_chunks(fold_chunks: list[list[int]]) -> list[dict[str, Any]]:
    all_indices = sorted(idx for chunk in fold_chunks for idx in chunk)
    folds = []
    for fold_id, val_indices in enumerate(fold_chunks, start=1):
        val_set = set(val_indices)
        folds.append(
            {
                "fold_id": fold_id,
                "train_indices": [idx for idx in all_indices if idx not in val_set],
                "val_indices": sorted(val_indices),
            }
        )
    return folds


def _standardized_matrix(x: list[list[float]]) -> list[list[float]]:
    if not x:
        return []
    n_features = len(x[0])
    means = [sum(row[col] for row in x) / len(x) for col in range(n_features)]
    stds = []
    for col in range(n_features):
        variance = sum((row[col] - means[col]) ** 2 for row in x) / max(1, len(x) - 1)
        stds.append(math.sqrt(variance) or 1.0)
    return [[(row[col] - means[col]) / stds[col] for col in range(n_features)] for row in x]


def _minmax_vector(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span == 0:
        return [0.0 for _ in values]
    return [(value - minimum) / span for value in values]


def _distance_matrix(x: list[list[float]]) -> list[list[float]]:
    n = len(x)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = _euclidean(x[i], x[j])
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def _combined_distance_matrix(x: list[list[float]], y: list[float]) -> list[list[float]]:
    matrix = _distance_matrix(x)
    max_x = max((value for row in matrix for value in row), default=1.0) or 1.0
    max_y = max((abs(left - right) for left in y for right in y), default=1.0) or 1.0
    for i in range(len(x)):
        for j in range(len(x)):
            matrix[i][j] = matrix[i][j] / max_x + abs(y[i] - y[j]) / max_y
    return matrix


def _kennard_stone_select(distance_matrix: list[list[float]], n_select: int) -> list[int]:
    if n_select >= len(distance_matrix):
        return list(range(len(distance_matrix)))
    first, second = _farthest_pair(distance_matrix)
    selected = [first, second]
    while len(selected) < n_select:
        remaining = [idx for idx in range(len(distance_matrix)) if idx not in set(selected)]
        next_idx = max(remaining, key=lambda idx: min(distance_matrix[idx][chosen] for chosen in selected))
        selected.append(next_idx)
    return sorted(selected)


def _duplex_assign(distance_matrix: list[list[float]], counts: dict[str, int], *, random_seed: int) -> dict[str, list[int]]:
    first, second = _farthest_pair(distance_matrix)
    assignments = {"train": [first], "test": [second], "val": []}
    assigned = {first, second}
    order = ["train", "test", "val"] if counts["val"] else ["train", "test"]
    while len(assigned) < len(distance_matrix):
        progressed = False
        for split_name in order:
            if len(assignments[split_name]) >= counts[split_name]:
                continue
            remaining = [idx for idx in range(len(distance_matrix)) if idx not in assigned]
            if not remaining:
                break
            selected = assignments[split_name] or [next(iter(assigned))]
            next_idx = max(remaining, key=lambda idx: min(distance_matrix[idx][chosen] for chosen in selected))
            assignments[split_name].append(next_idx)
            assigned.add(next_idx)
            progressed = True
        if not progressed:
            break
    remaining = [idx for idx in range(len(distance_matrix)) if idx not in assigned]
    random.Random(random_seed).shuffle(remaining)
    for idx in remaining:
        receiver = min(["train", "val", "test"], key=lambda name: len(assignments[name]) - counts[name])
        assignments[receiver].append(idx)
    return {name: sorted(values) for name, values in assignments.items()}


def _farthest_pair(distance_matrix: list[list[float]]) -> tuple[int, int]:
    best = (0, 1 if len(distance_matrix) > 1 else 0)
    best_distance = -1.0
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i][j] > best_distance:
                best = (i, j)
                best_distance = distance_matrix[i][j]
    return best


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

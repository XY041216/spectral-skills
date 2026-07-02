"""Unified compact spectral splitting workflow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from spectral_core.reader.response import error_response, ok_response

from .contract import build_split_contract
from .io import SplitInputError, load_split_package
from .methods import (
    SplitMethodError,
    choose_method,
    fold_label_distribution,
    fold_size_summary,
    group_leakage_check,
    group_split_indices,
    kfold_indices,
    label_distribution,
    leave_one_out_indices,
    monte_carlo_split_indices,
    predefined_split_indices,
    regression_bins,
    regression_stratified_split_indices,
    repeat_size_summary,
    representative_split_indices,
    random_split_indices,
    split_type_for_method,
    stratified_group_split_indices,
    stratified_kfold_indices,
    stratified_split_indices,
    x_space_coverage,
)
from .ratios import SplitRatioError, resolve_ratios
from .writer import SplitWriteError, write_split_outputs


def split_spectral_package(
    *,
    package_dir: str | Path,
    output_dir: str | Path | None = None,
    method: str | None = None,
    ratio: str | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    split_column: str | None = None,
    split_indices_file: str | Path | None = None,
    group_column: str | None = None,
    n_splits: int = 5,
    n_repeats: int = 100,
    shuffle: bool = True,
    n_bins: int | None = None,
    scale: str = "standardize",
    random_seed: int = 42,
    confirm_stratified: bool = False,
    confirm_incomplete_ratio: bool = False,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    try:
        package = load_split_package(package_dir)
        selected_method, warnings = choose_method(
            requested_method=method,
            task_hint=package.task_hint,
            has_y=package.labels is not None,
            confirm_stratified=confirm_stratified,
        )
        split_type = split_type_for_method(selected_method)
        ratios = _resolve_method_ratios(
            selected_method,
            ratio=ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            confirm_incomplete_ratio=confirm_incomplete_ratio,
        )
        assignments: dict[str, list[int]] | None = None
        folds: list[dict[str, Any]] | None = None
        repeats: list[dict[str, Any]] | None = None
        distribution: dict[str, Any] = {"enabled": False, "before": {}, "after": {}}
        diagnostics: dict[str, Any] = {}
        active_groups: list[str] | None = None

        if selected_method == "stratified":
            if package.labels is None:
                raise SplitMethodError("STRATIFIED_LABELS_REQUIRED", "Stratified split requires y.csv with class labels.")
            assignments = stratified_split_indices(package.labels, ratios, random_seed=random_seed)
            distribution = label_distribution(package.labels, assignments)
        elif selected_method == "predefined_split":
            split_values = _resolve_predefined_split_values(package, split_column=split_column, split_indices_file=split_indices_file)
            assignments = predefined_split_indices(package.sample_ids, split_values)
            distribution = label_distribution(package.labels, assignments) if package.labels else distribution
        elif selected_method in {"kfold", "stratified_kfold", "leave_one_out"}:
            if selected_method == "stratified_kfold":
                if package.labels is None:
                    raise SplitMethodError("STRATIFIED_LABELS_REQUIRED", "StratifiedKFold requires y.csv with class labels.")
                folds = stratified_kfold_indices(package.labels, n_splits=n_splits, random_seed=random_seed, shuffle=shuffle)
                distribution = fold_label_distribution(package.labels, folds)
            elif selected_method == "leave_one_out":
                folds = leave_one_out_indices(package.n_samples)
            else:
                folds = kfold_indices(package.n_samples, n_splits=n_splits, random_seed=random_seed, shuffle=shuffle)
            diagnostics["fold_size_summary"] = fold_size_summary(folds)
        elif selected_method in {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"}:
            labels = package.labels if selected_method == "stratified_monte_carlo_cv" else None
            repeats = monte_carlo_split_indices(package.n_samples, ratios, n_repeats=n_repeats, random_seed=random_seed, labels=labels)
            diagnostics["repeat_size_summary"] = repeat_size_summary(repeats)
            if labels is not None:
                distribution = {"enabled": True, "before": label_distribution(labels, repeats[0])["before"], "after_preview": label_distribution(labels, repeats[0])["after"]}
        elif selected_method in {"kennard_stone", "spxy", "duplex"}:
            y_values = _numeric_y(package.labels) if selected_method == "spxy" else None
            assignments = representative_split_indices(package.x, ratios, random_seed=random_seed, method=selected_method, y_values=y_values, scale=scale)
            distribution = label_distribution(package.labels, assignments) if package.labels and package.task_hint == "classification" else distribution
            diagnostics["x_space_coverage"] = x_space_coverage(package.x, assignments)
            diagnostics["distance"] = _representative_distance_metadata(selected_method, scale=scale, random_seed=random_seed)
        elif selected_method in {"regression_stratified", "y_binned_stratified"}:
            y_values = _numeric_y(package.labels)
            assignments = regression_stratified_split_indices(y_values, ratios, random_seed=random_seed, n_bins=n_bins)
            diagnostics["regression_target_summary"] = _regression_target_summary(y_values, assignments)
            diagnostics["y_bins"] = dict(__import__("collections").Counter(regression_bins(y_values, n_bins=n_bins, ratios=ratios)))
            diagnostics["binning"] = {"strategy": "quantile", "requested_n_bins": n_bins, "effective_n_bins": len(diagnostics["y_bins"])}
        elif selected_method in {"group", "group_aware", "stratified_group"}:
            groups = _resolve_groups(package, group_column=group_column)
            active_groups = groups
            if selected_method == "stratified_group":
                if package.labels is None:
                    raise SplitMethodError("STRATIFIED_LABELS_REQUIRED", "Stratified group split requires y.csv with class labels.")
                assignments = stratified_group_split_indices(package.labels, groups, ratios, random_seed=random_seed)
                distribution = label_distribution(package.labels, assignments)
            else:
                assignments = group_split_indices(groups, ratios, random_seed=random_seed)
                distribution = label_distribution(package.labels, assignments) if package.labels else distribution
            diagnostics["group_leakage_check"] = group_leakage_check(groups, assignments)
            diagnostics["group_column"] = group_column or _infer_group_column(package)
            diagnostics["group_distribution"] = _group_distribution(groups, assignments)
        else:
            assignments = random_split_indices(package.n_samples, ratios, random_seed=random_seed)
            distribution = {"enabled": False, "before": {}, "after": {}}
        if assignments is not None:
            _assert_complete_assignments(package.n_samples, assignments)

        result_preview = {
            "status": "ready",
            "split_type": split_type,
            "method": selected_method,
            "random_seed": random_seed,
            "ratios": ratios,
            "n_samples": {
                "total": package.n_samples,
                "train": len(assignments["train"]) if assignments else None,
                "val": len(assignments["val"]) if assignments else None,
                "test": len(assignments["test"]) if assignments else None,
            },
            "n_splits": n_splits if folds is not None and selected_method != "leave_one_out" else None,
            "n_repeats": n_repeats if repeats is not None else None,
            "label_distribution": distribution,
            "diagnostics": diagnostics,
            "warnings": warnings,
        }
        if output_dir is None:
            result_preview["handoff_ready"] = False
            result_preview["message"] = "No output_dir was provided; returning split preview without writing split_contract.json."
            return ok_response("split_spectral_package", result_preview, backend=backend, warnings=warnings)

        split_contract = build_split_contract(
            package,
            output_dir=output_dir,
            method=selected_method,
            split_type=split_type,
            random_seed=random_seed,
            ratios=ratios,
            assignments=assignments,
            folds=folds,
            repeats=repeats,
            label_distribution=distribution,
            diagnostics=diagnostics,
            warnings=warnings,
            backend=backend,
        )
        summary = dict(result_preview)
        summary["handoff_ready"] = True
        written = write_split_outputs(
            output_dir=output_dir,
            sample_ids=package.sample_ids,
            labels=package.labels,
            groups=active_groups,
            assignments=assignments,
            folds=folds,
            repeats=repeats,
            split_contract=split_contract,
            summary=summary,
            overwrite=overwrite,
        )
        written["label_distribution"] = distribution
        written["warnings"] = warnings
        return ok_response("split_spectral_package", written, backend=backend, warnings=warnings)
    except SplitInputError as exc:
        return error_response("split_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details)
    except SplitRatioError as exc:
        return error_response("split_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "needs_confirmation"}, details=exc.details)
    except SplitMethodError as exc:
        return error_response("split_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "needs_confirmation"}, details=exc.details)
    except SplitWriteError as exc:
        return error_response("split_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details)


def _assert_complete_assignments(n_samples: int, assignments: dict[str, list[int]]) -> None:
    all_indices = [idx for split_name in ["train", "val", "test"] for idx in assignments.get(split_name, [])]
    if len(all_indices) != len(set(all_indices)):
        raise SplitMethodError("SPLIT_DUPLICATE_SAMPLE", "Generated split contains duplicate samples.")
    expected = set(range(n_samples))
    observed = set(all_indices)
    if observed != expected:
        raise SplitMethodError("SPLIT_INCOMPLETE", "Generated split omitted or added samples.", missing=sorted(expected - observed), extra=sorted(observed - expected))


def _resolve_method_ratios(
    method: str,
    *,
    ratio: str | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    confirm_incomplete_ratio: bool = False,
) -> dict[str, float]:
    if method in {"kfold", "stratified_kfold", "leave_one_out", "predefined_split"}:
        return {"train": 0.0, "val": 0.0, "test": 0.0}
    if method in {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"} and ratio is None and train_ratio is None and test_ratio is None:
        return {"train": 0.7, "val": 0.0, "test": 0.3}
    if method in {"kennard_stone", "spxy", "duplex"} and ratio is None and train_ratio is None and test_ratio is None:
        return {"train": 0.8, "val": 0.0, "test": 0.2}
    return resolve_ratios(
        ratio=ratio,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        confirm_incomplete_ratio=confirm_incomplete_ratio,
    )


def _resolve_predefined_split_values(package: Any, *, split_column: str | None, split_indices_file: str | Path | None) -> list[str]:
    if split_indices_file is not None:
        return _read_external_split_indices(package, split_indices_file)
    candidate_columns = [split_column] if split_column else ["split", "set", "partition"]
    for column in candidate_columns:
        if not column:
            continue
        if column in package.metadata_header:
            return [row.get(column, "") for row in package.metadata_rows]
    raise SplitMethodError("PREDEFINED_SPLIT_SOURCE_MISSING", "predefined_split requires split_indices_file or a metadata split column.", split_column=split_column)


def _read_external_split_indices(package: Any, path: str | Path) -> list[str]:
    sample_to_split: dict[str, str] = {}
    duplicate_assignments: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "split" not in reader.fieldnames:
            raise SplitMethodError("PREDEFINED_SPLIT_FILE_INVALID", "External split file must contain a split column.")
        id_field = "sample_id" if "sample_id" in reader.fieldnames else "index" if "index" in reader.fieldnames else None
        if id_field is None:
            raise SplitMethodError("PREDEFINED_SPLIT_FILE_INVALID", "External split file must contain sample_id or index.")
        for row in reader:
            if id_field == "sample_id":
                sample_id = str(row[id_field]).strip()
            else:
                idx = int(str(row[id_field]).strip())
                if idx < 0 or idx >= len(package.sample_ids):
                    raise SplitMethodError("PREDEFINED_SPLIT_INDEX_OUT_OF_RANGE", "External split file contains an out-of-range sample index.", index=idx)
                sample_id = package.sample_ids[idx]
            if sample_id in sample_to_split:
                duplicate_assignments.append(sample_id)
            sample_to_split[sample_id] = str(row["split"]).strip()
    if duplicate_assignments:
        raise SplitMethodError(
            "PREDEFINED_SPLIT_DUPLICATE_ASSIGNMENT",
            "External split file assigns at least one sample more than once.",
            sample_ids=sorted(set(duplicate_assignments))[:20],
        )
    unknown_samples = [sample_id for sample_id in sample_to_split if sample_id not in set(package.sample_ids)]
    if unknown_samples:
        raise SplitMethodError("PREDEFINED_SPLIT_UNKNOWN_SAMPLE_ID", "External split file contains sample IDs not present in the package.", sample_ids=unknown_samples[:20])
    missing = [sample_id for sample_id in package.sample_ids if sample_id not in sample_to_split]
    if missing:
        raise SplitMethodError("PREDEFINED_SPLIT_FILE_INCOMPLETE", "External split file omits package samples.", sample_ids=missing[:20])
    return [sample_to_split[sample_id] for sample_id in package.sample_ids]


def _numeric_y(labels: list[str] | None) -> list[float]:
    if labels is None:
        raise SplitMethodError("NUMERIC_Y_REQUIRED", "This split method requires numeric y.csv values.")
    try:
        return [float(value) for value in labels]
    except ValueError as exc:
        raise SplitMethodError("NUMERIC_Y_REQUIRED", "This split method requires numeric y.csv values.") from exc


def _resolve_groups(package: Any, *, group_column: str | None) -> list[str]:
    candidates = [group_column] if group_column else _group_column_candidates()
    for column in candidates:
        if column and column in package.metadata_header:
            groups = [row.get(column, "") for row in package.metadata_rows]
            if any(value == "" for value in groups):
                raise SplitMethodError("GROUP_COLUMN_HAS_EMPTY_VALUES", "Group column contains empty values.", group_column=column)
            return groups
    raise SplitMethodError("GROUP_COLUMN_REQUIRED", "Group-aware split requires a metadata group column.", candidates=candidates)


def _group_column_candidates() -> list[str]:
    return ["group_id", "group", "batch", "year", "origin", "replicate_id"]


def _infer_group_column(package: Any) -> str | None:
    for column in _group_column_candidates():
        if column in package.metadata_header:
            return column
    return None


def _regression_target_summary(y_values: list[float], assignments: dict[str, list[int]]) -> dict[str, Any]:
    return {
        split_name: {
            "min": min((y_values[idx] for idx in indices), default=None),
            "max": max((y_values[idx] for idx in indices), default=None),
            "mean": sum(y_values[idx] for idx in indices) / len(indices) if indices else None,
        }
        for split_name, indices in assignments.items()
    }


def _representative_distance_metadata(method: str, *, scale: str, random_seed: int) -> dict[str, Any]:
    metadata = {
        "x_metric": "euclidean",
        "x_scaling": scale,
        "numeric_dtype": "float64",
        "tie_breaking": "deterministic_index_order_then_random_seed_for_remainder",
        "random_seed": random_seed,
    }
    if method == "spxy":
        metadata.update(
            {
                "y_metric": "euclidean",
                "y_scaling": "minmax",
                "combine_rule": "normalized_sum",
            }
        )
    return metadata


def _group_distribution(groups: list[str], assignments: dict[str, list[int]]) -> dict[str, Any]:
    return {
        split_name: dict(__import__("collections").Counter(groups[idx] for idx in indices))
        for split_name, indices in assignments.items()
    }

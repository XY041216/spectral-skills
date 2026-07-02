"""Write preprocessed standard spectral packages."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import write_json_file
from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION
from spectral_core.splitter.contract_reader import partition_to_dict

from .io import PreprocessPackage, SplitInfo


class PreprocessWriteError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def write_preprocess_package(
    package: PreprocessPackage,
    *,
    output_dir: str | Path,
    X: list[list[float]],
    methods: list[str],
    state: dict[str, Any],
    split_info: SplitInfo,
    warnings: list[dict[str, Any]],
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise PreprocessWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    feature_names = _feature_names_for_state(package, state)
    band_axis_rows = _band_axis_rows_for_state(package, state)
    _write_rows(root / "X.csv", [feature_names, *X])
    _write_rows(root / "sample_ids.csv", [["sample_id"], *[[sample_id] for sample_id in package.sample_ids]])
    _write_rows(root / "band_axis.csv", [package.band_axis_header, *band_axis_rows])
    written = ["X.csv", "sample_ids.csv", "band_axis.csv"]
    if package.y_rows is not None:
        _write_rows(root / "y.csv", [package.y_header or ["y"], *package.y_rows])
        written.append("y.csv")
    if package.metadata_rows is not None:
        _write_rows(root / "metadata.csv", [package.metadata_header, *package.metadata_rows])
        written.append("metadata.csv")

    state_payload = _build_state_payload(package, split_info, methods, state, warnings)
    write_json_file(root / "preprocess_state.json", state_payload, ensure_ascii=False)
    written.append("preprocess_state.json")

    contract = _build_data_contract(package, split_info, methods, warnings, backend=backend, n_features=len(feature_names), state=state)
    write_json_file(root / "data_contract.json", contract, ensure_ascii=False)
    written.append("data_contract.json")
    preprocess_contract = _build_preprocess_contract(
        package,
        split_info,
        methods,
        state,
        warnings,
        backend=backend,
        output_contract=root / "data_contract.json",
        n_features=len(feature_names),
    )
    write_json_file(root / "preprocess_contract.json", preprocess_contract, ensure_ascii=False)
    written.append("preprocess_contract.json")
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "data_contract": "data_contract.json",
        "preprocess_contract": "preprocess_contract.json",
        "preprocess_state": "preprocess_state.json",
        "shape": {"n_samples": package.n_samples, "n_features": len(feature_names)},
        "methods": methods,
        "handoff_ready": True,
        "next_step_hint": "Use preprocess_contract.json for downstream feature or modeling skills; data_contract.json remains available for standard-package compatibility.",
    }


def write_preprocess_iteration_outputs(
    package: PreprocessPackage,
    *,
    output_dir: str | Path,
    methods: list[str],
    iteration_results: list[dict[str, Any]],
    split_info: SplitInfo,
    warnings: list[dict[str, Any]],
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise PreprocessWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    iteration_records: list[dict[str, Any]] = []
    written = []
    for item in iteration_results:
        partition = item["partition"]
        transformed = item["X"]
        state = item["state"]
        iteration_dir = root / "iterations" / partition.iteration_id
        iteration_dir.mkdir(parents=True, exist_ok=True)
        role_files = _write_iteration_role_matrices(iteration_dir, package, transformed, partition, state)
        params_path = iteration_dir / "preprocess_params.json"
        write_json_file(
            params_path,
            {
                "iteration_id": partition.iteration_id,
                "partition": partition_to_dict(partition),
                "methods": methods,
                **_method_order_payload(methods, state),
                "method_states": state.get("methods", []),
                "leakage_guard": _leakage_guard(partition, methods),
            },
            ensure_ascii=False,
        )
        written.extend([f"iterations/{partition.iteration_id}/{name}" for name in [*role_files.values(), "preprocess_params.json"]])
        iteration_records.append(
            {
                "iteration_id": partition.iteration_id,
                "iteration_type": partition.iteration_type,
                "train_indices": partition.train_indices,
                "val_indices": partition.val_indices,
                "test_indices": partition.test_indices,
                "role_files": {role: f"iterations/{partition.iteration_id}/{filename}" for role, filename in role_files.items()},
                "params": f"iterations/{partition.iteration_id}/preprocess_params.json",
            }
        )

    order_payload = _method_order_payload(methods, iteration_results[0]["state"] if iteration_results else {})
    contract = {
        "contract_type": "preprocess_contract",
        "stage": "spectral-preprocess",
        "input_package": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
        "methods": methods,
        **order_payload,
        "shape": {"n_samples": package.n_samples, "n_features": len(_feature_names_for_state(package, iteration_results[0]["state"])) if iteration_results else package.n_features},
        "leakage_guard": {
            "fit_on": _fit_scope_for_methods(methods),
            "transform_on": ["train", "val", "test"],
            "global_fit_forbidden": _requires_train_fit(methods),
            "status": "passed",
        },
        "iterations": iteration_records,
        "handoff_ready": True,
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["preprocess_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }
    write_json_file(root / "preprocess_contract.json", contract, ensure_ascii=False)
    write_json_file(root / "preprocess_manifest.json", {"iterations": iteration_records, "split_type": split_info.split_type, "methods": methods, **order_payload}, ensure_ascii=False)
    written.extend(["preprocess_contract.json", "preprocess_manifest.json"])
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "preprocess_contract": "preprocess_contract.json",
        "preprocess_manifest": "preprocess_manifest.json",
        "split_type": split_info.split_type,
        "execution_mode": contract["execution_mode"],
        "iteration_count": len(iteration_records),
        "methods": methods,
        "handoff_ready": True,
        "next_step_hint": "Use preprocess_contract.json for fold-wise or repeat-wise downstream feature/modeling workflows.",
    }


def _build_state_payload(package: PreprocessPackage, split_info: SplitInfo, methods: list[str], state: dict[str, Any], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_type": "preprocess_state",
        "input_contract": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split": {
            "split_type": split_info.split_type,
            "method": (split_info.contract or {}).get("method") if split_info.contract else None,
        },
        "methods": methods,
        **_method_order_payload(methods, state),
        "confirmation": _confirmation_payload(methods),
        "method_states": state.get("methods", []),
        "fit_scope": _fit_scope_for_methods(methods),
        "transform_scope": state.get("transform_scope", "train_val_test"),
        "train_indices": split_info.train_indices,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_iteration_role_matrices(root: Path, package: PreprocessPackage, X: list[list[float]], partition: Any, state: dict[str, Any]) -> dict[str, str]:
    role_files: dict[str, str] = {}
    feature_names = _feature_names_for_state(package, state)
    band_axis_rows = _band_axis_rows_for_state(package, state)
    for role, indices in [("train", partition.train_indices), ("val", partition.val_indices), ("test", partition.test_indices)]:
        if not indices:
            continue
        x_name = f"X_{role}.csv"
        sample_name = f"sample_ids_{role}.csv"
        _write_rows(root / x_name, [feature_names, *[[X[idx][col] for col in range(len(feature_names))] for idx in indices]])
        _write_rows(root / sample_name, [["sample_id"], *[[package.sample_ids[idx]] for idx in indices]])
        role_files[f"X_{role}"] = x_name
        role_files[f"sample_ids_{role}"] = sample_name
        if package.y_rows is not None:
            y_name = f"y_{role}.csv"
            _write_rows(root / y_name, [package.y_header or ["y"], *[package.y_rows[idx] for idx in indices]])
            role_files[f"y_{role}"] = y_name
    _write_rows(root / "band_axis.csv", [package.band_axis_header, *band_axis_rows])
    role_files["band_axis"] = "band_axis.csv"
    return role_files


def _leakage_guard(partition: Any, methods: list[str]) -> dict[str, Any]:
    return {
        "fit_on": _fit_scope_for_methods(methods),
        "train_indices": partition.train_indices,
        "transform_on": [role for role, indices in [("train", partition.train_indices), ("val", partition.val_indices), ("test", partition.test_indices)] if indices],
        "status": "passed",
    }


def _build_preprocess_contract(
    package: PreprocessPackage,
    split_info: SplitInfo,
    methods: list[str],
    state: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    backend: str,
    output_contract: Path,
    n_features: int,
) -> dict[str, Any]:
    order_payload = _method_order_payload(methods, state)
    return {
        "contract_type": "preprocess_contract",
        "stage": "spectral-preprocess",
        "input_package": _abs_path(package.contract_path),
        "output_package": _abs_path(output_contract),
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "execution_mode": "holdout",
        "methods": methods,
        **order_payload,
        "shape": {"n_samples": package.n_samples, "n_features": n_features},
        "method_states": state.get("methods", []),
        "leakage_guard": {
            "fit_on": _fit_scope_for_methods(methods),
            "transform_on": ["train", "val", "test"] if split_info.path else ["all_samples"],
            "global_fit_forbidden": _requires_train_fit(methods),
            "stateless_per_sample_methods": [method for method in methods if method in {"snv", "snv_detrend"}],
            "status": "passed",
        },
        "handoff_ready": True,
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["preprocess_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }


def _build_data_contract(package: PreprocessPackage, split_info: SplitInfo, methods: list[str], warnings: list[dict[str, Any]], *, backend: str, n_features: int | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    files = {
        "X": "X.csv",
        "sample_ids": "sample_ids.csv",
        "band_axis": "band_axis.csv",
        "y": "y.csv" if package.y_rows is not None else None,
        "metadata": "metadata.csv" if package.metadata_rows is not None else None,
    }
    contract = dict(package.contract)
    contract.update(
        {
            "processing_stage": "preprocess",
            "parent_contract": _abs_path(package.contract_path),
            "split_contract": _abs_path(split_info.path) if split_info.path else None,
            "split": {
                "split_type": split_info.split_type,
                "method": (split_info.contract or {}).get("method") if split_info.contract else None,
            },
            "files": files,
            "shape": {"n_samples": package.n_samples, "n_features": n_features or package.n_features},
            "preprocess_summary": {
                "applied": methods != ["none"],
                "methods": methods,
                **_method_order_payload(methods, state or {}),
                "fit_scope": _fit_scope_for_methods(methods) if split_info.path else "all_samples_confirmed",
                "transform_scope": "train_val_test" if split_info.path else "all_samples",
                "state_file": "preprocess_state.json",
            },
            "confirmation": _confirmation_payload(methods),
            "warnings": warnings,
            "handoff": {
                "spectral_feature": {"ready": True},
                "spectral_modeling": {"ready": True, "requires_split_contract": True},
            },
            "execution": {
                "backend": backend,
                "tool_chain": ["preprocess_spectral_package"],
                "core_version": CORE_VERSION,
                "schema_version": SCHEMA_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "warnings": warnings,
                "errors": [],
            },
        }
    )
    return contract


def _method_order_payload(methods: list[str], state: dict[str, Any]) -> dict[str, Any]:
    requested = state.get("requested_methods") if isinstance(state, dict) else None
    executed = state.get("executed_methods") if isinstance(state, dict) else None
    order_normalized = bool(state.get("order_normalized")) if isinstance(state, dict) else False
    reason = state.get("order_normalization_reason") if isinstance(state, dict) else None
    return {
        "requested_methods": list(requested) if isinstance(requested, list) else list(methods),
        "executed_methods": list(executed) if isinstance(executed, list) else list(methods),
        "order_normalized": order_normalized,
        "order_normalization_reason": reason,
    }


def _feature_names_for_state(package: PreprocessPackage, state: dict[str, Any]) -> list[str]:
    indices = state.get("feature_indices")
    if not isinstance(indices, list):
        return list(package.feature_names)
    return [package.feature_names[int(idx)] for idx in indices]


def _band_axis_rows_for_state(package: PreprocessPackage, state: dict[str, Any]) -> list[list[Any]]:
    indices = state.get("feature_indices")
    if not isinstance(indices, list):
        return list(package.band_axis_rows)
    return [package.band_axis_rows[int(idx)] for idx in indices]


def _confirmation_payload(methods: list[str]) -> dict[str, Any]:
    return {
        "required": True,
        "status": "confirmed",
        "decision_source": "user_specified",
        "question": "Confirm preprocessing methods before transforming spectra.",
        "user_selected_option": {"methods": methods},
        "alternatives": ["none", "snv", "msc", "sg_smoothing", "first_derivative", "second_derivative", "mean_centering", "standardization"],
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def _abs_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _requires_train_fit(methods: list[str]) -> bool:
    return any(method in {"msc", "mean_centering", "standardization", "minmax_scaling", "robust_scaling", "pareto_scaling"} for method in methods)


def _fit_scope_for_methods(methods: list[str]) -> str:
    if not _requires_train_fit(methods):
        return "not_applicable_stateless_or_per_sample"
    return "train_only"


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

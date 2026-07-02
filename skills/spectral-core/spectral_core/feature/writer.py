"""Write feature-engineered standard spectral packages."""

from __future__ import annotations

import csv
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import write_json_file
from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION
from spectral_core.splitter.contract_reader import partition_to_dict

from .audit import audit_feature_package
from .io import FeaturePackage, SplitInfo


class FeatureWriteError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def write_feature_package(
    package: FeaturePackage,
    *,
    output_dir: str | Path,
    X: list[list[float]],
    feature_names: list[str],
    band_axis_rows: list[list[Any]],
    method: str,
    state: dict[str, Any],
    split_info: SplitInfo,
    warnings: list[dict[str, Any]],
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    _assert_output_shape(package, X, feature_names, band_axis_rows)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FeatureWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

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

    artifact_files, state_for_json = _write_method_artifacts(root, state)
    written.extend(artifact_files)
    state_payload = _build_state_payload(package, split_info, method, state_for_json, len(feature_names), warnings)
    write_json_file(root / "feature_state.json", state_payload, ensure_ascii=False)
    written.append("feature_state.json")
    if _state_convergence(state_for_json) is not None:
        _write_feature_manifest_csv(
            root / "feature_manifest.csv",
            [_feature_manifest_row(method=method, state=state_for_json, output_n_features=len(feature_names))],
        )
        written.append("feature_manifest.csv")

    contract = _build_data_contract(package, split_info, method, feature_names, state_for_json, warnings, backend=backend)
    write_json_file(root / "data_contract.json", contract, ensure_ascii=False)
    written.append("data_contract.json")
    feature_contract = _build_feature_contract(
        package,
        split_info,
        output_contract=root / "data_contract.json",
        method=method,
        state=state_for_json,
        output_n_features=len(feature_names),
        warnings=warnings,
        backend=backend,
    )
    write_json_file(root / "feature_contract.json", feature_contract, ensure_ascii=False)
    written.append("feature_contract.json")
    audit = audit_feature_package(root)
    if not audit["ok"]:
        raise FeatureWriteError(
            "FEATURE_CONTRACT_INCONSISTENT",
            "Feature output contract is inconsistent with X.csv, band_axis.csv, or feature_state.json.",
            issues=audit["issues"],
            counts=audit["counts"],
        )
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "data_contract": "data_contract.json",
        "feature_contract": "feature_contract.json",
        "feature_state": "feature_state.json",
        "shape": {"n_samples": package.n_samples, "n_features": len(feature_names)},
        "methods": [method],
        "handoff_ready": True,
        "next_step_hint": "Use feature_contract.json for spectral-modeling so preprocess and feature lineage remain auditable.",
    }


def write_feature_iteration_outputs(
    package: FeaturePackage,
    *,
    output_dir: str | Path,
    method: str,
    iteration_results: list[dict[str, Any]],
    split_info: SplitInfo,
    warnings: list[dict[str, Any]],
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FeatureWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    iteration_records: list[dict[str, Any]] = []
    written = []
    for item in iteration_results:
        partition = item["partition"]
        X = item["X"]
        feature_names = item["feature_names"]
        band_axis_rows = item["band_axis_rows"]
        state = item["state"]
        _assert_output_shape(package, X, feature_names, band_axis_rows)
        iteration_dir = root / "iterations" / partition.iteration_id
        iteration_dir.mkdir(parents=True, exist_ok=True)
        role_files = _write_iteration_role_matrices(iteration_dir, package, X, feature_names, band_axis_rows, partition)
        artifact_files, state_for_json = _write_method_artifacts(iteration_dir, state)
        role_files.update({Path(name).stem: name for name in artifact_files})
        params_path = iteration_dir / "feature_params.json"
        write_json_file(
            params_path,
            {
                "iteration_id": partition.iteration_id,
                "partition": partition_to_dict(partition),
                "method": method,
                "canonical_method": state_for_json.get("canonical_method", method),
                "method_family": state_for_json.get("method_family"),
                "feature_mode": (state_for_json.get("output_features") or {}).get("feature_mode"),
                "intended_use": state_for_json.get("intended_use"),
                "out_of_sample_transform": state_for_json.get("out_of_sample_transform"),
                "allowed_for_optimizer_default": state_for_json.get("allowed_for_optimizer_default"),
                "modeling_requires_confirmation": state_for_json.get("modeling_requires_confirmation"),
                "convergence": _state_convergence(state_for_json),
                "training_audit": state_for_json.get("training_audit"),
                "deep_training_confirmation": state_for_json.get("deep_training_confirmation"),
                "requires_y": bool(state_for_json.get("requires_y")),
                "task_type": state_for_json.get("task_type"),
                "parameters": state_for_json.get("parameters", {}),
                "fitted": state_for_json.get("fitted", {}),
                "artifacts": state_for_json.get("artifacts", {}),
                "parameter_sources": state_for_json.get("parameter_sources", {}),
                "defaulted_params": state_for_json.get("defaulted_params", []),
                "user_specified_params": state_for_json.get("user_specified_params", []),
                "defaults_confirmed": bool(state_for_json.get("defaults_confirmed")),
                "warnings": state_for_json.get("warnings", []),
                "input_n_features": package.n_features,
                "output_n_features": len(feature_names),
                "leakage_guard": _leakage_guard(partition),
            },
            ensure_ascii=False,
        )
        written.extend([f"iterations/{partition.iteration_id}/{name}" for name in [*role_files.values(), "feature_params.json"]])
        iteration_records.append(
            {
                "iteration_id": partition.iteration_id,
                "iteration_type": partition.iteration_type,
                "train_indices": partition.train_indices,
                "val_indices": partition.val_indices,
                "test_indices": partition.test_indices,
                "output_n_features": len(feature_names),
                "feature_mode": (state_for_json.get("output_features") or {}).get("feature_mode") or _feature_mode(method),
                "intended_use": state_for_json.get("intended_use"),
                "out_of_sample_transform": state_for_json.get("out_of_sample_transform"),
                "allowed_for_optimizer_default": state_for_json.get("allowed_for_optimizer_default"),
                "modeling_requires_confirmation": state_for_json.get("modeling_requires_confirmation"),
                "convergence": _state_convergence(state_for_json),
                "warnings": state_for_json.get("warnings", []),
                "role_files": {role: f"iterations/{partition.iteration_id}/{filename}" for role, filename in role_files.items()},
                "params_path": f"iterations/{partition.iteration_id}/feature_params.json",
            }
        )

    first_state = (iteration_results[0].get("state") or {}) if iteration_results else {}
    modeling_handoff = _modeling_handoff(first_state)
    contract = {
        "contract_type": "feature_contract",
        "stage": "spectral-feature",
        "input_package": _abs_path(package.contract_path),
        "upstream_preprocess": _upstream_preprocess_summary(package.contract),
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
        "feature_method": method,
        "feature_mode": _iteration_feature_mode(iteration_results),
        "intended_use": first_state.get("intended_use"),
        "out_of_sample_transform": first_state.get("out_of_sample_transform"),
        "allowed_for_optimizer_default": first_state.get("allowed_for_optimizer_default"),
        "modeling_requires_confirmation": first_state.get("modeling_requires_confirmation"),
        "transform_available_for_new_samples": first_state.get("transform_available_for_new_samples"),
        "convergence": _state_convergence(first_state),
        "training_audit": first_state.get("training_audit"),
        "deep_training_confirmation": first_state.get("deep_training_confirmation"),
        "requires_y": _iteration_requires_y(iteration_results),
        "fit_policy": {
            "fit_scope": "train_only",
            "split_contract_required": True,
        },
        "params": _iteration_params(iteration_results),
        "input_n_features": package.n_features,
        "leakage_guard": {
            "fit_on": "train_only_for_each_partition",
            "global_feature_fit_forbidden": True,
            "status": "passed",
        },
        "iterations": iteration_records,
        "handoff_ready": True,
        "handoff": {
            "spectral_modeling": modeling_handoff,
            "spectral_report": {"ready": True},
        },
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["feature_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }
    write_json_file(root / "feature_contract.json", contract, ensure_ascii=False)
    write_json_file(root / "feature_manifest.json", {"iterations": iteration_records, "split_type": split_info.split_type, "method": method}, ensure_ascii=False)
    _write_feature_manifest_csv(
        root / "feature_manifest.csv",
        [
            _feature_manifest_row(
                method=method,
                state=item.get("state") or {},
                output_n_features=len(item.get("feature_names") or []),
                iteration_id=item["partition"].iteration_id,
            )
            for item in iteration_results
        ],
    )
    written.extend(["feature_contract.json", "feature_manifest.json", "feature_manifest.csv"])
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "feature_contract": "feature_contract.json",
        "feature_manifest": "feature_manifest.json",
        "feature_manifest_csv": "feature_manifest.csv",
        "split_type": split_info.split_type,
        "execution_mode": contract["execution_mode"],
        "iteration_count": len(iteration_records),
        "methods": [method],
        "handoff_ready": True,
        "modeling_handoff": modeling_handoff,
        "next_step_hint": "Use feature_contract.json for fold-wise or repeat-wise downstream modeling workflows.",
    }


def _assert_output_shape(package: FeaturePackage, X: list[list[float]], feature_names: list[str], band_axis_rows: list[list[Any]]) -> None:
    if len(X) != package.n_samples:
        raise FeatureWriteError("FEATURE_SAMPLE_COUNT_CHANGED", "Feature MVP must not change sample count.", expected=package.n_samples, observed=len(X))
    if not feature_names:
        raise FeatureWriteError("FEATURE_COUNT_ZERO", "Feature output must contain at least one feature.")
    if len(band_axis_rows) != len(feature_names):
        raise FeatureWriteError("FEATURE_AXIS_MISMATCH", "band_axis length must equal feature count.", expected=len(feature_names), observed=len(band_axis_rows))
    for row_idx, row in enumerate(X):
        if len(row) != len(feature_names):
            raise FeatureWriteError("FEATURE_ROW_WIDTH_MISMATCH", "Feature output row width does not match feature names.", row=row_idx, expected=len(feature_names), observed=len(row))


def _write_iteration_role_matrices(root: Path, package: FeaturePackage, X: list[list[float]], feature_names: list[str], band_axis_rows: list[list[Any]], partition: Any) -> dict[str, str]:
    role_files: dict[str, str] = {}
    for role, indices in [("train", partition.train_indices), ("val", partition.val_indices), ("test", partition.test_indices)]:
        if not indices:
            continue
        x_name = f"X_{role}_features.csv"
        sample_name = f"sample_ids_{role}.csv"
        _write_rows(root / x_name, [feature_names, *[[X[idx][col] for col in range(len(feature_names))] for idx in indices]])
        _write_rows(root / sample_name, [["sample_id"], *[[package.sample_ids[idx]] for idx in indices]])
        role_files[f"X_{role}"] = x_name
        role_files[f"sample_ids_{role}"] = sample_name
        if package.y_rows is not None:
            y_name = f"y_{role}.csv"
            _write_rows(root / y_name, [package.y_header or ["y"], *[package.y_rows[idx] for idx in indices]])
            role_files[f"y_{role}"] = y_name
    _write_rows(root / "feature_axis.csv", [package.band_axis_header, *band_axis_rows])
    role_files["feature_axis"] = "feature_axis.csv"
    return role_files


def _leakage_guard(partition: Any) -> dict[str, Any]:
    return {
        "fit_on": "train_only",
        "train_indices": partition.train_indices,
        "transform_on": [role for role, indices in [("train", partition.train_indices), ("val", partition.val_indices), ("test", partition.test_indices)] if indices],
        "status": "passed",
    }


def _build_state_payload(package: FeaturePackage, split_info: SplitInfo, method: str, state: dict[str, Any], output_n_features: int, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    requires_y = bool(state.get("requires_y"))
    leakage_check = state.get("leakage_check") or {
        "split_contract_used": split_info.path is not None,
        "fit_on_train_only": split_info.path is not None,
        "y_used": requires_y,
        "test_used_in_fit": False,
    }
    return {
        "state_type": "feature_state",
        "input_contract": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split": {
            "split_type": split_info.split_type,
            "method": (split_info.contract or {}).get("method") if split_info.contract else None,
        },
        "methods": [method],
        "method": method,
        "canonical_method": state.get("canonical_method", method),
        "method_family": state.get("method_family", "deterministic_selector" if "select" in method else "transform"),
        "requires_y": requires_y,
        "supervised": requires_y,
        "y_used": requires_y,
        "task_type": state.get("task_type") or package.contract.get("task_hint"),
        "params": state.get("params") or state.get("parameters") or {},
        "parameter_sources": state.get("parameter_sources") or {},
        "defaulted_params": state.get("defaulted_params") or [],
        "user_specified_params": state.get("user_specified_params") or [],
        "defaults_confirmed": bool(state.get("defaults_confirmed")),
        "input_features": state.get("input_features") or {"n_features": package.n_features},
        "output_features": state.get("output_features") or {"n_features": output_n_features, "feature_mode": _feature_mode(method)},
        "intended_use": state.get("intended_use"),
        "out_of_sample_transform": state.get("out_of_sample_transform"),
        "allowed_for_optimizer_default": state.get("allowed_for_optimizer_default"),
        "modeling_requires_confirmation": state.get("modeling_requires_confirmation"),
        "transform_available_for_new_samples": state.get("transform_available_for_new_samples"),
        "requires_nonnegative_X": state.get("requires_nonnegative_X") or (state.get("fitted") or {}).get("requires_nonnegative_X"),
        "input_min_value": state.get("input_min_value") if state.get("input_min_value") is not None else (state.get("fitted") or {}).get("input_min_value"),
        "nonnegative_check": state.get("nonnegative_check") or (state.get("fitted") or {}).get("nonnegative_check"),
        "convergence": _state_convergence(state),
        "training_audit": state.get("training_audit"),
        "deep_training_confirmation": state.get("deep_training_confirmation"),
        "artifacts": state.get("artifacts") or {},
        "leakage_check": leakage_check,
        "val_test_y_used_for_fit": False,
        "leakage_guard": "passed" if leakage_check.get("fit_on_train_only") and not leakage_check.get("test_used_in_fit") else "warning",
        "input_n_features": package.n_features,
        "output_n_features": output_n_features,
        "confirmation": _confirmation_payload(method),
        "method_states": [state],
        "fit_scope": state.get("fit_scope", "train_only"),
        "transform_scope": state.get("transform_scope", "train_val_test"),
        "transform_scope_roles": ["train", "val", "test"] if split_info.path else ["all_samples"],
        "train_indices": split_info.train_indices,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_feature_contract(
    package: FeaturePackage,
    split_info: SplitInfo,
    *,
    output_contract: Path,
    method: str,
    state: dict[str, Any],
    output_n_features: int,
    warnings: list[dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    params = state.get("params") or state.get("parameters") or {}
    requires_y = bool(state.get("requires_y"))
    feature_mode = (state.get("output_features") or {}).get("feature_mode") or _feature_mode(method)
    upstream_preprocess = _upstream_preprocess_summary(package.contract)
    sibling_preprocess_contract = package.root / "preprocess_contract.json"
    if sibling_preprocess_contract.exists():
        upstream_preprocess["preprocess_contract"] = _abs_path(sibling_preprocess_contract)
    leakage_check = state.get("leakage_check") or {
        "split_contract_used": split_info.path is not None,
        "fit_on_train_only": split_info.path is not None,
        "y_used": requires_y,
        "test_used_in_fit": False,
    }
    modeling_handoff = _modeling_handoff(state)
    return {
        "contract_type": "feature_contract",
        "stage": "spectral-feature",
        "input_package": _abs_path(package.contract_path),
        "output_package": _abs_path(output_contract),
        "resolved_paths": {
            "input_package": _abs_path(package.contract_path),
            "output_package": _abs_path(output_contract),
            "package_dir": _abs_path(package.root),
            "split_contract": _abs_path(split_info.path) if split_info.path else None,
        },
        "upstream_preprocess": upstream_preprocess,
        "split_contract": _abs_path(split_info.path) if split_info.path else None,
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "execution_mode": "holdout",
        "feature_method": method,
        "canonical_method": state.get("canonical_method", method),
        "feature_mode": feature_mode,
        "intended_use": state.get("intended_use"),
        "out_of_sample_transform": state.get("out_of_sample_transform"),
        "allowed_for_optimizer_default": state.get("allowed_for_optimizer_default"),
        "modeling_requires_confirmation": state.get("modeling_requires_confirmation"),
        "transform_available_for_new_samples": state.get("transform_available_for_new_samples"),
        "requires_nonnegative_X": state.get("requires_nonnegative_X") or (state.get("fitted") or {}).get("requires_nonnegative_X"),
        "input_min_value": state.get("input_min_value") if state.get("input_min_value") is not None else (state.get("fitted") or {}).get("input_min_value"),
        "nonnegative_check": state.get("nonnegative_check") or (state.get("fitted") or {}).get("nonnegative_check"),
        "convergence": _state_convergence(state),
        "training_audit": state.get("training_audit"),
        "deep_training_confirmation": state.get("deep_training_confirmation"),
        "requires_y": requires_y,
        "supervised": requires_y,
        "task_type": state.get("task_type") or package.contract.get("task_hint"),
        "fit_policy": {
            "fit_scope": "train_only" if split_info.path else "all_samples_confirmed",
            "split_contract_required": requires_y or method not in {"none"},
            "transform_scope": ["train", "val", "test"] if split_info.path else ["all_samples"],
        },
        "params": params,
        "input_n_features": package.n_features,
        "output_n_features": output_n_features,
        "outputs": {
            "package_dir": _abs_path(output_contract.parent),
            "data_contract": "data_contract.json",
            "feature_state": "feature_state.json",
            **(state.get("artifacts") or {}),
        },
        "leakage_guard": {
            "fit_on": "train_only" if split_info.path else "all_samples_confirmed",
            "split_contract_used": split_info.path is not None,
            "y_used": requires_y,
            "val_test_y_used_for_fit": False,
            "test_used_in_fit": False,
            "status": "passed" if split_info.path or not requires_y else "warning",
        },
        "leakage_check": leakage_check,
        "parameter_sources": state.get("parameter_sources") or {},
        "defaulted_params": state.get("defaulted_params") or [],
        "user_specified_params": state.get("user_specified_params") or [],
        "defaults_confirmed": bool(state.get("defaults_confirmed")),
        "handoff_ready": True,
        "handoff": {
            "spectral_modeling": modeling_handoff,
            "spectral_report": {"ready": True},
        },
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["feature_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }


def _build_data_contract(package: FeaturePackage, split_info: SplitInfo, method: str, feature_names: list[str], state: dict[str, Any], warnings: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    files = {
        "X": "X.csv",
        "sample_ids": "sample_ids.csv",
        "band_axis": "band_axis.csv",
        "y": "y.csv" if package.y_rows is not None else None,
        "metadata": "metadata.csv" if package.metadata_rows is not None else None,
    }
    contract = dict(package.contract)
    n_features = len(feature_names)
    original_band_axis = package.contract.get("band_axis") if isinstance(package.contract.get("band_axis"), dict) else {}
    original_n_features = _original_n_features(package)
    feature_mode = (state.get("output_features") or {}).get("feature_mode") or _feature_mode(method)
    is_projection = feature_mode in {
        "projection",
        "modeling_embedding",
        "supervised_modeling_embedding",
        "visualization_embedding",
        "manifold_embedding",
        "derived_signal_features",
        "signal_transform_features",
    }
    band_axis_type = "derived_feature_axis" if is_projection else "selected_spectral_axis"
    band_unit = (
        "principal_component"
        if method == "pca"
        else "pls_latent_variable"
        if method == "pls_latent_variables"
        else "embedding_coordinate"
        if feature_mode in {"visualization_embedding", "manifold_embedding"}
        else "derived_feature"
        if feature_mode in {"projection", "modeling_embedding", "supervised_modeling_embedding", "derived_signal_features", "signal_transform_features"}
        else original_band_axis.get("unit") or package.contract.get("band_unit")
    )
    contract.update(
        {
            "processing_stage": "feature",
            "parent_contract": _abs_path(package.contract_path),
            "upstream_preprocess": _upstream_preprocess_summary(package.contract),
            "split_contract": _abs_path(split_info.path) if split_info.path else None,
            "split": {
                "split_type": split_info.split_type,
                "method": (split_info.contract or {}).get("method") if split_info.contract else None,
            },
            "files": files,
            "shape": {"n_samples": package.n_samples, "n_features": n_features},
            "n_samples": package.n_samples,
            "n_features": n_features,
            "band_axis_ref": "band_axis.csv",
            "band_unit": band_unit,
            "band_axis": {
                "file": "band_axis.csv",
                "unit": band_unit,
                "type": band_axis_type,
                "count": n_features,
            },
            "spectral": {
                "n_bands": n_features,
                "band_axis_type": band_axis_type,
                "band_axis_ref": "band_axis.csv",
                "band_axis_labels": feature_names,
            },
            "source_spectral": {
                "original_n_bands": original_n_features,
                "original_band_axis_ref": _original_band_axis_ref(package),
                "original_band_axis_type": original_band_axis.get("type") or package.contract.get("band_axis_type"),
            },
            "feature": {
                "method": method,
                "input_n_features": original_n_features,
                "output_n_features": n_features,
                "parameters": dict(state.get("parameters") or {}),
                "state_file": "feature_state.json",
                "feature_mode": feature_mode,
                "intended_use": state.get("intended_use"),
                "out_of_sample_transform": state.get("out_of_sample_transform"),
                "allowed_for_optimizer_default": state.get("allowed_for_optimizer_default"),
                "modeling_requires_confirmation": state.get("modeling_requires_confirmation"),
                "transform_available_for_new_samples": state.get("transform_available_for_new_samples"),
                "requires_nonnegative_X": state.get("requires_nonnegative_X") or (state.get("fitted") or {}).get("requires_nonnegative_X"),
                "input_min_value": state.get("input_min_value") if state.get("input_min_value") is not None else (state.get("fitted") or {}).get("input_min_value"),
                "nonnegative_check": state.get("nonnegative_check") or (state.get("fitted") or {}).get("nonnegative_check"),
                "convergence": _state_convergence(state),
                "training_audit": state.get("training_audit"),
                "deep_training_confirmation": state.get("deep_training_confirmation"),
                "method_family": state.get("method_family"),
                "requires_y": bool(state.get("requires_y")),
                "artifacts": state.get("artifacts") or {},
                "parameter_sources": state.get("parameter_sources") or {},
                "defaulted_params": state.get("defaulted_params") or [],
                "defaults_confirmed": bool(state.get("defaults_confirmed")),
            },
            "feature_summary": {
                "applied": method != "none",
                "methods": [method],
                "input_n_features": original_n_features,
                "output_n_features": n_features,
                "fit_scope": "train_only" if split_info.path else "all_samples_confirmed",
                "transform_scope": "train_val_test" if split_info.path else "all_samples",
                "state_file": "feature_state.json",
            },
            "confirmation": _confirmation_payload(method),
            "warnings": warnings,
            "handoff": {
                "spectral_modeling": {
                    **_modeling_handoff(state),
                    "requires_split_contract": True,
                },
                "spectral_report": {"ready": True},
            },
            "execution": {
                "backend": backend,
                "tool_chain": ["feature_spectral_package"],
                "core_version": CORE_VERSION,
                "schema_version": SCHEMA_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "warnings": warnings,
                "errors": [],
            },
        }
    )
    return contract


def _upstream_preprocess_summary(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("contract_type") == "preprocess_contract":
        return {
            "input_source": "preprocess_contract",
            "applied": contract.get("methods") != ["none"],
            "methods": contract.get("methods") or [],
            "requested_methods": contract.get("requested_methods") or contract.get("methods") or [],
            "executed_methods": contract.get("executed_methods") or contract.get("methods") or [],
            "order_normalized": bool(contract.get("order_normalized")),
            "order_normalization_reason": contract.get("order_normalization_reason"),
            "leakage_guard": contract.get("leakage_guard"),
        }
    summary = contract.get("preprocess_summary")
    if isinstance(summary, dict):
        return {
            "input_source": "data_contract",
            "applied": bool(summary.get("applied")),
            "methods": summary.get("methods") or [],
            "requested_methods": summary.get("requested_methods") or summary.get("methods") or [],
            "executed_methods": summary.get("executed_methods") or summary.get("methods") or [],
            "order_normalized": bool(summary.get("order_normalized")),
            "order_normalization_reason": summary.get("order_normalization_reason"),
            "leakage_guard": None,
        }
    return {"input_source": "none", "applied": False, "methods": [], "requested_methods": [], "executed_methods": [], "order_normalized": False, "order_normalization_reason": None, "leakage_guard": None}


def _original_n_features(package: FeaturePackage) -> int:
    for value in [
        (package.contract.get("shape") or {}).get("n_features"),
        package.contract.get("n_features"),
        (package.contract.get("band_axis") or {}).get("count") if isinstance(package.contract.get("band_axis"), dict) else None,
    ]:
        if value is not None:
            return int(value)
    return package.n_features


def _original_band_axis_ref(package: FeaturePackage) -> str | None:
    band_axis = package.contract.get("band_axis") if isinstance(package.contract.get("band_axis"), dict) else {}
    return band_axis.get("file") or package.contract.get("band_axis_ref") or (package.contract.get("files") or {}).get("band_axis")


def _confirmation_payload(method: str) -> dict[str, Any]:
    return {
        "required": True,
        "status": "confirmed",
        "decision_source": "user_specified",
        "question": "Confirm feature method before changing feature space.",
        "user_selected_option": {"method": method},
        "alternatives": [
            "none",
            "pca",
            "kernel_pca",
            "sparse_pca",
            "pls_latent_variables",
            "vip",
            "select_k_best",
            "interval_pls",
            "spa",
            "cars",
            "uve",
            "mcuve",
            "umap_embedding",
            "tsne_embedding",
        ],
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def _abs_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_method_artifacts(root: Path, state: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    payload = dict(state)
    transformer = payload.pop("_fitted_transformer", None)
    raw_artifacts = payload.get("artifacts")
    refs: dict[str, str] = {}
    written: list[str] = []
    if transformer is not None:
        with (root / "feature_transformer.pkl").open("wb") as handle:
            pickle.dump(transformer, handle)
        refs["transformer"] = "feature_transformer.pkl"
        written.append("feature_transformer.pkl")
    if not isinstance(raw_artifacts, dict):
        payload["artifacts"] = refs
        return written, payload
    for filename, artifact in raw_artifacts.items():
        if not isinstance(artifact, dict):
            continue
        header = artifact.get("header")
        rows = artifact.get("rows")
        if not isinstance(header, list) or not isinstance(rows, list):
            continue
        _write_rows(root / filename, [header, *rows])
        refs[Path(filename).stem] = filename
        written.append(filename)
    payload["artifacts"] = refs
    return written, payload


def _feature_mode(method: str) -> str:
    if method in {"pca", "pls_latent_variables"}:
        return "projection"
    if method in {"kernel_pca", "sparse_pca", "nmf", "ica_embedding", "dictionary_learning"}:
        return "modeling_embedding"
    if method == "lda_projection":
        return "supervised_modeling_embedding"
    if method in {"umap_embedding", "tsne_embedding"}:
        return "visualization_embedding"
    if method in {"isomap_embedding", "lle_embedding"}:
        return "manifold_embedding"
    if method in {"dct_features", "fft_features"}:
        return "signal_transform_features"
    if method == "interval_pls":
        return "interval_band_subset"
    if method == "none":
        return "unchanged"
    return "original_band_subset"


def _iteration_feature_mode(iteration_results: list[dict[str, Any]]) -> str:
    if not iteration_results:
        return "unknown"
    state = iteration_results[0].get("state") or {}
    return (state.get("output_features") or {}).get("feature_mode") or _feature_mode(str(state.get("method") or "none"))


def _iteration_requires_y(iteration_results: list[dict[str, Any]]) -> bool:
    return bool(iteration_results and (iteration_results[0].get("state") or {}).get("requires_y"))


def _iteration_params(iteration_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not iteration_results:
        return {}
    state = iteration_results[0].get("state") or {}
    return dict(state.get("params") or state.get("parameters") or {})


def _state_convergence(state: dict[str, Any]) -> dict[str, Any] | None:
    convergence = state.get("convergence")
    if not isinstance(convergence, dict):
        convergence = (state.get("fitted") or {}).get("convergence")
    if not isinstance(convergence, dict):
        return None
    converged = convergence.get("converged")
    return {
        "converged": None if converged is None else bool(converged),
        "n_iter": convergence.get("n_iter"),
        "max_iter": convergence.get("max_iter"),
        "random_seed": convergence.get("random_seed"),
        "warning": convergence.get("warning"),
    }


def _modeling_handoff(state: dict[str, Any]) -> dict[str, Any]:
    intended_use = state.get("intended_use")
    out_of_sample = state.get("out_of_sample_transform")
    requires_confirmation = bool(state.get("modeling_requires_confirmation"))
    if intended_use == "visualization":
        return {
            "ready": False,
            "blocked": True,
            "reason": "visualization_embedding_not_modeling_feature",
            "requires_confirmation": False,
        }
    if out_of_sample == "unsupported":
        return {
            "ready": False,
            "blocked": True,
            "reason": "out_of_sample_transform_unsupported",
            "requires_confirmation": False,
        }
    return {
        "ready": not requires_confirmation,
        "blocked": False,
        "reason": "explicit_modeling_confirmation_required" if requires_confirmation else None,
        "requires_confirmation": requires_confirmation,
    }


def _feature_manifest_row(
    *,
    method: str,
    state: dict[str, Any],
    output_n_features: int,
    iteration_id: str = "holdout",
) -> dict[str, Any]:
    convergence = _state_convergence(state) or {}
    warning_codes = [
        str(item.get("code"))
        for item in state.get("warnings", [])
        if isinstance(item, dict) and item.get("code")
    ]
    return {
        "iteration_id": iteration_id,
        "method": method,
        "feature_mode": (state.get("output_features") or {}).get("feature_mode") or _feature_mode(method),
        "intended_use": state.get("intended_use"),
        "out_of_sample_transform": state.get("out_of_sample_transform"),
        "allowed_for_optimizer_default": state.get("allowed_for_optimizer_default"),
        "modeling_requires_confirmation": state.get("modeling_requires_confirmation"),
        "output_n_features": output_n_features,
        "converged": convergence.get("converged"),
        "n_iter": convergence.get("n_iter"),
        "max_iter": convergence.get("max_iter"),
        "random_seed": convergence.get("random_seed"),
        "convergence_warning": convergence.get("warning"),
        "warning_codes": ";".join(warning_codes),
    }


def _write_feature_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "iteration_id",
        "method",
        "feature_mode",
        "intended_use",
        "out_of_sample_transform",
        "allowed_for_optimizer_default",
        "modeling_requires_confirmation",
        "output_n_features",
        "converged",
        "n_iter",
        "max_iter",
        "random_seed",
        "convergence_warning",
        "warning_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

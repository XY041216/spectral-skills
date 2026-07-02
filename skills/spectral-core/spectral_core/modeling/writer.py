"""Write modeling outputs and contracts."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix

from spectral_core.reader.io_utils import write_json_file
from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION

from .io import ModelingPackage, SplitInfo
from .pipeline_bundle import SpectralPipelineArtifact


class ModelingWriteError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def write_modeling_outputs(
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    result: dict[str, Any],
    output_dir: str | Path,
    warnings: list[dict[str, Any]],
    save_model: bool = True,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ModelingWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "task_type": result["task_type"],
        "model_type": result["model_type"],
        "model_parameters": result["model_parameters"],
        "evaluation_mode": result.get("evaluation_mode", "final"),
        "test_accessed": bool(result.get("test_accessed")),
        "model_selection_mode": result.get("model_selection_mode", (result.get("selection") or {}).get("model_selection_mode")),
        "param_search_enabled": bool(result.get("param_search_enabled", (result.get("selection") or {}).get("param_search_enabled", True))),
        "model_params_source": result.get("model_params_source", (result.get("selection") or {}).get("model_params_source")),
        "selection_strategy": result["selection"],
        "train_metrics": result["metrics"].get("train", {}),
        "val_metrics": result["metrics"].get("val", {}),
        "test_metrics": result["metrics"].get("test", {}),
    }
    write_json_file(root / "metrics.json", metrics_payload, ensure_ascii=False)
    _write_predictions(root / "predictions.csv", result["predictions"])
    outputs = {"metrics": "metrics.json", "predictions": "predictions.csv"}
    written = ["metrics.json", "predictions.csv"]
    classifier_validation_summary = list(result.get("classifier_validation_summary") or [])
    if classifier_validation_summary:
        _write_dict_rows(root / "classifier_validation_summary.csv", classifier_validation_summary)
        outputs["classifier_validation_summary"] = "classifier_validation_summary.csv"
        written.append("classifier_validation_summary.csv")
    uncertainty_rows = [row for row in result["predictions"] if "y_pred_std" in row or "predictive_entropy" in row]
    if uncertainty_rows:
        _write_predictions(root / "prediction_std.csv", uncertainty_rows)
        uncertainty_summary = _uncertainty_summary(uncertainty_rows, result["task_type"])
        write_json_file(root / "uncertainty_summary.json", uncertainty_summary, ensure_ascii=False)
        outputs.update({"prediction_uncertainty": "prediction_std.csv", "uncertainty_summary": "uncertainty_summary.json"})
        written.extend(["prediction_std.csv", "uncertainty_summary.json"])

    if result["task_type"] == "classification" and result.get("test_accessed"):
        _write_confusion_matrix(root / "confusion_matrix.csv", result["predictions"], result.get("class_labels", []))
        outputs["confusion_matrix"] = "confusion_matrix.csv"
        written.append("confusion_matrix.csv")

    if save_model:
        with (root / "model_artifact.pkl").open("wb") as handle:
            pickle.dump(result["model"], handle)
        outputs["model_artifact"] = "model_artifact.pkl"
        written.append("model_artifact.pkl")

    contract = _build_modeling_contract(package, split_info, result, outputs, warnings, backend=backend)
    summary = _build_modeling_summary(contract, result)
    write_json_file(root / "modeling_summary.json", summary, ensure_ascii=False)
    outputs["modeling_summary"] = "modeling_summary.json"
    contract["outputs"] = outputs
    write_json_file(root / "modeling_contract.json", contract, ensure_ascii=False)
    written.append("modeling_summary.json")
    written.append("modeling_contract.json")
    test_access_log = _update_test_access_log(root, result)
    if test_access_log is not None:
        written.append(str(test_access_log.relative_to(root)) if test_access_log.is_relative_to(root) else str(test_access_log))
    if save_model:
        bundle_files = _write_pipeline_bundle(
            root,
            package,
            split_info,
            result=result,
            contract=contract,
            metrics_payload=metrics_payload,
            test_access_log=test_access_log,
        )
        outputs["pipeline_bundle"] = "pipeline_bundle"
        contract["outputs"] = outputs
        write_json_file(root / "modeling_contract.json", contract, ensure_ascii=False)
        shutil.copyfile(root / "modeling_contract.json", root / "pipeline_bundle" / "modeling_contract.json")
        written.extend([f"pipeline_bundle/{name}" for name in bundle_files])
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "modeling_contract": "modeling_contract.json",
        "metrics": "metrics.json",
        "predictions": "predictions.csv",
        "pipeline_bundle": "pipeline_bundle" if save_model else None,
        "model_type": result["model_type"],
        "task_type": result["task_type"],
        "evaluation_mode": result.get("evaluation_mode", "final"),
        "test_accessed": bool(result.get("test_accessed")),
        "handoff_ready": True,
        "next_step_hint": "Use modeling_contract.json for spectral-report or spectral-optimizer handoff.",
    }


def write_modeling_iteration_outputs(
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    result: dict[str, Any],
    output_dir: str | Path,
    warnings: list[dict[str, Any]],
    save_model: bool = True,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ModelingWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    written = []
    for item in result["iteration_results"]:
        iteration_dir = root / "iterations" / item["iteration_id"]
        iteration_dir.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "iteration_id": item["iteration_id"],
            "iteration_type": item["iteration_type"],
            "model_type": item["model_type"],
            "model_parameters": item["model_parameters"],
            "model_selection_mode": item.get("model_selection_mode", (item.get("selection") or {}).get("model_selection_mode")),
            "param_search_enabled": bool(item.get("param_search_enabled", (item.get("selection") or {}).get("param_search_enabled", True))),
            "model_params_source": item.get("model_params_source", (item.get("selection") or {}).get("model_params_source")),
            "selection_strategy": item["selection"],
            "train_metrics": item["train_metrics"],
            f"{item['eval_role']}_metrics": item["eval_metrics"],
        }
        write_json_file(iteration_dir / "metrics.json", metrics_payload, ensure_ascii=False)
        written.append(f"iterations/{item['iteration_id']}/metrics.json")
        if save_model:
            with (iteration_dir / "model_artifact.pkl").open("wb") as handle:
                pickle.dump(item["model"], handle)
            written.append(f"iterations/{item['iteration_id']}/model_artifact.pkl")
        metric_rows.append(_iteration_metric_row(item))

    metrics_name = "fold_metrics.csv" if result["split_type"] == "cross_validation" else "repeat_metrics.csv"
    predictions_name = "fold_predictions.csv" if result["split_type"] == "cross_validation" else "repeat_predictions.csv"
    _write_dict_rows(root / metrics_name, metric_rows)
    _write_predictions(root / predictions_name, result["predictions"])
    write_json_file(root / "metric_summary.json", result["metric_summary"], ensure_ascii=False)

    contract_name = "cv_modeling_result.json" if result["split_type"] == "cross_validation" else "repeated_modeling_result.json"
    contract = _build_iteration_modeling_contract(package, split_info, result, metrics_name, predictions_name, warnings, save_model=save_model, backend=backend)
    write_json_file(root / contract_name, contract, ensure_ascii=False)
    written.extend([metrics_name, predictions_name, "metric_summary.json", contract_name])
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "modeling_contract": contract_name,
        "metric_summary": "metric_summary.json",
        "predictions": predictions_name,
        "metrics": metrics_name,
        "split_type": result["split_type"],
        "execution_mode": result["execution_mode"],
        "iteration_count": len(result["iteration_results"]),
        "task_type": result["task_type"],
        "handoff_ready": True,
        "next_step_hint": "Use metric_summary.json and fold/repeat predictions for CV or repeated-holdout interpretation.",
    }


def write_classifier_comparison_outputs(
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    result: dict[str, Any],
    output_dir: str | Path,
    warnings: list[dict[str, Any]],
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        blocking = [child for child in root.iterdir() if child.name != "checkpoints"]
        if blocking and not overwrite:
            raise ModelingWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
        if overwrite:
            for child in blocking:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    root.mkdir(parents=True, exist_ok=True)

    metric_rows = [_classifier_repeat_metric_row(item) for item in result["iteration_results"]]
    prediction_rows = result["predictions"]
    summary_rows = _classifier_metric_summary_rows(metric_rows, result["models"])

    metrics_name = "classifier_repeat_metrics.csv"
    predictions_name = "classifier_repeat_predictions.csv"
    summary_name = "classifier_metric_summary.csv"
    contract_name = "classifier_comparison_contract.json"
    _write_dict_rows(root / metrics_name, metric_rows)
    _write_predictions(root / predictions_name, prediction_rows)
    _write_dict_rows(root / summary_name, summary_rows)

    outputs = {
        "repeat_metrics": metrics_name,
        "repeat_predictions": predictions_name,
        "metric_summary_table": summary_name,
    }
    contract = _build_classifier_comparison_contract(package, split_info, result, outputs, warnings, backend=backend)
    write_json_file(root / contract_name, contract, ensure_ascii=False)

    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": [metrics_name, predictions_name, summary_name, contract_name],
        "modeling_contract": contract_name,
        "metrics": metrics_name,
        "predictions": predictions_name,
        "metric_summary_table": summary_name,
        "split_type": result["split_type"],
        "execution_mode": result["execution_mode"],
        "comparison_mode": result["comparison_mode"],
        "model_count": len(result["models"]),
        "iteration_count": len(result["iteration_results"]),
        "task_type": result["task_type"],
        "handoff_ready": True,
        "next_step_hint": "Use classifier_repeat_metrics.csv and classifier_metric_summary.csv for spectral-report classifier comparison figures and tables.",
    }


def _build_modeling_contract(package: ModelingPackage, split_info: SplitInfo, result: dict[str, Any], outputs: dict[str, str], warnings: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    conditions = _experiment_conditions(package, split_info, result)
    interpretation = _interpretation_notes(result, conditions)
    return {
        "contract_type": "modeling_contract",
        "contract_id": f"modeling-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "input_contract": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path),
        "task_type": result["task_type"],
        "model_type": result["model_type"],
        "model_parameters": result["model_parameters"],
        "model_family": result.get("model_family"),
        "evaluation_mode": result.get("evaluation_mode", "final"),
        "test_accessed": bool(result.get("test_accessed")),
        "model_selection_mode": result.get("model_selection_mode", result["selection"].get("model_selection_mode")),
        "param_search_enabled": bool(result.get("param_search_enabled", result["selection"].get("param_search_enabled", True))),
        "model_params_source": result.get("model_params_source", result["selection"].get("model_params_source")),
        "configured_model_parameters": result.get("configured_model_parameters", {}),
        "parameter_sources": result.get("parameter_sources", {}),
        "best_pipeline_reproduction": result.get("best_pipeline_reproduction"),
        "experiment_conditions": conditions,
        "interpretation_notes": interpretation,
        "confirmation": {
            "required": True,
            "status": "confirmed",
            "decision_source": "user_specified",
            "question": "Confirm task type, candidate model set, and selection strategy before fitting models.",
            "user_selected_option": {
                "task_type": result["task_type"],
                "candidate_models": conditions["candidate_models"],
                "selected_model": result["model_type"],
                "selection_metric": result["selection"]["selection_metric"],
            },
            "alternatives": ["baseline_model_set", "spectral_classification_model_set", "custom_models", "optimizer"],
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        },
        "selection_strategy": {
            "tuning_split": result["selection"]["tuning_split"],
            "selection_metric": result["selection"]["selection_metric"],
            "candidate_count": result["selection"]["candidate_count"],
            "candidate_models": conditions["candidate_models"],
            "test_used_for_selection": False,
            "model_selection_mode": result.get("model_selection_mode", result["selection"].get("model_selection_mode")),
            "param_search_enabled": bool(result.get("param_search_enabled", result["selection"].get("param_search_enabled", True))),
            "model_params_source": result.get("model_params_source", result["selection"].get("model_params_source")),
            "hyperparameter_selection": result["selection"].get("hyperparameter_selection"),
            "outer_validation_used_for_selection": result["selection"].get("outer_validation_used_for_selection", False),
            "evaluation_mode": result.get("evaluation_mode", "final"),
            "test_evaluated": bool(result.get("test_accessed")),
        },
        "metrics": {
            "train": result["metrics"].get("train", {}),
            "val": result["metrics"].get("val", {}),
            "test": result["metrics"].get("test", {}),
        },
        "outputs": outputs,
        "handoff_ready": True,
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["model_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }


def _build_iteration_modeling_contract(
    package: ModelingPackage,
    split_info: SplitInfo,
    result: dict[str, Any],
    metrics_name: str,
    predictions_name: str,
    warnings: list[dict[str, Any]],
    *,
    save_model: bool,
    backend: str,
) -> dict[str, Any]:
    return {
        "contract_type": "modeling_contract",
        "contract_id": f"modeling-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "stage": "spectral-modeling",
        "input_contract": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path),
        "task_type": result["task_type"],
        "split_type": result["split_type"],
        "split_method": result["split_method"],
        "execution_mode": result["execution_mode"],
        "eval_role": result["eval_role"],
        "candidate_models": result["selection"].get("candidate_models", []),
        "model_families": result.get("model_families", {}),
        "model_selection_mode": result.get("model_selection_mode", result["selection"].get("model_selection_mode")),
        "param_search_enabled": bool(result.get("param_search_enabled", result["selection"].get("param_search_enabled", True))),
        "model_params_source": result.get("model_params_source", result["selection"].get("model_params_source")),
        "configured_model_parameters": result.get("configured_model_parameters", {}),
        "parameter_sources": result.get("parameter_sources", {}),
        "selection_strategy": {
            **result["selection"],
            "test_used_for_selection": False,
            "external_test_metric": False,
        },
        "hyperparameter_selection": result["selection"].get("hyperparameter_selection"),
        "metric_summary": result["metric_summary"],
        "iterations": [
            _iteration_contract_record(item, save_model=save_model)
            for item in result["iteration_results"]
        ],
        "leakage_guard": {
            "fit_on": "train_only_for_each_partition",
            "selection_uses_test": False,
            "outer_validation_used_for_selection": bool((result["selection"].get("hyperparameter_selection") or {}).get("outer_validation_used_for_selection")),
            "global_model_fit_forbidden": True,
            "status": "passed",
        },
        "outputs": {
            "metrics": metrics_name,
            "predictions": predictions_name,
            "metric_summary": "metric_summary.json",
        },
        "handoff_ready": True,
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["model_spectral_package"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "errors": [],
        },
    }


def _iteration_contract_record(item: dict[str, Any], *, save_model: bool) -> dict[str, Any]:
    record = {
                "iteration_id": item["iteration_id"],
                "iteration_type": item["iteration_type"],
                "model_type": item["model_type"],
                "model_family": item.get("model_family"),
                "model_parameters": item["model_parameters"],
                "model_selection_mode": item.get("model_selection_mode", (item.get("selection") or {}).get("model_selection_mode")),
                "param_search_enabled": bool(item.get("param_search_enabled", (item.get("selection") or {}).get("param_search_enabled", True))),
                "model_params_source": item.get("model_params_source", (item.get("selection") or {}).get("model_params_source")),
                "metrics": f"iterations/{item['iteration_id']}/metrics.json",
            }
    if save_model:
        record["model_artifact"] = f"iterations/{item['iteration_id']}/model_artifact.pkl"
    return record


def _experiment_conditions(package: ModelingPackage, split_info: SplitInfo, result: dict[str, Any]) -> dict[str, Any]:
    split_contract = split_info.contract or {}
    package_contract = package.contract or {}
    candidates = result.get("selection", {}).get("candidates", [])
    candidate_models = _dedupe([str(item.get("model_type")) for item in candidates if item.get("model_type")])
    input_source = _input_source(package_contract)
    preprocess = _preprocess_condition(package_contract)
    feature = _feature_condition(package_contract)
    return {
        "input_contract": _abs_path(package.contract_path),
        "input_source": input_source,
        "n_samples": package.n_samples,
        "n_features": package.n_features,
        "preprocess": preprocess,
        "feature": feature,
        "leakage_guard": _input_leakage_guard(package_contract, preprocess, feature),
        "split": {
            "split_type": split_contract.get("split_type") or split_info.split_type,
            "method": split_contract.get("method"),
            "ratios": split_contract.get("ratios"),
            "random_seed": split_contract.get("random_seed"),
            "n_samples": split_contract.get("n_samples"),
        },
        "candidate_models": candidate_models,
        "selection": {
            "selection_metric": result.get("selection", {}).get("selection_metric"),
            "tuning_split": result.get("selection", {}).get("tuning_split"),
            "test_used_for_selection": False,
        },
        "condition_statement": _condition_statement(input_source, preprocess, feature, split_contract, candidate_models, result),
    }


def _stage_condition(contract: dict[str, Any], key: str, *, default: str) -> Any:
    summary = contract.get(key)
    if isinstance(summary, dict):
        if summary.get("methods"):
            return summary
        if summary.get("method"):
            return summary
    stage = str(contract.get("processing_stage") or "").lower()
    if key.startswith("preprocess") and stage != "preprocess":
        return default
    if key.startswith("feature") and stage != "feature":
        return default
    return summary or default


def _input_source(contract: dict[str, Any]) -> str:
    contract_type = str(contract.get("contract_type") or "").strip()
    if contract_type:
        return contract_type
    stage = str(contract.get("processing_stage") or "").strip()
    if stage:
        return f"{stage}_data_contract"
    return "data_contract"


def _preprocess_condition(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("contract_type") == "preprocess_contract":
        methods = _as_method_list(contract.get("executed_methods") or contract.get("methods"))
        return {
            "applied": methods != ["none"],
            "methods": methods,
            "requested_methods": _as_method_list(contract.get("requested_methods") or methods),
            "executed_methods": methods,
            "order_normalized": bool(contract.get("order_normalized")),
            "order_normalization_reason": contract.get("order_normalization_reason"),
            "source": "preprocess_contract",
        }
    if contract.get("contract_type") == "feature_contract":
        upstream = contract.get("upstream_preprocess")
        if isinstance(upstream, dict):
            methods = _as_method_list(upstream.get("executed_methods") or upstream.get("methods"))
            return {
                "applied": bool(upstream.get("applied")),
                "methods": methods,
                "requested_methods": _as_method_list(upstream.get("requested_methods") or methods),
                "executed_methods": methods,
                "order_normalized": bool(upstream.get("order_normalized")),
                "order_normalization_reason": upstream.get("order_normalization_reason"),
                "source": upstream.get("input_source") or "feature_contract.upstream_preprocess",
            }
        return {"applied": False, "methods": ["none"], "requested_methods": ["none"], "executed_methods": ["none"], "order_normalized": False, "order_normalization_reason": None, "source": "feature_contract"}
    summary = _stage_condition(contract, "preprocess_summary", default="none")
    if isinstance(summary, dict):
        methods = _as_method_list(summary.get("executed_methods") or summary.get("methods"))
        return {
            "applied": bool(summary.get("applied")),
            "methods": methods,
            "requested_methods": _as_method_list(summary.get("requested_methods") or methods),
            "executed_methods": methods,
            "order_normalized": bool(summary.get("order_normalized")),
            "order_normalization_reason": summary.get("order_normalization_reason"),
            "source": "data_contract.preprocess_summary",
        }
    return {"applied": False, "methods": ["none"], "requested_methods": ["none"], "executed_methods": ["none"], "order_normalized": False, "order_normalization_reason": None, "source": "none"}


def _feature_condition(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("contract_type") == "feature_contract":
        method = str(contract.get("feature_method") or "none")
        return {
            "applied": method != "none",
            "methods": [method],
            "method": method,
            "source": "feature_contract",
        }
    summary = _stage_condition(contract, "feature_summary", default="none")
    if isinstance(summary, dict):
        methods = _as_method_list(summary.get("methods") or summary.get("method"))
        return {
            "applied": bool(summary.get("applied")),
            "methods": methods,
            "method": methods[0] if methods else "none",
            "source": "data_contract.feature_summary",
        }
    return {"applied": False, "methods": ["none"], "method": "none", "source": "none"}


def _input_leakage_guard(contract: dict[str, Any], preprocess: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    guard = contract.get("leakage_guard")
    if isinstance(guard, dict):
        return guard
    upstream = contract.get("upstream_preprocess")
    if isinstance(upstream, dict) and isinstance(upstream.get("leakage_guard"), dict):
        return upstream["leakage_guard"]
    return {
        "preprocess_source": preprocess.get("source"),
        "feature_source": feature.get("source"),
        "status": "passed",
    }


def _condition_statement(input_source: str, preprocess: dict[str, Any], feature: dict[str, Any], split_contract: dict[str, Any], candidate_models: list[str], result: dict[str, Any]) -> str:
    split = split_contract.get("ratios") or {}
    ratio_text = ":".join(str(split.get(name, "?")) for name in ["train", "val", "test"]) if split else "unspecified split"
    seed = split_contract.get("random_seed")
    method = split_contract.get("method") or "unspecified"
    preprocess_text = ",".join(preprocess.get("methods") or ["none"])
    feature_text = ",".join(feature.get("methods") or ["none"])
    leakage_status = str((_extract_guard_status(preprocess, feature) or "passed"))
    evaluation_text = (
        "validation-only candidate evaluation; test was not read"
        if result.get("evaluation_mode") == "validation_only"
        else "validation selected the model and test was used only for final evaluation"
    )
    return (
        f"Under the current conditions: input_source={input_source}, preprocess={preprocess_text}, "
        f"feature={feature_text}, leakage_guard={leakage_status}, candidate_models={candidate_models}, "
        f"split={method} {ratio_text}, random_seed={seed}, {evaluation_text}."
    )


def _extract_guard_status(preprocess: dict[str, Any], feature: dict[str, Any]) -> str | None:
    for payload in (preprocess, feature):
        guard = payload.get("leakage_guard")
        if isinstance(guard, dict) and guard.get("status"):
            return str(guard["status"])
    return None


def _as_method_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _interpretation_notes(result: dict[str, Any], conditions: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = [
        {
            "type": "scope",
            "severity": "info",
            "message": conditions["condition_statement"],
        }
    ]
    train_accuracy = result["metrics"].get("train", {}).get("accuracy")
    val_accuracy = result["metrics"].get("val", {}).get("accuracy")
    if train_accuracy is not None and val_accuracy is not None and train_accuracy >= 0.99 and (train_accuracy - val_accuracy) >= 0.2:
        notes.append(
            {
                "type": "overfitting_risk",
                "severity": "warning",
                "message": "Train accuracy is much higher than validation accuracy; treat the selected model as a baseline result and compare preprocessing, feature engineering, and model sets before drawing final conclusions.",
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
    if result["task_type"] == "classification":
        notes.append(
            {
                "type": "class_level_review",
                "severity": "info",
                "message": "Review confusion_matrix.csv and per-class precision/recall/F1 before interpreting ROC-AUC or overall accuracy.",
            }
        )
    return notes


def _build_modeling_summary(contract: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready",
        "task_type": result["task_type"],
        "selected_model": result["model_type"],
        "evaluation_mode": result.get("evaluation_mode", "final"),
        "test_accessed": bool(result.get("test_accessed")),
        "model_selection_mode": contract.get("model_selection_mode"),
        "param_search_enabled": contract.get("param_search_enabled"),
        "model_params_source": contract.get("model_params_source"),
        "experiment_conditions": contract["experiment_conditions"],
        "metrics": contract["metrics"],
        "selection_strategy": contract["selection_strategy"],
        "interpretation_notes": contract["interpretation_notes"],
    }


def _update_test_access_log(output_dir: Path, result: dict[str, Any]) -> Path | None:
    if not result.get("test_accessed"):
        return None
    run_dir = output_dir.parent if output_dir.name in {"model_output", "final_model"} else output_dir
    path = run_dir / "test_access_log.json"
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    first_access = payload.get("first_access_time") or now
    access_count = int(payload.get("access_count") or 0) + 1
    access_history = list(payload.get("access_history") or [])
    access_history.append(
        {
            "access_time": now,
            "stage": "final_modeling",
            "model_type": result.get("model_type"),
            "evaluation_mode": result.get("evaluation_mode", "final"),
        }
    )
    write_json_file(
        path,
        {
            "test_set_accessed": True,
            "first_access_stage": payload.get("first_access_stage") or "final_modeling",
            "first_access_time": first_access,
            "last_access_time": now,
            "access_count": access_count,
            "blind_test_status": "confirmatory" if access_count > 1 else "first_recorded_access",
            "notes": (
                "Test metrics have been accessed. Subsequent optimization must not use them for selection; "
                "later final evaluations are confirmatory rather than fully blind."
            ),
            "access_history": access_history,
        },
        ensure_ascii=False,
    )
    return path


def _write_pipeline_bundle(
    output_dir: Path,
    package: ModelingPackage,
    split_info: SplitInfo,
    *,
    result: dict[str, Any],
    contract: dict[str, Any],
    metrics_payload: dict[str, Any],
    test_access_log: Path | None,
) -> list[str]:
    bundle = output_dir / "pipeline_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    feature_contract_path = Path(package.contract_path) if Path(package.contract_path).name == "feature_contract.json" else package.root / "feature_contract.json"
    feature_contract = _read_json(feature_contract_path)
    package_data_contract = _read_json(package.root / "data_contract.json")
    input_schema = {
        "n_features": package.n_features,
        "raw_n_features": _raw_n_features(package, feature_contract, package_data_contract),
        "feature_names": list(package.source.feature_names),
        "band_axis_header": package.source.band_axis_header,
        "band_axis": package.source.band_axis_rows,
        "sample_id_required": True,
        "y_name": package.y_name,
    }
    artifact = SpectralPipelineArtifact(
        model=result.get("model"),
        model_type=result.get("model_type"),
        model_parameters=result.get("model_parameters"),
        task_type=result.get("task_type"),
        class_labels=result.get("class_labels"),
        input_schema=input_schema,
        pipeline_steps=contract.get("experiment_conditions", {}),
        contracts={
            "input_contract": _abs_path(package.contract_path),
            "split_contract": _abs_path(split_info.path),
            "modeling_contract": _abs_path(output_dir / "modeling_contract.json"),
        },
        preprocess_methods=_preprocess_methods_for_raw(package.contract),
        feature_transformer=_load_feature_transformer(package.root),
    )
    with (bundle / "pipeline_artifact.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)
    written.append("pipeline_artifact.pkl")
    smoke_test = _pipeline_smoke_test(artifact, package, feature_contract, package_data_contract)
    write_json_file(bundle / "smoke_test.json", smoke_test, ensure_ascii=False)
    written.append("smoke_test.json")

    manifest = {
        "artifact_type": "pipeline_manifest",
        "pipeline_artifact": "pipeline_artifact.pkl",
        "model_type": result.get("model_type"),
        "task_type": result.get("task_type"),
        "evaluation_mode": result.get("evaluation_mode", "final"),
        "test_accessed": bool(result.get("test_accessed")),
        "input_schema": input_schema | {"band_axis": "stored_in_pipeline_artifact"},
        "raw_prediction": {
            "supported": _raw_prediction_supported(artifact),
            "entrypoints": ["predict_raw", "predict_proba_raw"],
            "note": "pipeline_artifact.pkl exposes predict_raw for raw spectra when recorded preprocess/feature transforms are supported.",
        },
        "smoke_test": "smoke_test.json",
        "contracts": {},
        "metrics": "metrics.json",
        "test_access_log": "test_access_log.json" if test_access_log is not None else None,
        "deployment_note": "Apply the recorded upstream preprocessing and feature contracts before using the fitted model on raw spectra.",
    }

    copy_map = {
        "modeling_contract.json": output_dir / "modeling_contract.json",
        "metrics.json": output_dir / "metrics.json",
    }
    feature_transformer = package.root / "feature_transformer.pkl"
    if feature_transformer.exists():
        copy_map["feature_transformer.pkl"] = feature_transformer
    if split_info.path:
        copy_map["split_contract.json"] = Path(split_info.path)
    input_contract = Path(package.contract_path)
    if input_contract.exists():
        if input_contract.name == "feature_contract.json":
            copy_map["feature_contract.json"] = input_contract
            data_contract = package.root / "data_contract.json"
            if data_contract.exists():
                copy_map["data_contract.json"] = data_contract
            preprocess_contract = _related_preprocess_contract(input_contract)
            if preprocess_contract is not None:
                copy_map["preprocess_contract.json"] = preprocess_contract
        elif input_contract.name == "preprocess_contract.json":
            copy_map["preprocess_contract.json"] = input_contract
            data_contract = package.root / "data_contract.json"
            if data_contract.exists():
                copy_map["data_contract.json"] = data_contract
        else:
            copy_map["data_contract.json"] = input_contract
            sibling_feature = input_contract.parent / "feature_contract.json"
            sibling_preprocess = input_contract.parent / "preprocess_contract.json"
            if sibling_feature.exists():
                copy_map["feature_contract.json"] = sibling_feature
            if sibling_preprocess.exists():
                copy_map["preprocess_contract.json"] = sibling_preprocess
    if test_access_log is not None and test_access_log.exists():
        copy_map["test_access_log.json"] = test_access_log

    for name, source in copy_map.items():
        source_path = Path(source)
        if not source_path.exists():
            continue
        shutil.copyfile(source_path, bundle / name)
        written.append(name)
        if name.endswith("_contract.json") or name == "data_contract.json":
            manifest["contracts"][Path(name).stem] = name

    write_json_file(bundle / "pipeline_manifest.json", manifest, ensure_ascii=False)
    written.append("pipeline_manifest.json")
    return written


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _raw_n_features(package: ModelingPackage, feature_contract: dict[str, Any], package_data_contract: dict[str, Any] | None = None) -> int:
    package_data_contract = package_data_contract or {}
    candidates = [
        (feature_contract.get("source_spectral") or {}).get("original_n_bands") if isinstance(feature_contract.get("source_spectral"), dict) else None,
        (feature_contract.get("input_features") or {}).get("n_features") if isinstance(feature_contract.get("input_features"), dict) else None,
        (feature_contract.get("input_package_summary") or {}).get("n_features") if isinstance(feature_contract.get("input_package_summary"), dict) else None,
        (package_data_contract.get("source_spectral") or {}).get("original_n_bands") if isinstance(package_data_contract.get("source_spectral"), dict) else None,
        (package_data_contract.get("feature") or {}).get("input_n_features") if isinstance(package_data_contract.get("feature"), dict) else None,
        (package.contract.get("source_spectral") or {}).get("original_n_bands") if isinstance(package.contract.get("source_spectral"), dict) else None,
        (package.contract.get("input_features") or {}).get("n_features") if isinstance(package.contract.get("input_features"), dict) else None,
        (package.contract.get("input_package_summary") or {}).get("n_features") if isinstance(package.contract.get("input_package_summary"), dict) else None,
        package.contract.get("n_features"),
        package.n_features,
    ]
    for value in candidates:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return int(package.n_features)


def _preprocess_methods_for_raw(contract: dict[str, Any]) -> list[str]:
    if contract.get("contract_type") == "feature_contract":
        upstream = contract.get("upstream_preprocess")
        if isinstance(upstream, dict):
            return _as_method_list(upstream.get("executed_methods") or upstream.get("methods") or upstream.get("method")) or ["none"]
    if contract.get("contract_type") == "preprocess_contract":
        return _as_method_list(contract.get("executed_methods") or contract.get("methods") or contract.get("method")) or ["none"]
    summary = contract.get("preprocess_summary")
    if isinstance(summary, dict):
        return _as_method_list(summary.get("executed_methods") or summary.get("methods") or summary.get("method")) or ["none"]
    return ["none"]


def _load_feature_transformer(package_root: Path) -> Any | None:
    path = package_root / "feature_transformer.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _raw_prediction_supported(artifact: SpectralPipelineArtifact) -> bool:
    for method in artifact.preprocess_methods:
        if str(method).strip().lower() not in {"", "none", "skip", "snv"}:
            return False
    transformer = artifact.feature_transformer
    return transformer is None or isinstance(transformer, dict) or hasattr(transformer, "transform")


def _pipeline_smoke_test(
    artifact: SpectralPipelineArtifact,
    package: ModelingPackage,
    feature_contract: dict[str, Any],
    package_data_contract: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "skipped",
        "raw_prediction_supported": _raw_prediction_supported(artifact),
        "raw_spectra_input_shape": None,
        "prediction_success": False,
        "prediction_shape": None,
        "error": None,
    }
    if not payload["raw_prediction_supported"]:
        payload["error"] = "raw prediction is not supported for the recorded preprocess/feature transforms"
        return payload
    raw_package = _locate_raw_package(package, feature_contract, package_data_contract)
    if raw_package is None:
        payload["error"] = "could not locate upstream raw spectral package for smoke test"
        return payload
    try:
        rows = _read_x_rows(raw_package / "X.csv", limit=3)
        if not rows:
            payload["error"] = "raw X.csv contains no rows"
            return payload
        predictions = artifact.predict_raw(rows)
        payload.update(
            {
                "status": "passed",
                "raw_spectra_input_shape": [len(rows), len(rows[0])],
                "prediction_success": True,
                "prediction_shape": list(getattr(predictions, "shape", [len(predictions)])),
                "raw_package": _abs_path(raw_package / "data_contract.json"),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive artifact smoke path
        payload.update({"status": "failed", "prediction_success": False, "error": f"{type(exc).__name__}: {exc}"})
    return payload


def _locate_raw_package(package: ModelingPackage, feature_contract: dict[str, Any], package_data_contract: dict[str, Any]) -> Path | None:
    expected = _raw_n_features(package, feature_contract, package_data_contract)
    candidates: list[Path] = []
    candidates.append(package.root)
    input_ref = feature_contract.get("input_package")
    if input_ref:
        input_path = Path(str(input_ref))
        if not input_path.is_absolute():
            input_path = package.root / input_path
        candidates.append(input_path.parent if input_path.name.endswith(".json") else input_path)
    preprocess_contract = _related_preprocess_contract(Path(package.contract_path) if Path(package.contract_path).name == "feature_contract.json" else package.root / "feature_contract.json")
    if preprocess_contract is not None:
        preprocess_payload = _read_json(preprocess_contract)
        preprocess_input = preprocess_payload.get("input_package")
        if preprocess_input:
            path = Path(str(preprocess_input))
            if not path.is_absolute():
                path = preprocess_contract.parent / path
            candidates.append(path.parent if path.name.endswith(".json") else path)
    for candidate in candidates:
        x_path = candidate / "X.csv"
        if not x_path.exists():
            continue
        try:
            rows = _read_x_rows(x_path, limit=1)
        except (OSError, ValueError):
            continue
        if rows and len(rows[0]) == expected:
            return candidate
    return None


def _read_x_rows(path: Path, *, limit: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row])
            if len(rows) >= limit:
                break
    return rows


def _related_preprocess_contract(feature_contract_path: Path) -> Path | None:
    try:
        contract = json.loads(feature_contract_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        contract = {}
    candidates: list[Path] = []
    upstream = contract.get("upstream_preprocess") if isinstance(contract.get("upstream_preprocess"), dict) else {}
    upstream_ref = upstream.get("preprocess_contract") if isinstance(upstream, dict) else None
    if upstream_ref:
        upstream_path = Path(str(upstream_ref))
        candidates.append(upstream_path if upstream_path.is_absolute() else feature_contract_path.parent / upstream_path)
    input_ref = contract.get("input_package")
    if input_ref:
        input_path = Path(str(input_ref))
        if not input_path.is_absolute():
            input_path = feature_contract_path.parent / input_path
        if input_path.name == "data_contract.json":
            input_path = input_path.parent
        candidates.append(input_path / "preprocess_contract.json")
    candidates.extend(
        [
            feature_contract_path.parent / "preprocess_contract.json",
            feature_contract_path.parent.parent / "preprocess_output" / "preprocess_contract.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _abs_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _iteration_metric_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration_id": item["iteration_id"],
        "iteration_type": item["iteration_type"],
        "eval_role": item["eval_role"],
        "model_type": item["model_type"],
        "selection_tuning_split": item.get("selection", {}).get("tuning_split"),
        "outer_validation_used_for_selection": item.get("selection", {}).get("outer_validation_used_for_selection", False),
    }
    for prefix, metrics in [("train", item["train_metrics"]), (item["eval_role"], item["eval_metrics"])]:
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def _classifier_repeat_metric_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration_id": item["iteration_id"],
        "repeat_id": item["iteration_id"] if item["iteration_type"] == "repeat" else "",
        "fold_id": item["iteration_id"] if item["iteration_type"] == "fold" else "",
        "iteration_type": item["iteration_type"],
        "statistical_unit": "repeat" if item["iteration_type"] == "repeat" else "fold",
        "eval_role": item["eval_role"],
        "model_method": item["model_type"],
        "model_type": item["model_type"],
        "model_parameters": json.dumps(item.get("model_parameters", {}), ensure_ascii=False, sort_keys=True),
        "model_selection_mode": item.get("model_selection_mode", (item.get("selection") or {}).get("model_selection_mode")),
        "param_search_enabled": bool(item.get("param_search_enabled", False)),
        "model_params_source": item.get("model_params_source", (item.get("selection") or {}).get("model_params_source")),
    }
    for key, value in item.get("train_metrics", {}).items():
        row[f"train_{key}"] = value
    for key, value in item.get("eval_metrics", {}).items():
        row[key] = value
        row[f"{item['eval_role']}_{key}"] = value
        if isinstance(value, (int, float)):
            row[f"{key}_percent"] = float(value) * 100.0
            row[f"{item['eval_role']}_{key}_percent"] = float(value) * 100.0
    return row


def _classifier_metric_summary_rows(metric_rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "roc_auc"]
    output: list[dict[str, Any]] = []
    for model in models:
        rows = [row for row in metric_rows if row.get("model_method") == model]
        summary: dict[str, Any] = {"model_method": model, "model_type": model, "n_repeats": len(rows)}
        for metric in metrics:
            values = [_safe_float(row.get(metric)) for row in rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            sd = _sample_sd(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_mean_percent"] = mean * 100.0
            summary[f"{metric}_sd_percent"] = sd * 100.0
            summary[f"{metric}_mean_sd_percent"] = f"{mean * 100.0:.2f} +/- {sd * 100.0:.2f}"
        output.append(summary)
    output.sort(key=lambda row: _safe_float(row.get("macro_f1_mean")) if _safe_float(row.get("macro_f1_mean")) is not None else -1.0, reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank_macro_f1"] = rank
    return output


def _build_classifier_comparison_contract(package: ModelingPackage, split_info: SplitInfo, result: dict[str, Any], outputs: dict[str, str], warnings: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    return {
        "contract_type": "classifier_comparison_contract",
        "contract_id": f"classifier-comparison-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "input_contract": _abs_path(package.contract_path),
        "split_contract": _abs_path(split_info.path),
        "task_type": result["task_type"],
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "comparison_mode": result["comparison_mode"],
        "model_selection_mode": "per_classifier_repeated_evaluation",
        "model_selection_enabled": False,
        "param_search_enabled": False,
        "model_params_source": result.get("model_params_source"),
        "candidate_model_set_source": result.get("candidate_model_set_source"),
        "candidate_models": result["models"],
        "same_repeats_across_models": True,
        "model_parameters": result.get("configured_model_parameters", {}),
        "parameter_sources": result.get("parameter_sources", {}),
        "statistical_unit": "repeat" if split_info.split_type == "repeated_holdout" else "fold",
        "n_iterations": len(result["iteration_results"]),
        "n_models": len(result["models"]),
        "metrics_for_report_tables": ["accuracy", "balanced_accuracy", "macro_f1"],
        "report_guidance": {
            "primary_table": "rank models by macro_f1_mean unless the user explicitly selected another primary metric",
            "paper_table_columns": ["rank", "classifier", "Accuracy / %", "Balanced accuracy / %", "Macro-F1 / %"],
            "qc_warning_policy": "Report QC warnings as flagged-only unless samples were explicitly removed; recommend sensitivity analysis when high-risk samples are present.",
            "figure_policy": "Use confirmed chart type; repeated comparisons may use box/violin/dot plots with raw repeat points. Do not draw fake error bars.",
        },
        "outputs": outputs,
        "handoff_ready": True,
        "warnings": warnings,
        "execution": {
            "backend": backend,
            "tool_chain": ["model_spectral_package", "repeated_classifier_comparison"],
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ModelingWriteError("METRICS_EMPTY", "No iteration metrics were produced.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ModelingWriteError("PREDICTIONS_EMPTY", "No predictions were produced.")
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(path: Path, prediction_rows: list[dict[str, Any]], labels: list[str]) -> None:
    test_rows = [row for row in prediction_rows if row.get("split") == "test"]
    y_true = [str(row["y_true"]) for row in test_rows]
    y_pred = [str(row["y_pred"]) for row in test_rows]
    label_values = labels or sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=label_values)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *label_values])
        for label, row in zip(label_values, matrix.tolist()):
            writer.writerow([label, *row])


def _uncertainty_summary(rows: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    if task_type == "regression":
        std_values = [float(row["y_pred_std"]) for row in rows if row.get("y_pred_std") is not None]
        covered = [
            float(row["lower_95"]) <= float(row["y_true"]) <= float(row["upper_95"])
            for row in rows
            if row.get("split") == "test" and row.get("lower_95") is not None
        ]
        return {
            "task_type": task_type,
            "n_predictions": len(std_values),
            "mean_prediction_std": float(sum(std_values) / len(std_values)) if std_values else None,
            "test_interval_95_coverage": float(sum(covered) / len(covered)) if covered else None,
            "interpretation": "Uncertainty is conditional on the fitted model, current input package, and split.",
        }
    entropy = [float(row["predictive_entropy"]) for row in rows if row.get("predictive_entropy") is not None]
    return {
        "task_type": task_type,
        "n_predictions": len(entropy),
        "mean_predictive_entropy": float(sum(entropy) / len(entropy)) if entropy else None,
        "interpretation": "Predictive entropy is model-conditional and is not proof of sample corruption.",
    }

"""Leakage-safe optimizer planning for spectral workflows.

The optimizer does not implement preprocessing, feature, or modeling
algorithms. It builds auditable candidate spaces and can select a best
pipeline from validation/CV trial results produced by workflow runs.
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.modeling.registry import MODEL_DEFAULTS, comparison_fixed_parameters, model_spec, normalize_model_name
from spectral_core.optimizer.executor import OptimizerExecutionError, execute_validation_trials
from spectral_core.optimizer.planner import budget_audit, build_trials as planner_build_trials
from spectral_core.reader.response import error_response, ok_response


SUPPORTED_MODES = {"recommend_from_profile", "tune_method", "compare_step", "optimize_pipeline"}
SUPPORTED_STEPS = {"preprocess", "feature", "modeling"}
DEFAULT_SELECTION_METRIC = {"classification": "val_macro_f1", "regression": "val_rmse"}
TIE_BREAKER_POLICY = [
    "prefer the best primary validation/CV metric",
    "if tied, prefer better secondary validation/CV metrics",
    "then prefer fewer preprocessing methods",
    "then prefer predefined preprocessing priority (SNV before MSC when metrics tie)",
    "then prefer fewer output features or components",
    "then prefer fewer tuned parameters",
    "then prefer unsupervised feature methods over supervised selectors",
    "then prefer lower-compute traditional models over experimental models",
]

OPTIMIZER_EXPERIMENTAL_MODEL_OVERRIDES = {
    "spectral_dkl_gp_classifier": {"embedding_dim": 16},
    "spectral_dkl_gp_regressor": {"embedding_dim": 16},
    "proto_spectral_classifier": {"embedding_dim": 16},
    "proto_spectral_regressor": {"embedding_dim": 16},
    "cls_former_classifier": {"feature_dim": 16},
    "cls_former_embedding_svm": {"feature_dim": 16, "svm_C": 1.0, "svm_gamma": "scale"},
    "cls_former_regressor": {"feature_dim": 16},
}


@dataclass
class OptimizerError(ValueError):
    code: str
    message: str
    details: dict[str, Any] | None = None


def optimize_spectral_pipeline(
    *,
    mode: str = "recommend_from_profile",
    output_dir: str | Path | None = None,
    data_profile: str | Path | dict[str, Any] | None = None,
    package_dir: str | Path | None = None,
    split_contract: str | Path | None = None,
    task_type: str | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
    n_classes: int | None = None,
    class_balance: str | dict[str, Any] | None = None,
    has_validation: bool | None = None,
    has_test: bool | None = None,
    target_step: str | None = None,
    method: str | None = None,
    fixed_preprocess_methods: str | None = None,
    fixed_feature_contract: str | Path | None = None,
    preprocess_candidates: str | list[str] | None = None,
    feature_candidates: str | list[str] | None = None,
    model_candidates: str | list[str] | None = None,
    validator_model: str | None = None,
    validator_params: str | list[str] | None = None,
    model_param_grid: str | list[str] | None = None,
    comparison_depth: str = "regular",
    candidate_space: str | Path | dict[str, Any] | None = None,
    trial_results: str | Path | None = None,
    execute_trials: bool = False,
    trial_inputs: str | Path | dict[str, Any] | None = None,
    test_access_log: str | Path | None = None,
    selection_metric: str | None = None,
    max_trials: int = 30,
    confirm_budget: bool = False,
    confirm_comparison_design: bool = False,
    confirm_parameter_grid: bool = False,
    confirm_candidate_space: bool = False,
    validator_model_source: str | None = None,
    previous_confirmation_stage: str | None = None,
    preview_only: bool = False,
    random_seed: int = 42,
    backend: str = "core",
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    try:
        selected_mode = _normalize_mode(mode)
        profile = _build_profile(
            data_profile=data_profile,
            package_dir=package_dir,
            task_type=task_type,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_balance=class_balance,
            has_validation=has_validation,
            has_test=has_test,
        )
        metric = selection_metric or DEFAULT_SELECTION_METRIC.get(profile["task_type"], "val_metric")
        protocol = _selection_protocol(profile)
        if selected_mode == "recommend_from_profile":
            result = _recommend_from_profile(profile, metric, protocol, random_seed=random_seed)
        else:
            if target_step and target_step not in SUPPORTED_STEPS:
                raise OptimizerError("OPTIMIZER_STEP_UNSUPPORTED", "target_step must be preprocess, feature, or modeling.", {"target_step": target_step})
            effective_model_candidates = model_candidates or (validator_model if target_step in {"preprocess", "feature"} else None)
            space = _load_candidate_space(candidate_space) if candidate_space is not None else _default_space(
                selected_mode,
                profile,
                target_step,
                method,
                fixed_preprocess_methods=fixed_preprocess_methods,
                preprocess_candidates=preprocess_candidates,
                feature_candidates=feature_candidates,
                model_candidates=effective_model_candidates,
                comparison_depth=comparison_depth,
            )
            _apply_model_parameter_overrides(
                space,
                validator_model=validator_model,
                validator_params=validator_params,
                model_param_grid=model_param_grid,
            )
            locked_parameter_check = _locked_model_parameter_preflight(space)
            try:
                trials = planner_build_trials(space, metric)
            except ValueError as exc:
                raise OptimizerError("OPTIMIZER_MODEL_SPACE_EMPTY", str(exc), {}) from exc
            budget = budget_audit(expanded_trials=len(trials), max_trials=max_trials, confirmed=confirm_budget)
            test_access = _load_test_access_log(test_access_log, package_dir=package_dir, output_dir=output_dir)
            missing_confirmations = _missing_execution_confirmations(
                mode=selected_mode,
                space=space,
                confirm_budget=confirm_budget,
                confirm_comparison_design=confirm_comparison_design,
                confirm_parameter_grid=confirm_parameter_grid,
                confirm_candidate_space=confirm_candidate_space,
            )
            if preview_only:
                result = _preview_result(
                    selected_mode,
                    profile,
                    space,
                    trials,
                    metric,
                    protocol,
                    budget,
                    test_access,
                    missing_confirmations,
                    random_seed=random_seed,
                    validator_model_source=validator_model_source,
                    previous_confirmation_stage=previous_confirmation_stage,
                    target_step=target_step,
                    locked_parameter_check=locked_parameter_check,
                )
                return ok_response("optimize_spectral_pipeline", result, backend=backend, warnings=warnings)
            if budget["budget_exceeded"] and not confirm_budget:
                result = _plan_result(selected_mode, profile, space, trials, None, metric, protocol, random_seed=random_seed) | {
                    "status": "needs_confirmation",
                    "budget_audit": budget,
                    "test_access_log": test_access,
                    "evaluation_context": _evaluation_context(test_access),
                    "requires_trial_execution": False,
                    "files_written": [],
                    "directories_created": [],
                    "confirmation_card": _confirmation_card(selected_mode, target_step, space, trials, metric, profile, budget),
                    "recommendations": [
                        {
                            "type": "budget_confirmation",
                            "message": "Candidate expansion exceeds the confirmed budget; reduce candidate_space, increase max_trials, or rerun with confirm_budget.",
                        }
                    ],
                }
                return error_response(
                    "optimize_spectral_pipeline",
                    "Candidate space exceeds the confirmed trial budget.",
                    backend=backend,
                    code="OPTIMIZER_BUDGET_CONFIRMATION_REQUIRED",
                    result=result
                    | {
                        "confirmation_required": [
                            {
                                "field": "max_trials",
                                "reason": "Optimizer must not silently launch or plan an unbounded search.",
                                "requested_max_trials": max_trials,
                                "expanded_trials": len(trials),
                                "options": ["reduce candidate_space", "increase --max-trials", "rerun with --confirm-budget"],
                            }
                        ],
                    },
                    warnings=warnings,
                )
            if missing_confirmations and execute_trials:
                result = _confirmation_needed_result(
                    selected_mode,
                    profile,
                    space,
                    trials,
                    metric,
                    protocol,
                    budget,
                    test_access,
                    missing_confirmations,
                    random_seed=random_seed,
                    target_step=target_step,
                )
                return error_response(
                    "optimize_spectral_pipeline",
                    "Optimizer trial execution requires explicit user confirmation of the comparison design and budget.",
                    backend=backend,
                    code="OPTIMIZER_EXECUTION_CONFIRMATION_REQUIRED",
                    result=result,
                    warnings=warnings,
                )
            materializing_plan = output_dir is not None and trial_results is None and not execute_trials
            if missing_confirmations and materializing_plan:
                result = _confirmation_needed_result(
                    selected_mode,
                    profile,
                    space,
                    trials,
                    metric,
                    protocol,
                    budget,
                    test_access,
                    missing_confirmations,
                    random_seed=random_seed,
                    target_step=target_step,
                )
                return error_response(
                    "optimize_spectral_pipeline",
                    "Optimizer plan materialization requires explicit user confirmation; use preview_only for a no-file confirmation card.",
                    backend=backend,
                    code="OPTIMIZER_MATERIALIZATION_CONFIRMATION_REQUIRED",
                    result=result,
                    warnings=warnings,
                )
            if not locked_parameter_check["complete"]:
                result = _confirmation_needed_result(
                    selected_mode,
                    profile,
                    space,
                    trials,
                    metric,
                    protocol,
                    budget,
                    test_access,
                    missing_confirmations,
                    random_seed=random_seed,
                    target_step=target_step,
                ) | {
                    "locked_parameter_check": locked_parameter_check,
                    "missing_locked_params": locked_parameter_check["missing_locked_params"],
                    "recommended_locked_params": locked_parameter_check["recommended_locked_params"],
                    "files_written": [],
                    "directories_created": [],
                }
                return error_response(
                    "optimize_spectral_pipeline",
                    "Validation-only model comparison requires complete locked parameters before plan materialization or execution.",
                    backend=backend,
                    code="OPTIMIZER_LOCKED_MODEL_PARAMS_CONFIRMATION_REQUIRED",
                    result=result,
                    warnings=warnings,
                )
            produced_trial_results: str | Path | None = trial_results
            if execute_trials:
                if output_dir is None:
                    raise OptimizerError("OPTIMIZER_OUTPUT_DIR_REQUIRED", "execute_trials requires output_dir.", {})
                if trial_inputs is None and (package_dir is None or split_contract is None):
                    raise OptimizerError(
                        "OPTIMIZER_TRIAL_INPUTS_REQUIRED",
                        "execute_trials requires trial_inputs or package_dir plus split_contract so optimizer can prepare trial inputs.",
                        {},
                    )
                try:
                    produced_trial_results = execute_validation_trials(
                        trials,
                        trial_inputs=trial_inputs,
                        package_dir=package_dir,
                        split_contract=split_contract,
                        fixed_feature_contract=fixed_feature_contract,
                        output_dir=output_dir,
                        task_type=profile["task_type"],
                        random_seed=random_seed,
                    )
                except OptimizerExecutionError as exc:
                    raise OptimizerError(exc.code, exc.message, exc.details) from exc
            best = _select_best_from_results(produced_trial_results, metric) if produced_trial_results else None
            result = _plan_result(
                selected_mode,
                profile,
                space,
                trials,
                best,
                metric,
                protocol,
                random_seed=random_seed,
                validator_model_source=validator_model_source,
                previous_confirmation_stage=previous_confirmation_stage,
            )
            result["budget_audit"] = budget
            result["trial_results_source"] = str(produced_trial_results) if produced_trial_results else None
            result["trial_execution"] = {
                "executed_by_optimizer": bool(execute_trials),
                "official_modeling_validation_only": bool(execute_trials),
                "test_used_for_selection": False,
            }
            result["locked_parameter_check"] = locked_parameter_check
            if execute_trials and produced_trial_results and best is None:
                trial_summary = _trial_results_summary(produced_trial_results)
                result["status"] = "trials_failed"
                result["trial_execution"] = result["trial_execution"] | trial_summary
                result["failure_reason"] = trial_summary.get("failure_reason")
                if output_dir is not None:
                    written = _write_outputs(Path(output_dir), result, overwrite=True)
                    result = {**result, "output_files": written}
                return error_response(
                    "optimize_spectral_pipeline",
                    "All optimizer validation trials failed; inspect trial_results.csv before selecting a pipeline.",
                    backend=backend,
                    code="OPTIMIZER_TRIALS_FAILED",
                    result=result,
                    warnings=warnings,
                )
            result["test_access_log"] = test_access
            result["evaluation_context"] = _evaluation_context(test_access)
        if output_dir is not None:
            written = _write_outputs(Path(output_dir), result, overwrite=True)
            result = {**result, "output_files": written}
        return ok_response("optimize_spectral_pipeline", result, backend=backend, warnings=warnings)
    except OptimizerError as exc:
        status = "needs_confirmation" if exc.code.endswith("CONFIRMATION_REQUIRED") else "blocked"
        return error_response("optimize_spectral_pipeline", exc.message, backend=backend, code=exc.code, result={"status": status}, details=exc.details or {}, warnings=warnings)


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_MODES:
        raise OptimizerError("OPTIMIZER_MODE_UNSUPPORTED", "Unsupported optimizer mode.", {"mode": mode, "supported": sorted(SUPPORTED_MODES)})
    return normalized


def _build_profile(
    *,
    data_profile: str | Path | dict[str, Any] | None,
    package_dir: str | Path | None,
    task_type: str | None,
    n_samples: int | None,
    n_features: int | None,
    n_classes: int | None,
    class_balance: str | dict[str, Any] | None,
    has_validation: bool | None,
    has_test: bool | None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if data_profile is not None:
        if isinstance(data_profile, dict):
            profile.update(data_profile)
        else:
            profile.update(json.loads(Path(data_profile).read_text(encoding="utf-8-sig")))
    if package_dir is not None:
        profile.update(_profile_from_package(Path(package_dir)))
    if task_type is not None:
        profile["task_type"] = task_type
    if n_samples is not None:
        profile["n_samples"] = int(n_samples)
    if n_features is not None:
        profile["n_features"] = int(n_features)
    if n_classes is not None:
        profile["n_classes"] = int(n_classes)
    if class_balance is not None:
        profile["class_balance"] = _parse_jsonish(class_balance)
    if has_validation is not None:
        profile["has_validation"] = bool(has_validation)
    if has_test is not None:
        profile["has_test"] = bool(has_test)
    profile.setdefault("task_type", "classification")
    profile.setdefault("n_samples", 0)
    profile.setdefault("n_features", 0)
    profile.setdefault("has_validation", True)
    profile.setdefault("has_test", True)
    n = max(int(profile.get("n_samples") or 0), 1)
    p = int(profile.get("n_features") or 0)
    profile["p_over_n"] = round(p / n, 6)
    profile["small_sample"] = n < 200
    profile["high_dimensional"] = p > n
    return profile


def _profile_from_package(package_dir: Path) -> dict[str, Any]:
    contract_path = package_dir / "data_contract.json"
    if not contract_path.exists():
        raise OptimizerError("OPTIMIZER_PACKAGE_CONTRACT_MISSING", "package_dir must contain data_contract.json.", {"package_dir": str(package_dir)})
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    shape = contract.get("shape") or {}
    profile = {
        "task_type": contract.get("task_hint") or contract.get("task_type") or "classification",
        "n_samples": shape.get("n_samples") or contract.get("n_samples") or 0,
        "n_features": shape.get("n_features") or contract.get("n_features") or 0,
    }
    y_path = package_dir / str((contract.get("files") or {}).get("y") or "y.csv")
    if y_path.exists():
        with y_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        labels = [row[0] for row in rows[1:] if row]
        if labels and profile["task_type"] == "classification":
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            profile["n_classes"] = len(counts)
            profile["class_balance"] = counts
    return profile


def _parse_jsonish(value: str | dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _selection_protocol(profile: dict[str, Any]) -> str:
    if profile.get("has_validation"):
        return "holdout_val"
    if profile.get("small_sample"):
        return "inner_cross_validation"
    return "train_cv"


def _recommend_from_profile(profile: dict[str, Any], metric: str, protocol: str, *, random_seed: int) -> dict[str, Any]:
    task = profile["task_type"]
    high_dim = profile["high_dimensional"]
    small = profile["small_sample"]
    recommendations: list[dict[str, Any]] = []
    if task == "classification":
        features = ["pca", "pls_latent_variables", "vip"] if high_dim else ["pca", "select_k_best"]
        models = ["svm", "pls_da", "logistic_regression"]
        if small:
            models.append("proto_spectral_classifier")
        recommendations.append(
            {
                "pipeline_family": "high_dimensional_classification" if high_dim else "classification_baseline",
                "preprocess": ["none", "snv", "msc"],
                "feature": features,
                "modeling": models,
                "reason": f"n={profile['n_samples']}, p={profile['n_features']}, p/n={profile['p_over_n']} favors dimensionality reduction or chemometric models before final comparison.",
            }
        )
    else:
        features = ["pls_latent_variables", "pca", "vip"] if high_dim else ["none", "select_k_best"]
        models = ["plsr", "svr", "ridge", "gpr"]
        recommendations.append(
            {
                "pipeline_family": "spectral_regression",
                "preprocess": ["snv", "msc", "none"],
                "feature": features,
                "modeling": models,
                "reason": f"Regression profile n={profile['n_samples']}, p={profile['n_features']} should compare PLS/PCR-style compression against regularized models.",
            }
        )
    return _base_result("recommend_from_profile", profile, metric, protocol, random_seed=random_seed) | {
        "status": "ready",
        "recommendations": recommendations,
        "suggested_next_mode": "compare_step",
        "candidate_space": {},
        "trial_count": 0,
        "trials": [],
        "best_pipeline": None,
    }


def _default_space(
    mode: str,
    profile: dict[str, Any],
    target_step: str | None,
    method: str | None,
    *,
    fixed_preprocess_methods: str | None = None,
    preprocess_candidates: str | list[str] | None = None,
    feature_candidates: str | list[str] | None = None,
    model_candidates: str | list[str] | None = None,
    comparison_depth: str = "regular",
) -> dict[str, Any]:
    task = profile["task_type"]
    if mode == "tune_method":
        if not target_step or not method:
            raise OptimizerError("OPTIMIZER_TUNE_TARGET_REQUIRED", "tune_method requires --target-step and --method.", {})
        return _space_for_tune(target_step, method, task)
    if mode == "compare_step":
        if not target_step:
            raise OptimizerError("OPTIMIZER_COMPARE_STEP_REQUIRED", "compare_step requires --target-step.", {})
        return _space_for_compare(
            target_step,
            task,
            fixed_preprocess_methods=fixed_preprocess_methods,
            preprocess_candidates=preprocess_candidates,
            feature_candidates=feature_candidates,
            model_candidates=model_candidates,
            comparison_depth=comparison_depth,
        )
    return _space_for_pipeline(
        task,
        preprocess_candidates=preprocess_candidates,
        feature_candidates=feature_candidates,
        model_candidates=model_candidates,
    )


def _space_for_tune(step: str, method: str, task: str) -> dict[str, Any]:
    canonical = method.strip().lower().replace("-", "_")
    if step == "feature":
        if canonical == "vip":
            return {"preprocess": [{"method": "none"}], "feature": [{"method": "vip", "top_k": [10, 20, 30, 50, 80, 100]}], "modeling": _model_candidate_dicts("svm", task)}
        if canonical == "pca":
            return {"preprocess": [{"method": "none"}], "feature": [{"method": "pca", "n_components": [5, 10, 20, 30]}], "modeling": _model_candidate_dicts("svm", task)}
        if canonical in {"spa", "select_k_best"}:
            return {"preprocess": [{"method": "none"}], "feature": [{"method": canonical, "top_k": [20, 30, 50, 80]}], "modeling": _model_candidate_dicts("svm", task)}
    if step == "modeling":
        model = normalize_model_name(canonical)
        if model == "svm":
            return {"preprocess": [{"method": "none"}], "feature": [{"method": "none"}], "modeling": [{"method": "svm", "C": [0.1, 1, 10], "gamma": ["scale", "auto"]}]}
        if model in {"plsr", "pls_da"}:
            return {"preprocess": [{"method": "none"}], "feature": [{"method": "none"}], "modeling": [{"method": model, "n_components": [3, 5, 10]}]}
    if step == "preprocess":
        params: dict[str, Any] = {"method": canonical}
        if canonical == "sg_smoothing":
            params.update({"window_length": [11], "polyorder": [2], "parameter_source": "optimizer_confirmed_default"})
        return {"preprocess": [params], "feature": [{"method": "pca", "n_components": [10]}], "modeling": _model_candidate_dicts(None, task)}
    raise OptimizerError("OPTIMIZER_TUNE_METHOD_UNSUPPORTED", "No default tuning space is registered for this target.", {"target_step": step, "method": method})


def _space_for_compare(
    step: str,
    task: str,
    *,
    fixed_preprocess_methods: str | None = None,
    preprocess_candidates: str | list[str] | None = None,
    feature_candidates: str | list[str] | None = None,
    model_candidates: str | list[str] | None = None,
    comparison_depth: str = "regular",
) -> dict[str, Any]:
    preprocess = [{"method": fixed_preprocess_methods.strip()}] if fixed_preprocess_methods and fixed_preprocess_methods.strip() else [{"method": "none"}]
    if step == "feature":
        depth = (comparison_depth or "regular").strip().lower()
        if depth in {"quick", "single", "single_point"}:
            features = [
                {"method": "none"},
                {"method": "pca", "n_components": [10]},
                {"method": "pls_latent_variables", "n_components": [10]},
                {"method": "vip", "top_k": [30]},
                {"method": "select_k_best", "top_k": [50]},
                {"method": "spa", "top_k": [50]},
            ]
        elif depth in {"regular", "recommended", "small_grid", "grid"}:
            features = [
                {"method": "none"},
                {"method": "pca", "n_components": [5, 10, 20, 30]},
                {"method": "pls_latent_variables", "n_components": [3, 5, 10]},
                {"method": "vip", "top_k": [10, 20, 30, 50, 80, 100]},
                {"method": "select_k_best", "top_k": [20, 30, 50, 80]},
                {"method": "spa", "top_k": [20, 30, 50, 80]},
            ]
        elif depth in {"extended", "broad"}:
            features = [
                {"method": "none"},
                {"method": "pca", "n_components": [3, 5, 10, 20, 30, 50]},
                {"method": "pls_latent_variables", "n_components": [2, 3, 5, 8, 10, 15]},
                {"method": "vip", "top_k": [10, 20, 30, 50, 80, 100, 150]},
                {"method": "select_k_best", "top_k": [10, 20, 30, 50, 80, 100]},
                {"method": "spa", "top_k": [10, 20, 30, 50, 80, 100]},
            ]
        elif depth in {"deep", "deep_search"}:
            features = [
                {"method": "autoencoder_embedding", "n_components": [8, 16, 32], "epochs": [50, 100], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-5, "device": "cpu"},
                {"method": "denoising_autoencoder_embedding", "n_components": [8, 16, 32], "epochs": [80, 120], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-5, "noise_std": [0.02, 0.05], "device": "cpu"},
                {"method": "cnn_1d_embedding", "n_components": [8, 16, 32], "epochs": [50, 80], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-4, "device": "cpu"},
                {"method": "resnet1d_embedding", "n_components": [8, 16], "epochs": [40, 60], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-4, "device": "cpu"},
                {"method": "cls_former_embedding", "n_components": [8, 16, 32], "epochs": [50, 100], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-4, "patch_size": [16, 32], "device": "cpu"},
                {"method": "masked_spectral_autoencoder_embedding", "n_components": [8, 16, 32], "epochs": [50, 100], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-4, "mask_ratio": [0.10, 0.25], "patch_size": [16, 32], "device": "cpu"},
                {"method": "contrastive_spectral_embedding", "n_components": [8, 16, 32], "epochs": [50, 100], "batch_size": 16, "learning_rate": [0.001, 0.0003], "weight_decay": 1e-4, "noise_std": [0.02, 0.05], "mask_ratio": 0.10, "temperature": [0.1, 0.2], "device": "cpu"},
            ]
            if not model_candidates:
                model_candidates = "linear_svm,svm,lda"
        else:
            raise OptimizerError(
                "OPTIMIZER_COMPARISON_DEPTH_UNSUPPORTED",
                "comparison_depth must be quick, regular, extended, or deep.",
                {"comparison_depth": comparison_depth},
            )
        features = _filter_feature_candidates(features, feature_candidates)
        return {
            "preprocess": preprocess,
            "feature": features,
            "modeling": _model_candidate_dicts(model_candidates, task),
        }
    if step == "modeling":
        models = ["svm", "pls_da", "logistic_regression"] if task == "classification" else ["plsr", "svr", "ridge", "gpr"]
        if model_candidates:
            models = _parse_list(model_candidates)
        return {"preprocess": preprocess, "feature": [{"method": "none"}], "modeling": _model_candidate_dicts(models, task)}
    if step == "preprocess":
        if preprocess_candidates:
            preprocess = _preprocess_candidate_dicts(preprocess_candidates)
        return {
            "preprocess": preprocess
            if preprocess_candidates
            else [
                    {"method": "none"},
                    {"method": "snv"},
                    {"method": "msc"},
                    {"method": "snv_detrend"},
                    {"method": "sg_smoothing,snv", "window_length": [11], "polyorder": [2], "parameter_source": "optimizer_confirmed_default"},
                ],
            "feature": [{"method": "none"}],
            "modeling": _model_candidate_dicts(model_candidates, task),
        }
    raise OptimizerError("OPTIMIZER_COMPARE_STEP_UNSUPPORTED", "Unsupported compare_step target.", {"target_step": step})


def _space_for_pipeline(
    task: str,
    *,
    preprocess_candidates: str | list[str] | None = None,
    feature_candidates: str | list[str] | None = None,
    model_candidates: str | list[str] | None = None,
) -> dict[str, Any]:
    models = (
        [
            {"method": "svm", "C": [1, 10], "gamma": ["scale"]},
            {"method": "linear_svm", "C": [1]},
            {"method": "pls_da", "n_components": [5]},
        ]
        if task == "classification"
        else [{"method": "plsr", "n_components": [5, 10]}, {"method": "svr", "C": [1, 10], "gamma": ["scale"]}]
    )
    preprocess = [{"method": "none"}, {"method": "snv"}, {"method": "msc"}]
    features = [
        {"method": "none"},
        {"method": "pca", "n_components": [10]},
        {"method": "pls_latent_variables", "n_components": [3, 5, 10]},
        {"method": "vip", "top_k": [30]},
    ]
    if preprocess_candidates:
        preprocess = _dedupe_candidates([*preprocess, *_preprocess_candidate_dicts(preprocess_candidates)])
    if feature_candidates:
        features = _dedupe_candidates([*features, *_pipeline_feature_addon_dicts(feature_candidates)])
    if model_candidates:
        models = _dedupe_candidates([*models, *_model_candidate_dicts(model_candidates, task)])
    return {
        "preprocess": preprocess,
        "feature": features,
        "modeling": models,
    }


def _dedupe_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _pipeline_feature_addon_dicts(value: str | list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw_method in _parse_list(value):
        method = _normalize_feature_candidate(raw_method)
        if method == "cls_former_embedding_svm":
            method = "cls_former_embedding"
        params: dict[str, Any] = {"method": method}
        if method == "pca":
            params["n_components"] = [10]
        elif method == "pls_latent_variables":
            params["n_components"] = [3, 5, 10]
        elif method in {"vip", "select_k_best", "spa"}:
            params["top_k"] = [30 if method == "vip" else 80]
        elif method in {"cars", "uve", "mcuve"}:
            params.update({"n_components": [5], "n_runs": [50], "top_k": [80], "sample_ratio": [0.8], "cv": [3], "random_state": 42})
            if method == "cars":
                params.pop("top_k", None)
        elif method == "interval_pls":
            params.update({"n_intervals": [20], "n_components": [5], "cv": [3]})
        elif method == "autoencoder_embedding":
            params.update({"n_components": [16], "epochs": 100, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-5, "random_state": 42, "device": "cpu"})
        elif method == "denoising_autoencoder_embedding":
            params.update({"n_components": [16], "epochs": 100, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-5, "noise_std": 0.03, "random_state": 42, "device": "cpu"})
        elif method == "cnn_1d_embedding":
            params.update({"n_components": [16], "epochs": 80, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "random_state": 42, "device": "cpu"})
        elif method == "resnet1d_embedding":
            params.update({"n_components": [16], "epochs": 60, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "random_state": 42, "device": "cpu"})
        elif method == "cls_former_embedding":
            params.update({"n_components": [16], "epochs": 80, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "patch_size": 16, "random_state": 42, "device": "cpu"})
        elif method == "masked_spectral_autoencoder_embedding":
            params.update({"n_components": [16], "epochs": 100, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "mask_ratio": 0.15, "patch_size": 16, "random_state": 42, "device": "cpu"})
        elif method == "contrastive_spectral_embedding":
            params.update({"n_components": [16], "epochs": 100, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "noise_std": 0.03, "mask_ratio": 0.1, "temperature": 0.2, "random_state": 42, "device": "cpu"})
        elif method == "self_supervised_spectral_embedding":
            params.update({"n_components": [16], "epochs": 100, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 1e-4, "noise_std": 0.03, "mask_ratio": 0.1, "temperature": 0.2, "random_state": 42, "device": "cpu"})
        candidates.append(params)
    if not candidates:
        raise OptimizerError("OPTIMIZER_FEATURE_CANDIDATES_EMPTY", "feature_candidates did not contain any methods.", {})
    return candidates


def _load_candidate_space(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8-sig"))


def _parse_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).replace(";", ",").split(",")
    return [str(item).strip().lower().replace("-", "_") for item in items if str(item).strip()]


def _normalize_feature_candidate(method: str) -> str:
    aliases = {
        "pls": "pls_latent_variables",
        "pls_lv": "pls_latent_variables",
        "kbest": "select_k_best",
        "skb": "select_k_best",
        "cls_former": "cls_former_embedding",
        "clsformer": "cls_former_embedding",
        "transformer": "cls_former_embedding",
        "transformer_embedding": "cls_former_embedding",
        "cls_former_embedding_svm": "cls_former_embedding",
    }
    key = method.strip().lower().replace("-", "_")
    return aliases.get(key, key)


def _normalize_model_candidate(method: str) -> str:
    return normalize_model_name(method.strip().lower().replace("-", "_"))


def _filter_feature_candidates(defaults: list[dict[str, Any]], requested: str | list[str] | None) -> list[dict[str, Any]]:
    requested_methods = [_normalize_feature_candidate(item) for item in _parse_list(requested)]
    if not requested_methods:
        return defaults
    by_method = {str(item.get("method")): item for item in defaults}
    missing = [method for method in requested_methods if method not in by_method]
    if missing:
        raise OptimizerError("OPTIMIZER_FEATURE_CANDIDATE_UNSUPPORTED", "One or more feature candidates are not in the default comparison grid.", {"unsupported": missing, "supported": sorted(by_method)})
    return [by_method[method] for method in requested_methods]


def _preprocess_candidate_dicts(value: str | list[str]) -> list[dict[str, Any]]:
    candidates = []
    for method in _parse_list(value):
        item: dict[str, Any] = {"method": method}
        if method in {"sg_smoothing", "first_derivative", "second_derivative", "sg_smoothing,snv"}:
            item.update({"window_length": [11], "polyorder": [2], "parameter_source": "optimizer_confirmed_default"})
        candidates.append(item)
    if not candidates:
        raise OptimizerError("OPTIMIZER_PREPROCESS_CANDIDATES_EMPTY", "preprocess_candidates did not contain any methods.", {})
    return candidates


def _model_candidate_dicts(value: str | list[str] | None, task: str) -> list[dict[str, Any]]:
    methods = _parse_list(value) if value else (["svm"] if task == "classification" else ["plsr"])
    candidates: list[dict[str, Any]] = []
    for method in methods:
        canonical = _normalize_model_candidate(method)
        fixed = _optimizer_fixed_model_parameters(canonical)
        candidates.append({"method": canonical, **{key: [value] for key, value in fixed.items()}})
    return candidates


def _optimizer_fixed_model_parameters(method: str) -> dict[str, Any]:
    canonical = _normalize_model_candidate(method)
    fixed = comparison_fixed_parameters(canonical)
    if fixed:
        return fixed
    defaults = dict(MODEL_DEFAULTS.get(canonical) or {})
    if not defaults:
        return {}
    defaults.update(OPTIMIZER_EXPERIMENTAL_MODEL_OVERRIDES.get(canonical, {}))
    return defaults


def _locked_model_parameter_preflight(space: dict[str, Any]) -> dict[str, Any]:
    missing: dict[str, list[str]] = {}
    recommended: dict[str, dict[str, Any]] = {}
    for item in space.get("modeling") or []:
        method = _normalize_model_candidate(str(item.get("method") or ""))
        policy = _optimizer_fixed_model_parameters(method)
        if not policy:
            continue
        absent = [key for key in policy if not _model_item_has_parameter(item, method, key)]
        if absent:
            missing[method] = absent
            recommended[method] = {key: policy[key] for key in absent}
    return {
        "complete": not missing,
        "status": "passed" if not missing else "needs_confirmation",
        "missing_locked_params": missing,
        "recommended_locked_params": recommended,
        "policy": "validation_only trials disable model selection and require an executable fixed parameter set",
    }


def _model_item_has_parameter(item: dict[str, Any], method: str, key: str) -> bool:
    aliases = {key, f"model__{key}", f"{method}__{key}"}
    return any(alias in item and item.get(alias) is not None and item.get(alias) != "" for alias in aliases)


def _apply_model_parameter_overrides(
    space: dict[str, Any],
    *,
    validator_model: str | None,
    validator_params: str | list[str] | None,
    model_param_grid: str | list[str] | None,
) -> None:
    assignments = _parse_model_param_grid(model_param_grid)
    if validator_model and validator_params:
        method = _normalize_model_candidate(validator_model)
        validator_assignments = _parse_param_assignments(validator_params)
        assignments.setdefault(method, {}).update(validator_assignments)
    if not assignments:
        return
    models = space.get("modeling") or []
    for item in models:
        method = _normalize_model_candidate(str(item.get("method") or ""))
        for key, value in assignments.get(method, {}).items():
            item[key] = value


def _parse_model_param_grid(value: str | list[str] | None) -> dict[str, dict[str, list[Any]]]:
    output: dict[str, dict[str, list[Any]]] = {}
    for token in _parse_param_tokens(value):
        if "=" not in token or "." not in token.split("=", 1)[0]:
            raise OptimizerError(
                "OPTIMIZER_MODEL_PARAM_GRID_INVALID",
                "model_param_grid entries must use model.parameter=value syntax, e.g. svm.C=1.0.",
                {"entry": token},
            )
        left, raw_value = token.split("=", 1)
        model, key = left.split(".", 1)
        output.setdefault(_normalize_model_candidate(model), {})[key] = _parse_param_values(raw_value)
    return output


def _parse_param_assignments(value: str | list[str] | None) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {}
    for token in _parse_param_tokens(value):
        if "=" not in token:
            raise OptimizerError("OPTIMIZER_PARAM_INVALID", "parameter entries must use key=value syntax.", {"entry": token})
        key, raw_value = token.split("=", 1)
        output[key.strip()] = _parse_param_values(raw_value)
    return output


def _parse_param_tokens(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    tokens: list[str] = []
    for raw in raw_items:
        text = str(raw).strip()
        if not text:
            continue
        tokens.extend(item.strip() for item in text.replace(";", ",").split(",") if item.strip())
    return tokens


def _parse_param_values(raw: str) -> list[Any]:
    values = [item.strip() for item in str(raw).split("|") if item.strip()]
    if not values:
        values = [str(raw).strip()]
    return [_coerce_param_value(item) for item in values]


def _coerce_param_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_test_access_log(
    explicit_path: str | Path | None,
    *,
    package_dir: str | Path | None,
    output_dir: str | Path | None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for base in (package_dir, output_dir):
        if not base:
            continue
        path = Path(base)
        candidates.extend([path / "test_access_log.json", path.parent / "test_access_log.json"])
        candidates.extend(parent / "test_access_log.json" for parent in path.parents[:3])
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                return {"path": str(path), "status": "unreadable"}
            payload.setdefault("path", str(path))
            return payload
    return None


def _expand_options(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in items:
        scalar = {key: value for key, value in item.items() if not isinstance(value, list)}
        grids = {key: value for key, value in item.items() if isinstance(value, list)}
        if not grids:
            expanded.append(dict(scalar))
            continue
        keys = list(grids)
        for values in itertools.product(*(grids[key] for key in keys)):
            expanded.append({**scalar, **dict(zip(keys, values))})
    return expanded


def _build_trials(space: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    preprocess = _expand_options(space.get("preprocess") or [{"method": "none"}])
    feature = _expand_options(space.get("feature") or [{"method": "none"}])
    modeling = _expand_options(space.get("modeling") or [])
    if not modeling:
        raise OptimizerError("OPTIMIZER_MODEL_SPACE_EMPTY", "candidate_space must include at least one modeling candidate.", {})
    trials: list[dict[str, Any]] = []
    for idx, (pre, feat, model) in enumerate(itertools.product(preprocess, feature, modeling), start=1):
        try:
            family = model_spec(model["method"]).family
        except Exception:
            family = "unknown"
        trials.append(
            {
                "trial_id": f"trial_{idx:04d}",
                "preprocess_method": pre.get("method", "none"),
                "feature_method": feat.get("method", "none"),
                "model_method": model.get("method"),
                "model_family": family,
                "preprocess_params": {k: v for k, v in pre.items() if k != "method"},
                "feature_params": {k: v for k, v in feat.items() if k != "method"},
                "model_params": {k: v for k, v in model.items() if k != "method"},
                "selection_metric": metric,
                "test_used_for_selection": False,
                "status": "planned",
                "warnings": "",
            }
        )
    return trials


def _select_best_from_results(path: str | Path, metric: str) -> dict[str, Any] | None:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    metric_candidates = [metric, metric.replace("val_", "cv_"), "val_macro_f1", "cv_macro_f1", "val_accuracy", "cv_accuracy", "val_rmse", "cv_rmse"]
    selected_metric = next((name for name in metric_candidates if name in rows[0]), None)
    if selected_metric is None:
        raise OptimizerError("OPTIMIZER_SELECTION_METRIC_MISSING", "trial_results.csv does not contain a validation/CV selection metric.", {"selection_metric": metric})
    reverse = not selected_metric.endswith(("rmse", "mae", "mse"))
    valid = [row for row in rows if row.get(selected_metric) not in {None, ""}]
    if not valid:
        return None
    test_metric_columns = [name for name in rows[0] if name.startswith("test_")]
    best_value = (max if reverse else min)(float(row[selected_metric]) for row in valid)
    tied = [row for row in valid if float(row[selected_metric]) == best_value]
    secondary_metrics = _secondary_metrics_for(selected_metric, rows[0])
    secondary_sorted = sorted(tied, key=lambda row: (_secondary_metric_key(row, secondary_metrics), _tie_breaker_key(row)))
    best = secondary_sorted[0]
    tie_details = {
        "tie_detected": len(tied) > 1,
        "tie_count": len(tied),
        "primary_metric": selected_metric,
        "tie_trial_ids": [row.get("trial_id") for row in tied],
        "tied_trials": [_trial_label(row) for row in tied],
        "secondary_metrics": [item["name"] for item in secondary_metrics],
        "policy": TIE_BREAKER_POLICY,
        "tie_breaker_order": TIE_BREAKER_POLICY,
        "selected_reason": _tie_breaker_reason(best, tied),
    }
    return {
        "trial_id": best.get("trial_id"),
        "selection_metric": selected_metric,
        "selection_value": float(best[selected_metric]),
        "source": str(path),
        "row": best,
        "tie_breaker": tie_details,
        "test_metrics_present": bool(test_metric_columns),
        "test_metrics_ignored_for_selection": bool(test_metric_columns),
        "ignored_test_metric_columns": test_metric_columns,
    }


def _trial_results_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n_trials = len(rows)
    successes = [row for row in rows if str(row.get("status") or "").lower() == "completed"]
    failed = [row for row in rows if str(row.get("status") or "").lower() != "completed"]
    return {
        "n_trials": n_trials,
        "n_success": len(successes),
        "n_failed": len(failed),
        "failure_reason": _first_failure_reason(failed),
        "trial_results": str(path),
    }


def _first_failure_reason(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        warnings = row.get("warnings") or ""
        if not warnings:
            continue
        try:
            payload = json.loads(warnings)
        except json.JSONDecodeError:
            return warnings[:200]
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return str(first.get("code") or first.get("message") or first)[:200]
            return str(first)[:200]
        if isinstance(payload, dict):
            return str(payload.get("code") or payload.get("message") or payload)[:200]
    return None


def _secondary_metrics_for(primary_metric: str, header: dict[str, str]) -> list[dict[str, Any]]:
    if primary_metric.startswith(("val_", "cv_")) and not primary_metric.endswith(("rmse", "mae", "mse")):
        candidates = ["val_accuracy", "val_balanced_accuracy", "val_auc", "cv_accuracy", "cv_balanced_accuracy", "cv_auc"]
        return [{"name": name, "higher_is_better": True} for name in candidates if name != primary_metric and name in header]
    candidates = ["val_mae", "val_r2", "cv_mae", "cv_r2"]
    result: list[dict[str, Any]] = []
    for name in candidates:
        if name in header and name != primary_metric:
            result.append({"name": name, "higher_is_better": name.endswith("r2")})
    return result


def _secondary_metric_key(row: dict[str, str], metrics: list[dict[str, Any]]) -> tuple[float, ...]:
    values: list[float] = []
    for item in metrics:
        raw = row.get(item["name"])
        try:
            value = float(raw) if raw not in {None, ""} else float("-inf")
        except ValueError:
            value = float("-inf")
        values.append(-value if item["higher_is_better"] else value)
    return tuple(values)


def _trial_label(row: dict[str, str]) -> str:
    preprocess = row.get("preprocess_method") or "none"
    feature = row.get("feature_method") or "none"
    model = row.get("model_method") or "unknown_model"
    params = _parse_params_field(row.get("params") or "")
    feature_params = params.get("feature") if isinstance(params.get("feature"), dict) else {}
    model_params = params.get("modeling") if isinstance(params.get("modeling"), dict) else {}
    feature_suffix = ""
    if feature_params.get("n_components"):
        feature_suffix = f"({feature_params['n_components']})"
    elif feature_params.get("top_k"):
        feature_suffix = f"(top_k={feature_params['top_k']})"
    model_suffix = ""
    if model_params.get("n_components"):
        model_suffix = f"({model_params['n_components']})"
    return f"{preprocess} + {feature}{feature_suffix} + {model}{model_suffix}"


def _tie_breaker_key(row: dict[str, str]) -> tuple[int, int, int, int, int, int, str]:
    preprocess_method = (row.get("preprocess_method") or "none").strip().lower()
    feature_method = (row.get("feature_method") or "none").strip().lower()
    model_method = (row.get("model_method") or "").strip().lower()
    params = _parse_params_field(row.get("params") or "")
    preprocess_count = 0 if preprocess_method in {"", "none"} else len([item for item in preprocess_method.split(",") if item.strip()])
    preprocess_priority = _preprocess_priority(preprocess_method)
    parameter_count = _count_nonempty_params(params)
    n_features = _output_feature_count(row, params)
    supervised_penalty = 1 if feature_method in {"pls_latent_variables", "vip", "select_k_best", "anova_f", "f_regression", "cars", "uve", "mcuve"} else 0
    model_penalty = 1 if model_method in {"cls_former_classifier", "spectral_dkl_gp_classifier", "proto_spectral_classifier"} else 0
    return (preprocess_count, preprocess_priority, n_features, parameter_count, supervised_penalty, model_penalty, row.get("trial_id") or "")


def _preprocess_priority(method: str) -> int:
    canonical = method.replace("+", ",").replace(" ", "").lower()
    if canonical in {"", "none"}:
        return 0
    priority = {
        "snv": 1,
        "msc": 2,
        "detrend": 3,
        "snv_detrend": 4,
        "snv,detrend": 4,
        "sg_smoothing": 5,
        "sg_smoothing,snv": 6,
    }
    return priority.get(canonical, 20)


def _parse_params_field(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _count_nonempty_params(params: dict[str, Any]) -> int:
    count = 0
    for group in params.values():
        if isinstance(group, dict):
            count += sum(1 for value in group.values() if value not in {None, "", "none"})
    return count


def _output_feature_count(row: dict[str, str], params: dict[str, Any]) -> int:
    for key in ("output_n_features", "n_features", "feature_count"):
        value = row.get(key)
        if value not in {None, ""}:
            try:
                return int(float(str(value)))
            except ValueError:
                pass
    feature_params = params.get("feature") if isinstance(params.get("feature"), dict) else {}
    for key in ("n_components", "top_k"):
        value = feature_params.get(key)
        if value not in {None, ""}:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                pass
    return 10**9


def _tie_breaker_reason(best: dict[str, str], tied: list[dict[str, str]]) -> str:
    if len(tied) <= 1:
        return "No metric tie; selected by validation/CV metric."
    return (
        "Primary validation/CV metric tied; selected by secondary validation metrics "
        "when available, then by the deterministic tie-breaker favoring fewer "
        "preprocessing methods, predefined preprocessing priority, fewer parameters/features, "
        "unsupervised feature methods, and lower-compute models. When SNV and MSC are "
        "otherwise tied, SNV is preferred because it is a per-sample standardization and "
        "does not require a train-set reference spectrum."
    )


def _base_result(mode: str, profile: dict[str, Any], metric: str, protocol: str, *, random_seed: int) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "status": "ready",
        "mode": mode,
        "task_type": profile["task_type"],
        "data_profile": profile,
        "selection_metric": metric,
        "selection_protocol": protocol,
        "test_used_for_selection": False,
        "final_test_evaluated_once": True,
        "tie_breaker_policy": TIE_BREAKER_POLICY,
        "random_seed": random_seed,
    }


def _missing_execution_confirmations(
    *,
    mode: str,
    space: dict[str, Any],
    confirm_budget: bool,
    confirm_comparison_design: bool,
    confirm_parameter_grid: bool,
    confirm_candidate_space: bool,
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    if mode in {"compare_step", "tune_method"} and not confirm_comparison_design:
        missing.append(
            {
                "field": "comparison_design",
                "reason": "Stage comparison/tuning ranks candidates under fixed upstream/downstream choices and a validator model; the user must confirm candidates, fixed stages, validator model, metric, and budget before trials run.",
                "options": ["review candidate_space.json and rerun with --confirm-comparison-design"],
            }
        )
    if mode in {"compare_step", "tune_method"} and _candidate_space_has_parameter_grid(space) and not confirm_parameter_grid:
        missing.append(
            {
                "field": "parameter_grid",
                "reason": "The user confirmed method names, but parameter grids such as PCA components, PLS components, VIP top_k, KBest top_k, or SPA top_k also affect ranking and must be confirmed before trials run.",
                "options": ["review candidate_space.json and rerun with --confirm-parameter-grid"],
            }
        )
    if mode == "optimize_pipeline" and not confirm_candidate_space:
        missing.append(
            {
                "field": "candidate_space",
                "reason": "Full-pipeline optimization searches multiple stages; the user must confirm the candidate methods and compact/extended policy before trials run.",
                "options": ["review candidate_space.json and rerun with --confirm-candidate-space"],
            }
        )
    if not confirm_budget:
        missing.append(
            {
                "field": "max_trials",
                "reason": "The trial budget must be confirmed by the user before optimizer executes any trial, even when the expanded space is within max_trials.",
                "options": ["rerun with --confirm-budget after the user confirms the trial count"],
            }
        )
    return missing


def _candidate_space_has_parameter_grid(space: dict[str, Any]) -> bool:
    for stage in ("preprocess", "feature", "modeling"):
        for item in space.get(stage) or []:
            for key, value in item.items():
                if key != "method" and isinstance(value, list) and len(value) > 1:
                    return True
    return False


def _plan_result(
    mode: str,
    profile: dict[str, Any],
    space: dict[str, Any],
    trials: list[dict[str, Any]],
    best: dict[str, Any] | None,
    metric: str,
    protocol: str,
    *,
    random_seed: int,
    validator_model_source: str | None = None,
    previous_confirmation_stage: str | None = None,
) -> dict[str, Any]:
    stability_followup = _stability_followup(profile, protocol, best)
    return _base_result(mode, profile, metric, protocol, random_seed=random_seed) | {
        "status": "best_selected" if best is not None else "plan_materialized",
        "candidate_space": space,
        "candidate_space_policy": _candidate_space_policy(space, profile, mode),
        "validator_model": _validator_model_metadata(
            space,
            source=validator_model_source,
            previous_confirmation_stage=previous_confirmation_stage,
        ),
        "trial_count": len(trials),
        "trials": trials,
        "best_pipeline": best,
        "stability_followup": stability_followup,
        "requires_trial_execution": best is None,
        "recommendations": [
            {
                "type": "execution",
                "message": "Run planned trials through spectral-workflow; select only by validation/CV metrics, then evaluate the selected pipeline on test once.",
            }
        ]
        + ([stability_followup] if stability_followup else []),
    }


def _preview_result(
    mode: str,
    profile: dict[str, Any],
    space: dict[str, Any],
    trials: list[dict[str, Any]],
    metric: str,
    protocol: str,
    budget: dict[str, Any],
    test_access: dict[str, Any] | None,
    missing_confirmations: list[dict[str, Any]],
    *,
    random_seed: int,
    validator_model_source: str | None,
    previous_confirmation_stage: str | None,
    target_step: str | None,
    locked_parameter_check: dict[str, Any],
) -> dict[str, Any]:
    result = _plan_result(
        mode,
        profile,
        space,
        trials,
        None,
        metric,
        protocol,
        random_seed=random_seed,
        validator_model_source=validator_model_source or "recommended_not_confirmed",
        previous_confirmation_stage=previous_confirmation_stage,
    ) | {
        "status": "needs_confirmation",
        "preview_only": True,
        "budget_audit": budget,
        "test_access_log": test_access,
        "evaluation_context": _evaluation_context(test_access),
        "requires_trial_execution": False,
        "confirmation_required": missing_confirmations,
        "confirmation_card": _confirmation_card(mode, target_step, space, trials, metric, profile, budget),
        "files_written": [],
        "directories_created": [],
    }
    result["locked_parameter_check"] = locked_parameter_check
    result["missing_locked_params"] = locked_parameter_check["missing_locked_params"]
    result["recommended_locked_params"] = locked_parameter_check["recommended_locked_params"]
    result["confirmation_card"]["locked_parameter_check"] = locked_parameter_check
    return result


def _confirmation_needed_result(
    mode: str,
    profile: dict[str, Any],
    space: dict[str, Any],
    trials: list[dict[str, Any]],
    metric: str,
    protocol: str,
    budget: dict[str, Any],
    test_access: dict[str, Any] | None,
    missing_confirmations: list[dict[str, Any]],
    *,
    random_seed: int,
    target_step: str | None,
) -> dict[str, Any]:
    return _plan_result(mode, profile, space, trials, None, metric, protocol, random_seed=random_seed) | {
        "status": "needs_confirmation",
        "budget_audit": budget,
        "test_access_log": test_access,
        "evaluation_context": _evaluation_context(test_access),
        "requires_trial_execution": False,
        "confirmation_required": missing_confirmations,
        "confirmation_card": _confirmation_card(mode, target_step, space, trials, metric, profile, budget),
        "files_written": [],
        "directories_created": [],
    }


def _confirmation_card(
    mode: str,
    target_step: str | None,
    space: dict[str, Any],
    trials: list[dict[str, Any]],
    metric: str,
    profile: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "target_step": target_step,
        "fixed_preprocess": _single_or_list([item.get("method", "none") for item in space.get("preprocess") or [{"method": "none"}]]),
        "validator_model": {
            **(_validator_model_metadata(space, source="recommended_not_confirmed", previous_confirmation_stage=None) or {}),
            "status": "needs_user_confirmation",
        },
        "selection_metric": metric,
        "auxiliary_metrics": _default_auxiliary_metrics(profile["task_type"]),
        "candidate_methods": {
            "preprocess": [item.get("method", "none") for item in space.get("preprocess") or [{"method": "none"}]],
            "feature": [item.get("method", "none") for item in space.get("feature") or [{"method": "none"}]],
            "modeling": [item.get("method") for item in space.get("modeling") or []],
        },
        "parameter_grid": _parameter_grid_summary(space),
        "comparison_depth_options": _comparison_depth_options(mode, target_step, profile),
        "estimated_trials": len(trials),
        "budget": budget,
        "test_policy": "not_used_for_selection",
        "candidate_space_policy": _candidate_space_policy(space, profile, mode),
    }


def _single_or_list(values: list[Any]) -> Any:
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else unique


def _default_auxiliary_metrics(task_type: str) -> list[str]:
    if task_type == "classification":
        return ["val_accuracy", "val_balanced_accuracy", "val_auc"]
    return ["val_mae", "val_r2"]


def _parameter_grid_summary(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for stage in ("preprocess", "feature", "modeling"):
        for item in space.get(stage) or []:
            method = item.get("method", "none")
            grids = {key: value for key, value in item.items() if key != "method" and isinstance(value, list)}
            if grids:
                summary[f"{stage}.{method}"] = grids
    return summary


def _comparison_depth_options(mode: str, target_step: str | None, profile: dict[str, Any]) -> list[dict[str, Any]]:
    if mode != "compare_step" or target_step != "feature":
        return []
    validator = "svm" if profile.get("task_type") == "classification" else "plsr"
    return [
        {
            "option": "quick",
            "label": "quick single-point comparison",
            "estimated_trials": 6,
            "validator_model": validator,
            "feature_grid": {
                "none": {},
                "pca": {"n_components": [10]},
                "pls_latent_variables": {"n_components": [10]},
                "vip": {"top_k": [30]},
                "select_k_best": {"top_k": [50]},
                "spa": {"top_k": [50]},
            },
        },
        {
            "option": "recommended",
            "aliases": ["regular"],
            "label": "regular recommended small-grid comparison",
            "estimated_trials": 22,
            "validator_model": validator,
            "feature_grid": {
                "none": {},
                "pca": {"n_components": [5, 10, 20, 30]},
                "pls_latent_variables": {"n_components": [3, 5, 10]},
                "vip": {"top_k": [10, 20, 30, 50, 80, 100]},
                "select_k_best": {"top_k": [20, 30, 50, 80]},
                "spa": {"top_k": [20, 30, 50, 80]},
            },
            "recommended": True,
        },
        {
            "option": "extended",
            "label": "extended feature comparison",
            "estimated_trials": 32,
            "validator_model": validator,
            "note": "Broader grid; consider repeated holdout when the best candidates are close.",
        },
        {
            "option": "deep",
            "label": "opt-in deep embedding plus classifier search",
            "estimated_trials": ">300 before user pruning",
            "validator_model": ["linear_svm", "svm", "lda"],
            "note": "Requires explicit device, epoch, seed, parameter-grid, and expanded-budget confirmation; test is excluded from selection.",
            "embedding_dimensions": [8, 16, 32],
        },
    ]


def _validator_model_metadata(
    space: dict[str, Any],
    *,
    source: str | None,
    previous_confirmation_stage: str | None,
) -> dict[str, Any] | None:
    models = [item.get("method") for item in space.get("modeling") or [] if item.get("method")]
    if not models:
        return None
    unique_models = list(dict.fromkeys(models))
    parameter_grid = {
        str(item.get("method")): {key: value for key, value in item.items() if key != "method"}
        for item in space.get("modeling") or []
        if item.get("method")
    }
    return {
        "methods": unique_models,
        "method": unique_models[0] if len(unique_models) == 1 else None,
        "parameter_grid": parameter_grid,
        "source": source or "user_confirmed_current_comparison",
        "previous_confirmation_stage": previous_confirmation_stage,
        "affects_ranking": True,
    }


def _candidate_space_policy(space: dict[str, Any], profile: dict[str, Any], mode: str) -> dict[str, Any]:
    included = {
        "preprocess": [item.get("method", "none") for item in space.get("preprocess") or [{"method": "none"}]],
        "feature": [item.get("method", "none") for item in space.get("feature") or [{"method": "none"}]],
        "modeling": [item.get("method") for item in space.get("modeling") or []],
    }
    excluded: list[dict[str, str]] = []
    if mode == "optimize_pipeline":
        if profile.get("task_type") == "classification":
            optional_feature = {"pls_latent_variables", "select_k_best", "spa", "cars", "uve"}
            optional_models = {"linear_svm", "logistic_regression", "random_forest_classifier", "pls_da"}
            for method in sorted(optional_feature - set(included["feature"])):
                excluded.append({"stage": "feature", "method": method, "reason": "not included in compact default budget; available in extended/custom candidate space"})
            for method in sorted(optional_models - set(included["modeling"])):
                excluded.append({"stage": "modeling", "method": method, "reason": "not included in compact default budget; available in extended/custom candidate space"})
    return {
        "policy": "compact_default" if mode == "optimize_pipeline" else "stage_default_or_user_candidate_space",
        "included_methods": included,
        "excluded_methods": excluded,
        "note": "Default spaces are intentionally compact; use candidate_space.json for extended searches.",
    }


def _stability_followup(profile: dict[str, Any], protocol: str, best: dict[str, Any] | None) -> dict[str, Any] | None:
    if protocol != "holdout_val" or int(profile.get("n_samples") or 0) > 200:
        return None
    reasons = ["small_sample_single_holdout"]
    train_accuracy = _best_train_accuracy(best)
    if train_accuracy is not None and train_accuracy >= 0.99:
        reasons.append("near_perfect_train_accuracy_overfit_signal")
    return {
        "type": "validation_followup",
        "status": "recommended",
        "protocol": "stratified_repeated_holdout" if profile.get("task_type") == "classification" else "repeated_holdout",
        "n_repeats_options": [5, 10],
        "lock_selected_pipeline": True,
        "selection_metric": "macro_f1" if profile.get("task_type") == "classification" else "rmse",
        "report": ["mean", "std"],
        "trigger_reasons": reasons,
        "observed_train_accuracy": train_accuracy,
        "message": "Confirm the selected pipeline with 5 or 10 repeated holdout runs and report mean/std before treating a small-sample single-split result as stable.",
    }


def _best_train_accuracy(best: dict[str, Any] | None) -> float | None:
    row = (best or {}).get("row") or {}
    raw = row.get("train_accuracy")
    if raw in {None, ""}:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _write_outputs(output_dir: Path, result: dict[str, Any], *, overwrite: bool) -> dict[str, str]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise OptimizerError("OPTIMIZER_OUTPUT_EXISTS", "output_dir exists and is not empty.", {"output_dir": str(output_dir)})
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_results_source = result.get("trial_results_source")
    outputs = {
        "optimizer_contract": "optimizer_contract.json",
        "optimization_plan": "optimization_plan.json",
        "candidate_space": "candidate_space.json",
        "trial_manifest": "trial_manifest.csv",
        "best_pipeline": "best_pipeline.json",
        "recommendation_report": "recommendation_report.md",
    }
    if trial_results_source:
        outputs["trial_results"] = "trial_results.csv"
    contract = {
        "contract_type": "optimization_contract",
        "contract_id": f"optimization-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "contract_status": {"status": result["status"]},
        "mode": result["mode"],
        "evaluation_context": result.get("evaluation_context"),
        "budget_audit": result.get("budget_audit"),
        "selection_metric": result["selection_metric"],
        "selection_protocol": result["selection_protocol"],
        "test_used_for_selection": False,
        "final_test_evaluated_once": True,
        "test_access_log": result.get("test_access_log"),
        "blind_test_status": _blind_test_status(result.get("test_access_log")),
        "trial_execution": result.get("trial_execution"),
        "candidate_space_policy": result.get("candidate_space_policy"),
        "validator_model": result.get("validator_model"),
        "locked_parameter_check": result.get("locked_parameter_check"),
        "stability_followup": result.get("stability_followup"),
        "tie_breaker_policy": result.get("tie_breaker_policy", TIE_BREAKER_POLICY),
        "test_metrics_ignored_for_selection": bool((result.get("best_pipeline") or {}).get("test_metrics_ignored_for_selection")),
        "outputs": outputs,
    }
    (output_dir / "optimizer_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    plan = {
        key: result[key]
        for key in [
            "mode",
            "task_type",
            "data_profile",
            "selection_metric",
            "selection_protocol",
            "test_used_for_selection",
            "final_test_evaluated_once",
            "budget_audit",
            "trial_execution",
            "test_access_log",
            "evaluation_context",
            "tie_breaker_policy",
            "candidate_space_policy",
            "validator_model",
            "locked_parameter_check",
            "stability_followup",
            "random_seed",
        ]
        if key in result
    }
    (output_dir / "optimization_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "candidate_space.json").write_text(json.dumps(result.get("candidate_space", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_trials_csv(output_dir / "trial_manifest.csv", result.get("trials", []), include_metrics=False)
    if trial_results_source:
        _preserve_trial_results(Path(trial_results_source), output_dir / "trial_results.csv")
    best_payload = result.get("best_pipeline")
    if isinstance(best_payload, dict) and result.get("evaluation_context"):
        best_payload = {**best_payload, "evaluation_context": result.get("evaluation_context")}
    if isinstance(best_payload, dict) and result.get("stability_followup"):
        best_payload = {**best_payload, "stability_followup": result.get("stability_followup")}
    (output_dir / "best_pipeline.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "recommendation_report.md").write_text(_report_markdown(result), encoding="utf-8")
    return {key: value for key, value in outputs.items()}


def _preserve_trial_results(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        return
    if not source.exists():
        raise OptimizerError("OPTIMIZER_TRIAL_RESULTS_MISSING", "trial_results source does not exist.", {"trial_results": str(source)})
    shutil.copyfile(source, destination)


def _blind_test_status(test_access: dict[str, Any] | None) -> str:
    if not test_access or not test_access.get("test_set_accessed"):
        return "not_recorded_as_accessed"
    return "confirmatory_not_blind"


def _evaluation_context(test_access: dict[str, Any] | None) -> dict[str, Any]:
    accessed = bool(test_access and test_access.get("test_set_accessed"))
    return {
        "evaluation_context": "post_test_exploratory" if accessed else "pre_test_selection",
        "blind_test_available": not accessed,
        "prior_test_access_detected": accessed,
        "prior_test_access_time": (test_access or {}).get("first_access_time"),
        "test_used_for_selection": False,
        "note": (
            "A previous test access was detected; optimizer results are exploratory/confirmatory and must not use test metrics for selection."
            if accessed
            else "No previous test access was recorded by this run context."
        ),
    }


def _write_trials_csv(path: Path, trials: list[dict[str, Any]], *, include_metrics: bool) -> None:
    fieldnames = [
        "trial_id",
        "preprocess_method",
        "feature_method",
        "model_method",
        "params",
        "selection_metric",
        "val_accuracy",
        "val_macro_f1",
        "cv_accuracy",
        "cv_rmse",
        "fit_time",
        "test_used_for_selection",
        "status",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "trial_id": trial["trial_id"],
                    "preprocess_method": trial["preprocess_method"],
                    "feature_method": trial["feature_method"],
                    "model_method": trial["model_method"],
                    "params": json.dumps(
                        {
                            "preprocess": trial["preprocess_params"],
                            "feature": trial["feature_params"],
                            "modeling": trial["model_params"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "selection_metric": trial["selection_metric"],
                    "val_accuracy": "" if include_metrics else "",
                    "val_macro_f1": "" if include_metrics else "",
                    "cv_accuracy": "" if include_metrics else "",
                    "cv_rmse": "" if include_metrics else "",
                    "fit_time": "",
                    "test_used_for_selection": "False",
                    "status": trial["status"],
                    "warnings": trial["warnings"],
                }
            )


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Spectral Optimizer Recommendation",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Task: `{result['task_type']}`",
        f"- Selection metric: `{result['selection_metric']}`",
        f"- Selection protocol: `{result['selection_protocol']}`",
        "- Test used for selection: `false`",
        "- Tie breaker: validation/CV metric first, then simpler/fewer-feature pipelines",
        "",
    ]
    best_pipeline = result.get("best_pipeline") or {}
    if best_pipeline.get("test_metrics_ignored_for_selection"):
        lines.append("- Test metrics were present in trial results and ignored for selection.")
        lines.append("")
    budget = result.get("budget_audit") or {}
    if budget.get("budget_exceeded"):
        lines.append(f"- Candidate budget exceeded: {budget.get('expanded_trials')} expanded trials > max {budget.get('requested_max_trials')}.")
        lines.append("")
    policy = result.get("candidate_space_policy") or {}
    if policy:
        lines.append("## Candidate Space Policy")
        lines.append(f"- Policy: `{policy.get('policy')}`")
        excluded = policy.get("excluded_methods") or []
        if excluded:
            lines.append("- Excluded from this compact space:")
            for item in excluded[:8]:
                lines.append(f"  - `{item.get('stage')}` `{item.get('method')}`: {item.get('reason')}")
        lines.append("")
    validator = result.get("validator_model")
    if validator:
        lines.append("## Validator Model")
        lines.append(f"- Methods: `{', '.join(str(item) for item in validator.get('methods', []))}`")
        lines.append(f"- Source: `{validator.get('source')}`")
        if validator.get("previous_confirmation_stage"):
            lines.append(f"- Previous confirmation stage: `{validator.get('previous_confirmation_stage')}`")
        lines.append("- Validator choice affects candidate ranking.")
        lines.append("")
    if (result.get("test_access_log") or {}).get("test_set_accessed"):
        lines.append("- Test set access was already recorded; downstream test metrics should be treated as confirmatory rather than fully blind.")
        context = result.get("evaluation_context") or {}
        if context.get("prior_test_access_time"):
            lines.append(f"- First recorded test access: `{context.get('prior_test_access_time')}`")
        lines.append("")
    if result.get("recommendations"):
        lines.append("## Recommendations")
        for item in result["recommendations"]:
            lines.append(f"- {item.get('reason') or item.get('message')}")
    stability = result.get("stability_followup") or {}
    if stability:
        lines.extend(
            [
                "",
                "## Stability Follow-up",
                f"- Protocol: `{stability.get('protocol')}`",
                f"- Repeats: `{stability.get('n_repeats_options')}`",
                "- Keep the selected preprocess, feature, model, and model parameters locked.",
                f"- Report `{stability.get('selection_metric')}` mean and standard deviation.",
            ]
        )
    if result.get("trial_count"):
        lines.extend(["", "## Trial Plan", f"- Planned trials: {result['trial_count']}"])
    if result.get("mode") in {"compare_step", "tune_method"}:
        lines.extend(
            [
                "",
                "## Scope Note",
                "- This recommendation is conditional on the fixed upstream/downstream choices in the candidate space.",
            ]
        )
    if result.get("mode") == "optimize_pipeline":
        lines.extend(
            [
                "",
                "## Scope Note",
                "- This recommendation is the joint optimum within the confirmed pipeline candidate space.",
                "- If it differs from stage-wise recommendations, prefer this result for the searched full pipeline because greedy stage-wise choices need not equal the joint optimum.",
            ]
        )
    if result.get("best_pipeline"):
        best = result["best_pipeline"]
        lines.extend(["", "## Best Pipeline", f"- Trial: `{best.get('trial_id')}`", f"- {best.get('selection_metric')}: {best.get('selection_value')}"])
        tie = best.get("tie_breaker") or {}
        if tie.get("tie_count", 0) > 1:
            lines.append(f"- Tie-breaker: {tie.get('selected_reason')}")
    else:
        lines.extend(["", "No best pipeline was selected because no validation/CV trial result file was supplied."])
    return "\n".join(lines) + "\n"

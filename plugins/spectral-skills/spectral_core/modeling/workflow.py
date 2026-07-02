"""Unified leakage-safe spectral modeling workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from spectral_core.reader.response import error_response, ok_response

from .io import ModelingInputError, load_modeling_inputs_from_contract, load_modeling_package, load_modeling_split
from .methods import ModelingMethodError, infer_task_type, parse_models, train_and_evaluate, train_and_evaluate_iteration_packages
from .registry import comparison_fixed_parameters, load_model_config, missing_dependencies, prepare_model_parameters
from .writer import ModelingWriteError, write_classifier_comparison_outputs, write_modeling_iteration_outputs, write_modeling_outputs


def model_spectral_package(
    *,
    package_dir: str | Path | None = None,
    split_contract: str | Path | None = None,
    feature_contract: str | Path | None = None,
    preprocess_contract: str | Path | None = None,
    output_dir: str | Path | None = None,
    task_type: str | None = None,
    models: str | list[str] | None = None,
    model_config: str | Path | dict[str, Any] | None = None,
    best_pipeline: str | Path | dict[str, Any] | None = None,
    lock_best_pipeline_params: bool = False,
    disable_model_selection: bool = False,
    model_parameters: dict[str, Any] | None = None,
    auto_confirm_model_defaults: bool = False,
    cv_folds: int = 3,
    random_seed: int = 42,
    confirm_no_test: bool = False,
    require_test_confirmation: bool = False,
    confirm_test_evaluation: bool = False,
    confirm_confirmatory_test_evaluation: bool = False,
    evaluation_mode: str = "final",
    save_model: bool = True,
    overwrite: bool = False,
    backend: str = "core",
    mode: str = "standard",
    checkpoint_per_model: bool = False,
    candidate_model_set_source: str | None = None,
    confirm_gated_feature_modeling: bool = False,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    try:
        selected_evaluation_mode = str(evaluation_mode or "final").strip().lower().replace("-", "_")
        if selected_evaluation_mode not in {"final", "validation_only"}:
            return error_response(
                "model_spectral_package",
                "evaluation_mode must be final or validation_only.",
                backend=backend,
                code="MODELING_EVALUATION_MODE_UNSUPPORTED",
                result={"status": "blocked"},
                warnings=warnings,
            )
        selected_mode = str(mode or "standard").strip().lower().replace("-", "_")
        if selected_mode not in {"standard", "repeated_classifier_comparison"}:
            return error_response(
                "model_spectral_package",
                "mode must be standard or repeated_classifier_comparison.",
                backend=backend,
                code="MODELING_MODE_UNSUPPORTED",
                result={"status": "blocked"},
                warnings=warnings,
            )
        best_config = _load_best_pipeline_model_config(best_pipeline) if best_pipeline is not None else None
        if lock_best_pipeline_params:
            if best_config is None:
                return error_response(
                    "model_spectral_package",
                    "lock_best_pipeline_params requires --best-pipeline so final evaluation can exactly reproduce optimizer-selected model params.",
                    backend=backend,
                    code="BEST_PIPELINE_REQUIRED_FOR_LOCK",
                    result={"status": "blocked"},
                    warnings=warnings,
                )
            if feature_contract is None and preprocess_contract is None:
                upstream_contract = best_config.get("input_contract")
                if upstream_contract:
                    if best_config.get("input_stage") == "feature":
                        feature_contract = upstream_contract
                    elif best_config.get("input_stage") == "preprocess":
                        preprocess_contract = upstream_contract
                elif best_config.get("requires_upstream_contract"):
                    return error_response(
                        "model_spectral_package",
                        "--best-pipeline records a non-none preprocess/feature stage, but no matching upstream contract could be resolved. Refusing to run a model-only fallback because that would not reproduce the optimizer-selected full pipeline.",
                        backend=backend,
                        code="BEST_PIPELINE_UPSTREAM_CONTRACT_MISSING",
                        result={
                            "status": "blocked",
                            "best_pipeline": best_config.get("best_pipeline_path"),
                            "trial_id": best_config.get("trial_id"),
                            "preprocess_method": best_config.get("preprocess_method"),
                            "feature_method": best_config.get("feature_method"),
                            "expected_trial_dir": best_config.get("trial_dir"),
                        },
                        warnings=warnings,
                    )
        iteration_contract = feature_contract or preprocess_contract
        iteration_packages = None
        if iteration_contract is not None:
            if feature_contract is not None:
                gate = _feature_contract_modeling_gate(feature_contract, confirm_gated_feature_modeling)
                if gate["status"] != "ready":
                    return error_response(
                        "model_spectral_package",
                        gate["message"],
                        backend=backend,
                        code=gate["code"],
                        result={
                            "status": gate["status"],
                            "feature_method": gate.get("feature_method"),
                            "confirmation_required": gate.get("confirmation_required", []),
                        },
                        warnings=warnings,
                    )
                if gate.get("warning"):
                    warnings.append(gate["warning"])
            iteration_packages, split_info, package = load_modeling_inputs_from_contract(iteration_contract)
        else:
            if package_dir is None:
                return error_response(
                    "model_spectral_package",
                    "Provide package_dir for standard packages, feature_contract for fold/repeat-wise features, or preprocess_contract for fold/repeat-wise preprocessing.",
                    backend=backend,
                    code="MODELING_INPUT_REQUIRED",
                    result={"status": "needs_confirmation"},
                    warnings=warnings,
                )
            package = load_modeling_package(package_dir)
            split_info = load_modeling_split(split_contract, package)
        selected_task = infer_task_type(package, task_type)
        if lock_best_pipeline_params:
            models = best_config["model"]
            model_config = {best_config["model"]: best_config["params"]}
            disable_model_selection = True
        selected_models = parse_models(models, selected_task)
        configured = load_model_config(model_config)
        comparison_fixed_defaults: dict[str, dict[str, Any]] = {}
        if selected_mode == "repeated_classifier_comparison":
            if selected_task != "classification":
                return error_response(
                    "model_spectral_package",
                    "repeated_classifier_comparison is only supported for classification tasks.",
                    backend=backend,
                    code="CLASSIFIER_COMPARISON_REQUIRES_CLASSIFICATION",
                    result={"status": "blocked"},
                    warnings=warnings,
                )
            if split_info.split_type not in {"cross_validation", "repeated_holdout"}:
                return error_response(
                    "model_spectral_package",
                    "repeated_classifier_comparison requires a repeated-holdout or cross-validation split contract.",
                    backend=backend,
                    code="CLASSIFIER_COMPARISON_REQUIRES_REPEATED_OR_CV",
                    result={"status": "needs_confirmation"},
                    warnings=warnings,
                )
            disable_model_selection = True
            comparison_fixed_defaults = {model: comparison_fixed_parameters(model) for model in selected_models}
            configured = {model: {**comparison_fixed_defaults.get(model, {}), **configured.get(model, {})} for model in selected_models}
            if "gradient_boosting_classifier" in selected_models:
                warnings.append(
                    {
                        "code": "SLOW_MODEL_WARNING",
                        "model": "gradient_boosting_classifier",
                        "message": "Gradient Boosting can dominate runtime on high-dimensional spectra. Offer regular-fast without Gradient Boosting unless the user confirms regular-full.",
                        "alternatives": ["regular-fast", "regular-full"],
                    }
                )
        resolved_parameters, parameter_sources, confirmations = prepare_model_parameters(
            selected_models,
            configured,
            common_params=model_parameters,
            auto_confirm_defaults=auto_confirm_model_defaults,
            random_seed=random_seed,
            )
        if confirmations:
            return error_response(
                "model_spectral_package",
                "Please confirm the training-sensitive parameters for the selected experimental model.",
                backend=backend,
                code="MODEL_PARAMETERS_CONFIRMATION_REQUIRED",
                result={
                    "status": "needs_confirmation",
                    "confirmation_required": confirmations,
                    "models": selected_models,
                },
                warnings=warnings,
            )
        if selected_mode == "repeated_classifier_comparison":
            for model, defaults in comparison_fixed_defaults.items():
                sources = parameter_sources.setdefault(model, {})
                user_overrides = load_model_config(model_config).get(model, {}) if model_config is not None else {}
                for key in defaults:
                    if key not in user_overrides:
                        sources[key] = "comparison_fixed_default"
        dependencies = missing_dependencies(selected_models)
        if dependencies:
            return error_response(
                "model_spectral_package",
                "One or more selected models require optional dependencies that are not installed.",
                backend=backend,
                code="MODEL_DEPENDENCY_MISSING",
                result={"status": "blocked", "missing_dependencies": dependencies},
                warnings=warnings,
            )
        if selected_evaluation_mode == "validation_only" and split_info.split_type != "holdout":
            return error_response(
                "model_spectral_package",
                "validation_only is for holdout optimizer trials; cross-validation and repeated-holdout already use partition-wise evaluation.",
                backend=backend,
                code="VALIDATION_ONLY_REQUIRES_HOLDOUT",
                result={"status": "blocked"},
                warnings=warnings,
            )
        if selected_evaluation_mode == "validation_only" and not split_info.assignments.get("val"):
            return error_response(
                "model_spectral_package",
                "validation_only requires a non-empty validation split.",
                backend=backend,
                code="VALIDATION_SPLIT_REQUIRED",
                result={"status": "needs_confirmation"},
                warnings=warnings,
            )
        if selected_evaluation_mode == "final" and split_info.split_type == "holdout" and not split_info.assignments.get("test") and not confirm_no_test:
            return error_response(
                "model_spectral_package",
                "No test split is available; confirm if you only want validation performance without independent final test evaluation.",
                backend=backend,
                code="TEST_SPLIT_REQUIRED",
                result={"status": "needs_confirmation"},
                warnings=warnings,
            )
        if (
            selected_evaluation_mode == "final"
            and split_info.split_type == "holdout"
            and split_info.assignments.get("test")
            and require_test_confirmation
            and not confirm_test_evaluation
        ):
            return error_response(
                "model_spectral_package",
                "Final holdout test evaluation requires explicit user confirmation because test metrics should be accessed only after the pipeline is fixed.",
                backend=backend,
                code="TEST_EVALUATION_CONFIRMATION_REQUIRED",
                result={
                    "status": "needs_confirmation",
                    "confirmation_required": [
                        {
                            "field": "test_evaluation",
                            "reason": "The test split is for final/confirmatory evaluation, not method selection.",
                            "options": ["rerun with --confirm-test-evaluation after the user confirms final testing"],
                        }
                    ],
                },
                warnings=warnings,
            )
        prior_test_access = _prior_test_access(output_dir)
        if (
            selected_evaluation_mode == "final"
            and split_info.split_type == "holdout"
            and split_info.assignments.get("test")
            and prior_test_access
            and not confirm_confirmatory_test_evaluation
        ):
            return error_response(
                "model_spectral_package",
                "This run has already accessed the test split; another final test evaluation must be explicitly confirmed as confirmatory.",
                backend=backend,
                code="CONFIRMATORY_TEST_EVALUATION_REQUIRED",
                result={
                    "status": "needs_confirmation",
                    "test_access_log": prior_test_access,
                    "confirmation_required": [
                        {
                            "field": "confirmatory_test_evaluation",
                            "reason": "The test set is no longer a fully blind holdout after prior access.",
                            "options": ["rerun with --confirm-confirmatory-test-evaluation after the user confirms confirmatory testing"],
                        }
                    ],
                },
                warnings=warnings,
            )
        if selected_mode == "repeated_classifier_comparison":
            result = _run_repeated_classifier_comparison(
                package=package,
                split_info=split_info,
                iteration_packages=iteration_packages,
                task_type=selected_task,
                models=selected_models,
                model_parameters=resolved_parameters,
                parameter_sources=parameter_sources,
                cv_folds=cv_folds,
                random_seed=random_seed,
                model_params_source="comparison_fixed_default" if comparison_fixed_defaults else ("model_config" if configured else "modeling_internal_selection"),
                output_dir=output_dir,
                checkpoint_per_model=checkpoint_per_model,
                warnings=warnings,
                backend=backend,
            )
            result["configured_model_parameters"] = resolved_parameters
            result["parameter_sources"] = parameter_sources
            result["candidate_model_set_source"] = candidate_model_set_source or ("models_alias_or_list" if isinstance(models, str) else "models_list")
            preview = {
                "status": "ready",
                "task_type": result["task_type"],
                "split_type": result["split_type"],
                "execution_mode": result["execution_mode"],
                "comparison_mode": result["comparison_mode"],
                "model_count": len(result["models"]),
                "iteration_count": len(result["iteration_results"]),
                "models": result["models"],
                "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
                "warnings": warnings,
            }
            if output_dir is None:
                preview["handoff_ready"] = False
                preview["message"] = "No output_dir was provided; returning repeated classifier comparison preview without writing result files."
                return ok_response("model_spectral_package", preview, backend=backend, warnings=warnings)
            written = write_classifier_comparison_outputs(
                package,
                split_info,
                result=result,
                output_dir=output_dir,
                warnings=warnings,
                overwrite=overwrite,
                backend=backend,
            )
            return ok_response("model_spectral_package", written, backend=backend, warnings=warnings)
        if iteration_packages is not None:
            result = train_and_evaluate_iteration_packages(
                iteration_packages,
                split_info,
                task_type=selected_task,
                models=selected_models,
                model_parameters=resolved_parameters,
                parameter_sources=parameter_sources,
                cv_folds=cv_folds,
                random_seed=random_seed,
                param_search_enabled=not disable_model_selection,
                model_params_source="best_pipeline.json" if lock_best_pipeline_params else ("model_config" if configured else "modeling_internal_selection"),
            )
        else:
            result = train_and_evaluate(
                package,
                split_info,
                task_type=selected_task,
                models=selected_models,
                model_parameters=resolved_parameters,
                parameter_sources=parameter_sources,
                cv_folds=cv_folds,
                random_seed=random_seed,
                evaluation_mode=selected_evaluation_mode,
                param_search_enabled=not disable_model_selection,
                model_params_source="best_pipeline.json" if lock_best_pipeline_params else ("model_config" if configured else "modeling_internal_selection"),
            )
        result["evaluation_mode"] = selected_evaluation_mode
        result["test_accessed"] = bool(result.get("test_accessed", selected_evaluation_mode == "final" and split_info.split_type == "holdout"))
        result["configured_model_parameters"] = resolved_parameters
        result["parameter_sources"] = parameter_sources
        result["model_selection_mode"] = "locked_from_optimizer_best" if lock_best_pipeline_params else ("fixed_model_config" if disable_model_selection else "modeling_internal_selection")
        result["param_search_enabled"] = not disable_model_selection
        result["model_params_source"] = "best_pipeline.json" if lock_best_pipeline_params else ("model_config" if configured else "modeling_internal_selection")
        if lock_best_pipeline_params and best_config is not None:
            result["best_pipeline_reproduction"] = _best_pipeline_reproduction_payload(best_config, feature_contract=feature_contract, preprocess_contract=preprocess_contract)
        if not result.get("model_parameters") and result.get("model_type") in resolved_parameters:
            result["model_parameters"] = resolved_parameters[result["model_type"]]
        if result.get("execution_mode") in {"fold_wise", "repeat_wise"}:
            preview = {
                "status": "ready",
                "task_type": result["task_type"],
                "split_type": result["split_type"],
                "execution_mode": result["execution_mode"],
                "iteration_count": len(result["iteration_results"]),
                "metric_summary": result["metric_summary"],
                "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
                "warnings": warnings,
            }
            if output_dir is None:
                preview["handoff_ready"] = False
                preview["message"] = "No output_dir was provided; returning CV/repeated modeling preview without writing result files."
                return ok_response("model_spectral_package", preview, backend=backend, warnings=warnings)
            written = write_modeling_iteration_outputs(
                package,
                split_info,
                result=result,
                output_dir=output_dir,
                warnings=warnings,
                save_model=save_model,
                overwrite=overwrite,
                backend=backend,
            )
            return ok_response("model_spectral_package", written, backend=backend, warnings=warnings)
        preview = {
            "status": "ready",
            "task_type": result["task_type"],
            "model_type": result["model_type"],
            "model_parameters": result["model_parameters"],
            "selection_strategy": result["selection"],
            "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
            "warnings": warnings,
        }
        if output_dir is None:
            preview["handoff_ready"] = False
            preview["message"] = "No output_dir was provided; returning modeling preview without writing modeling_contract.json."
            return ok_response("model_spectral_package", preview, backend=backend, warnings=warnings)
        written = write_modeling_outputs(
            package,
            split_info,
            result=result,
            output_dir=output_dir,
            warnings=warnings,
            save_model=save_model and selected_evaluation_mode == "final",
            overwrite=overwrite,
            backend=backend,
        )
        return ok_response("model_spectral_package", written, backend=backend, warnings=warnings)
    except ModelingInputError as exc:
        status = "needs_confirmation" if exc.code == "SPLIT_CONTRACT_REQUIRED" else "blocked"
        return error_response("model_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": status}, details=exc.details, warnings=warnings)
    except ModelingMethodError as exc:
        status = "needs_confirmation" if exc.code in {"TASK_TYPE_REQUIRED", "MODEL_TYPE_REQUIRED", "TEST_SPLIT_REQUIRED", "MODEL_PARAMETERS_CONFIRMATION_REQUIRED"} else "blocked"
        return error_response("model_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": status}, details=exc.details, warnings=warnings)
    except ModelingWriteError as exc:
        return error_response("model_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details, warnings=warnings)


def _feature_contract_modeling_gate(contract_path: str | Path, confirmed: bool) -> dict[str, Any]:
    path = Path(contract_path)
    if not path.exists():
        return {"status": "ready"}
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_type") != "feature_contract":
        return {"status": "ready"}
    method = contract.get("feature_method")
    intended_use = contract.get("intended_use")
    out_of_sample = contract.get("out_of_sample_transform")
    handoff = ((contract.get("handoff") or {}).get("spectral_modeling") or {})
    if intended_use == "visualization" or out_of_sample == "unsupported" or handoff.get("blocked") is True:
        return {
            "status": "blocked",
            "code": "FEATURE_NOT_MODELING_COMPATIBLE",
            "message": "The feature contract is visualization-only or lacks a supported out-of-sample transform; it cannot be used for modeling.",
            "feature_method": method,
        }
    requires_confirmation = bool(contract.get("modeling_requires_confirmation") or handoff.get("requires_confirmation"))
    if requires_confirmation and not confirmed:
        return {
            "status": "needs_confirmation",
            "code": "GATED_FEATURE_MODELING_CONFIRMATION_REQUIRED",
            "message": "This manifold or visualization-first embedding requires explicit confirmation before modeling.",
            "feature_method": method,
            "confirmation_required": [
                {
                    "field": "confirm_gated_feature_modeling",
                    "reason": "Confirm limited/manifold out-of-sample semantics and exclude test metrics from feature choice.",
                }
            ],
        }
    convergence = contract.get("convergence")
    warning = None
    if isinstance(convergence, dict) and convergence.get("converged") is False:
        warning = {
            "code": "FEATURE_CONVERGENCE_WARNING",
            "message": "The upstream feature method did not clearly converge; modeling continues only with this warning recorded.",
            "feature_method": method,
            "convergence": convergence,
        }
    return {"status": "ready", "warning": warning}


def _run_repeated_classifier_comparison(
    *,
    package: Any,
    split_info: Any,
    iteration_packages: dict[str, Any] | None,
    task_type: str,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]],
    parameter_sources: dict[str, dict[str, str]],
    cv_folds: int,
    random_seed: int,
    model_params_source: str,
    output_dir: str | Path | None,
    checkpoint_per_model: bool,
    warnings: list[dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    all_iterations: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    checkpoint_root = Path(output_dir) / "checkpoints" if output_dir is not None and checkpoint_per_model else None
    for model in models:
        if iteration_packages is not None:
            result = train_and_evaluate_iteration_packages(
                iteration_packages,
                split_info,
                task_type=task_type,
                models=[model],
                model_parameters={model: model_parameters.get(model, {})},
                parameter_sources={model: parameter_sources.get(model, {})},
                cv_folds=cv_folds,
                random_seed=random_seed,
                param_search_enabled=False,
                model_params_source=model_params_source,
            )
        else:
            result = train_and_evaluate(
                package,
                split_info,
                task_type=task_type,
                models=[model],
                model_parameters={model: model_parameters.get(model, {})},
                parameter_sources={model: parameter_sources.get(model, {})},
                cv_folds=cv_folds,
                random_seed=random_seed,
                param_search_enabled=False,
                model_params_source=model_params_source,
            )
        result["model_selection_mode"] = "per_classifier_repeated_evaluation"
        result["param_search_enabled"] = False
        result["model_params_source"] = model_params_source
        for item in result["iteration_results"]:
            item["model_selection_mode"] = "per_classifier_repeated_evaluation"
            item["param_search_enabled"] = False
            item["model_params_source"] = model_params_source
            item["model_type"] = model
            for row in item.get("predictions", []):
                row["model_method"] = model
                row["model_type"] = model
        for row in result.get("predictions", []):
            row["model_method"] = model
            row["model_type"] = model
        if checkpoint_root is not None:
            write_modeling_iteration_outputs(
                package,
                split_info,
                result=result,
                output_dir=checkpoint_root / model,
                warnings=warnings,
                save_model=False,
                overwrite=True,
                backend=backend,
            )
        all_iterations.extend(result["iteration_results"])
        all_predictions.extend(result["predictions"])
    eval_role = "val" if split_info.split_type == "cross_validation" else "test"
    return {
        "task_type": task_type,
        "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
        "comparison_mode": "per_classifier_repeated_evaluation",
        "split_type": split_info.split_type,
        "split_method": split_info.method,
        "eval_role": eval_role,
        "models": models,
        "model_type": ",".join(models),
        "configured_model_parameters": model_parameters,
        "parameter_sources": parameter_sources,
        "model_params_source": model_params_source,
        "iteration_results": all_iterations,
        "predictions": all_predictions,
        "test_accessed": split_info.split_type == "repeated_holdout",
    }


def _load_best_pipeline_model_config(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    source_path: Path | None = None
    if isinstance(value, dict):
        payload = dict(value)
    else:
        source_path = Path(value).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    row = payload.get("row") if isinstance(payload.get("row"), dict) else {}
    model = payload.get("model_method") or row.get("model_method")
    trial_id = payload.get("trial_id") or row.get("trial_id")
    preprocess_method = payload.get("preprocess_method") or row.get("preprocess_method") or "none"
    feature_method = payload.get("feature_method") or row.get("feature_method") or "none"
    params = payload.get("model_params") if isinstance(payload.get("model_params"), dict) else None
    if params is None:
        params_payload = row.get("params")
        if isinstance(params_payload, str) and params_payload.strip():
            try:
                parsed = json.loads(params_payload)
            except json.JSONDecodeError:
                parsed = {}
            params = parsed.get("modeling") if isinstance(parsed.get("modeling"), dict) else {}
        else:
            params = {}
    if not model:
        raise ModelingMethodError("BEST_PIPELINE_MODEL_MISSING", "best_pipeline.json does not record a model_method.")
    input_contract = _best_pipeline_input_contract(payload, row, source_path)
    input_stage = None
    if input_contract is not None:
        name = input_contract.name
        input_stage = "feature" if name == "feature_contract.json" else "preprocess" if name == "preprocess_contract.json" else None
    requires_upstream = str(feature_method).strip().lower() not in {"", "none"} or str(preprocess_method).strip().lower() not in {"", "none"}
    return {
        "model": str(model),
        "params": dict(params or {}),
        "trial_id": str(trial_id) if trial_id else None,
        "preprocess_method": str(preprocess_method),
        "feature_method": str(feature_method),
        "input_contract": str(input_contract) if input_contract is not None else None,
        "input_stage": input_stage,
        "requires_upstream_contract": requires_upstream,
        "trial_dir": str(_best_pipeline_trial_dir(row, source_path)) if _best_pipeline_trial_dir(row, source_path) is not None else None,
        "best_pipeline_path": str(source_path) if source_path is not None else None,
    }


def _best_pipeline_input_contract(payload: dict[str, Any], row: dict[str, Any], source_path: Path | None) -> Path | None:
    explicit_refs = [
        payload.get("feature_contract"),
        row.get("feature_contract"),
        (payload.get("inputs") or {}).get("feature_contract") if isinstance(payload.get("inputs"), dict) else None,
    ]
    for ref in explicit_refs:
        path = _resolve_best_ref(source_path, ref)
        if path is not None and path.exists():
            return path
    trial_dir = _best_pipeline_trial_dir(row, source_path)
    if trial_dir is not None:
        for candidate in [
            trial_dir / "feature_output" / "feature_contract.json",
            trial_dir / "preprocess_output" / "preprocess_contract.json",
        ]:
            if candidate.exists():
                return candidate.resolve()
    explicit_preprocess_refs = [
        payload.get("preprocess_contract"),
        row.get("preprocess_contract"),
        (payload.get("inputs") or {}).get("preprocess_contract") if isinstance(payload.get("inputs"), dict) else None,
    ]
    for ref in explicit_preprocess_refs:
        path = _resolve_best_ref(source_path, ref)
        if path is not None and path.exists():
            return path
    return None


def _best_pipeline_trial_dir(row: dict[str, Any], source_path: Path | None) -> Path | None:
    modeling_output = row.get("modeling_output")
    modeling_path = _resolve_best_ref(source_path, modeling_output)
    if modeling_path is not None:
        return (modeling_path.parent if modeling_path.name == "model_output" else modeling_path).resolve()
    return None


def _resolve_best_ref(source_path: Path | None, ref: Any) -> Path | None:
    if ref in {None, ""}:
        return None
    path = Path(str(ref))
    if path.is_absolute():
        return path.resolve()
    base = source_path.parent if source_path is not None else Path.cwd()
    return (base / path).resolve()


def _best_pipeline_reproduction_payload(best_config: dict[str, Any], *, feature_contract: str | Path | None, preprocess_contract: str | Path | None) -> dict[str, Any]:
    return {
        "source": "best_pipeline.json",
        "best_pipeline_path": best_config.get("best_pipeline_path"),
        "trial_id": best_config.get("trial_id"),
        "preprocess_method": best_config.get("preprocess_method"),
        "feature_method": best_config.get("feature_method"),
        "model_method": best_config.get("model"),
        "model_params_locked": True,
        "input_stage": best_config.get("input_stage"),
        "resolved_input_contract": str(feature_contract or preprocess_contract) if (feature_contract or preprocess_contract) is not None else None,
        "full_pipeline_reproduced": bool(best_config.get("input_contract")) or not best_config.get("requires_upstream_contract"),
    }


def _prior_test_access(output_dir: str | Path | None) -> dict[str, Any] | None:
    if output_dir is None:
        return None
    out = Path(output_dir)
    run_dir = out.parent if out.name in {"model_output", "final_model"} else out
    path = run_dir / "test_access_log.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("test_set_accessed") else None

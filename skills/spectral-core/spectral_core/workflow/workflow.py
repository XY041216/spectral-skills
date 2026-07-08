"""Compact orchestration across spectral skills."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from spectral_core.reader.io_utils import load_json_file, write_json_file
from spectral_core.reader.response import error_response, ok_response
from spectral_core.feature.parameter_policy import missing_critical_parameters, normalize_feature_method
from spectral_core.workflow.run_layout import create_run_layout, ensure_run_dirs, relative_stage_outputs, update_runs_index, write_run_manifest
from spectral_core.workflow.state import create_workflow_plan, normalize_workflow_goal, update_workflow_result


SPLIT_PREPARATION_GOALS = {"split", "prepare_for_optimizer", "compare_preprocess"}
SUPPORTED_GOALS = {"read", "qc", *SPLIT_PREPARATION_GOALS, "preprocess", "feature", "classification", "regression", "modeling"}
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
StageRunner = Callable[[str, list[str]], dict[str, Any]]


def run_spectral_workflow(
    *,
    input_path: str | Path | None = None,
    package_dir: str | Path | None = None,
    data_contract: str | Path | None = None,
    split_contract: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    task_goal: str | None = None,
    task_type: str | None = None,
    include_qc: bool = False,
    skip_qc: bool = False,
    qc_mode: str = "check",
    split_ratio: str | None = None,
    confirm_incomplete_split_ratio: bool = False,
    split_method: str | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    shuffle: bool = True,
    preprocess_methods: str | list[str] | None = None,
    preprocess_window_length: int | None = None,
    preprocess_polyorder: int | None = None,
    preprocess_sigma: float | None = None,
    preprocess_poly_degree: int | None = None,
    preprocess_als_lambda: float | None = None,
    preprocess_als_p: float | None = None,
    preprocess_als_iter: int | None = None,
    preprocess_band_range: str | None = None,
    preprocess_remove_band_ranges: str | None = None,
    confirm_baseline: bool = False,
    confirm_absorbance: bool = False,
    confirm_band_change: bool = False,
    confirm_unsplit_preprocess_fit: bool = False,
    feature_method: str | None = None,
    feature_n_components: int | None = None,
    feature_explained_variance: float | None = None,
    feature_variance_threshold: float | None = None,
    feature_band_min: float | None = None,
    feature_band_max: float | None = None,
    feature_band_indices: str | None = None,
    feature_names: str | None = None,
    feature_index_base: int = 0,
    feature_top_k: int | None = None,
    feature_score_threshold: float | None = None,
    feature_n_intervals: int | None = None,
    feature_n_runs: int | None = None,
    feature_sample_ratio: float | None = None,
    feature_cv: int | None = None,
    feature_random_state: int | None = None,
    feature_correlation_method: str | None = None,
    feature_interval_mode: str | None = None,
    feature_config: str | Path | None = None,
    auto_confirm_feature_defaults: bool = False,
    confirm_unsplit_feature_fit: bool = False,
    models: str | list[str] | None = None,
    model_config: str | Path | None = None,
    model_n_components: int | None = None,
    model_embedding_dim: int | None = None,
    model_epochs: int | None = None,
    model_batch_size: int | None = None,
    model_alpha: float | None = None,
    model_lr: float | None = None,
    model_kernel: str | None = None,
    model_preprojection: str | None = None,
    model_encoder_type: str | None = None,
    model_metric: str | None = None,
    model_temperature: float | None = None,
    model_device: str | None = None,
    modeling_mode: str | None = None,
    checkpoint_per_model: bool = False,
    candidate_model_set_source: str | None = None,
    auto_confirm_model_defaults: bool = False,
    require_test_confirmation: bool = False,
    confirm_test_evaluation: bool = False,
    confirm_confirmatory_test_evaluation: bool = False,
    reader_sample_orientation: str | None = None,
    reader_label_column: str | None = None,
    reader_target_columns: list[str] | None = None,
    reader_sample_id_column: str | None = None,
    reader_sample_id_column_index: int | None = None,
    reader_spectral_start_column: str | int | None = None,
    reader_spectral_end_column: str | int | None = None,
    reader_band_type: str | None = None,
    reader_band_unit: str | None = None,
    reader_max_auto_columns: int = 10000,
    reader_max_spectral_columns: int = 20000,
    reader_wide_table_mode: str = "auto",
    reader_confirm_wide_table: bool = False,
    reader_confirm_read_plan: bool = False,
    random_seed: int = 42,
    overwrite: bool = False,
    backend: str = "skills",
    stage_runner: StageRunner | None = None,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    try:
        ratio_confirmation = _incomplete_split_ratio_confirmation(split_ratio)
        if ratio_confirmation and not confirm_incomplete_split_ratio:
            return _needs_confirmation(
                "SPLIT_RATIO_CONFIRMATION_REQUIRED",
                "The split ratio is incomplete and must be confirmed before reader/QC/split execution.",
                backend,
                required_fields=["split_ratio"],
                confirmation_required=[ratio_confirmation],
            )
        goal = normalize_workflow_goal(task_goal)
        layout = create_run_layout(
            output_root=output_root,
            output_dir=output_dir,
            run_name=run_name,
            input_path=input_path,
            package_dir=package_dir,
            data_contract=data_contract,
            split_method=split_method,
            split_ratio=split_ratio,
            n_splits=n_splits,
            n_repeats=n_repeats,
            preprocess_methods=preprocess_methods,
            feature_method=feature_method,
            feature_n_components=feature_n_components,
            models=models,
        )
        ensure_run_dirs(layout)
        root = layout.run_dir
        runner = stage_runner or ((lambda stage, args: _run_stage_script(stage, args, layout.stage_dirs["logs"])) if backend in {"skills", "script"} else None)
        effective_include_qc = include_qc or _default_qc_for_goal(goal, skip_qc)
        write_run_manifest(
            layout,
            task_goal=goal,
            status="running",
            parameters={
                "input_path": str(input_path) if input_path is not None else None,
                "package_dir": str(package_dir) if package_dir is not None else None,
                "split_method": split_method,
                "split_ratio": split_ratio,
                "confirm_incomplete_split_ratio": confirm_incomplete_split_ratio,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "preprocess_methods": preprocess_methods,
                "feature_method": feature_method,
                "feature_n_components": feature_n_components,
                "feature_top_k": feature_top_k,
                "feature_n_runs": feature_n_runs,
                "auto_confirm_feature_defaults": auto_confirm_feature_defaults,
                "models": models,
                "model_config": str(model_config) if model_config is not None else None,
                "model_n_components": model_n_components,
                "model_embedding_dim": model_embedding_dim,
                "model_epochs": model_epochs,
                "model_batch_size": model_batch_size,
                "model_alpha": model_alpha,
                "modeling_mode": modeling_mode,
                "checkpoint_per_model": checkpoint_per_model,
                "candidate_model_set_source": candidate_model_set_source,
                "auto_confirm_model_defaults": auto_confirm_model_defaults,
                "require_test_confirmation": require_test_confirmation,
                "confirm_test_evaluation": confirm_test_evaluation,
                "confirm_confirmatory_test_evaluation": confirm_confirmatory_test_evaluation,
                "reader_sample_orientation": reader_sample_orientation,
                "reader_label_column": reader_label_column,
                "reader_sample_id_column": reader_sample_id_column,
                "reader_sample_id_column_index": reader_sample_id_column_index,
                "reader_spectral_start_column": reader_spectral_start_column,
                "reader_spectral_end_column": reader_spectral_end_column,
                "reader_band_type": reader_band_type,
                "reader_band_unit": reader_band_unit,
                "reader_max_auto_columns": reader_max_auto_columns,
                "reader_max_spectral_columns": reader_max_spectral_columns,
                "reader_wide_table_mode": reader_wide_table_mode,
                "random_seed": random_seed,
                "include_qc": effective_include_qc,
                "skip_qc": skip_qc,
            },
        )
        workflow_plan = _write_workflow_plan(
            root,
            task_goal=goal,
            input_path=input_path,
            package_dir=package_dir,
            data_contract=data_contract,
            split_contract=split_contract,
            include_qc=effective_include_qc,
            qc_mode=qc_mode,
            split_ratio=split_ratio,
            split_method=split_method,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_splits=n_splits,
            n_repeats=n_repeats,
            shuffle=shuffle,
            preprocess_methods=preprocess_methods,
            preprocess_parameters={
                "window_length": preprocess_window_length,
                "polyorder": preprocess_polyorder,
                "sigma": preprocess_sigma,
                "poly_degree": preprocess_poly_degree,
                "als_lambda": preprocess_als_lambda,
                "als_p": preprocess_als_p,
                "als_iter": preprocess_als_iter,
                "band_range": preprocess_band_range,
                "remove_band_ranges": preprocess_remove_band_ranges,
                "confirm_baseline": confirm_baseline,
                "confirm_absorbance": confirm_absorbance,
                "confirm_band_change": confirm_band_change,
                "confirm_unsplit_fit": confirm_unsplit_preprocess_fit,
            },
            feature_method=feature_method,
            feature_parameters={
                "n_components": feature_n_components,
                "explained_variance": feature_explained_variance,
                "variance_threshold": feature_variance_threshold,
                "band_min": feature_band_min,
                "band_max": feature_band_max,
                "band_indices": feature_band_indices,
                "feature_names": feature_names,
                "index_base": feature_index_base,
                "confirm_unsplit_fit": confirm_unsplit_feature_fit,
                "top_k": feature_top_k,
                "score_threshold": feature_score_threshold,
                "n_intervals": feature_n_intervals,
                "n_runs": feature_n_runs,
                "sample_ratio": feature_sample_ratio,
                "cv": feature_cv,
                "random_state": feature_random_state,
                "correlation_method": feature_correlation_method,
                "interval_mode": feature_interval_mode,
                "feature_config": str(feature_config) if feature_config is not None else None,
                "auto_confirm_feature_defaults": auto_confirm_feature_defaults,
            },
            models=models,
            random_seed=random_seed,
        )
        if goal is None:
            return _needs_confirmation(
                "WORKFLOW_GOAL_REQUIRED",
                "Please confirm the goal: read, qc, split, preprocess, feature, classification, or regression.",
                backend,
                required_fields=["task_goal"],
                workflow_plan=workflow_plan,
            )
        if goal not in SUPPORTED_GOALS:
            return _blocked("WORKFLOW_GOAL_UNSUPPORTED", "Unsupported workflow goal.", backend, task_goal=task_goal)
        feature_confirmation = (
            _workflow_feature_confirmation(
                feature_method,
                n_components=feature_n_components,
                top_k=feature_top_k,
                score_threshold=feature_score_threshold,
                n_intervals=feature_n_intervals,
                n_runs=feature_n_runs,
                sample_ratio=feature_sample_ratio,
                cv=feature_cv,
                random_state=feature_random_state,
            )
            if goal in {"feature", "classification", "regression", "modeling"}
            else []
        )
        if feature_confirmation and not auto_confirm_feature_defaults and feature_config is None:
            return _needs_confirmation(
                "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED",
                f"Please confirm key parameters before running {feature_method}.",
                backend,
                required_fields=[str(item["field"]) for item in feature_confirmation],
                workflow_plan=workflow_plan,
                confirmation_required=feature_confirmation,
            )

        stage_outputs: dict[str, str] = {}
        current_package = _resolve_package_dir(package_dir=package_dir, data_contract=data_contract, input_path=input_path)
        current_contract = _contract_path(current_package)
        current_preprocess_contract: Path | None = None
        current_feature_contract: Path | None = None

        if current_package is None:
            if input_path is None:
                return _blocked("WORKFLOW_INPUT_REQUIRED", "Provide input_path, package_dir, or data_contract.", backend)
            reader_dir = layout.stage_dirs["reader"]
            reader_response = _call_reader(
                runner,
                input_path=input_path,
                output_dir=reader_dir,
                sample_orientation=reader_sample_orientation,
                label_column=reader_label_column,
                target_columns=reader_target_columns,
                sample_id_column=reader_sample_id_column,
                sample_id_column_index=reader_sample_id_column_index,
                spectral_start_column=reader_spectral_start_column,
                spectral_end_column=reader_spectral_end_column,
                band_type=reader_band_type,
                band_unit=reader_band_unit,
                max_auto_columns=reader_max_auto_columns,
                max_spectral_columns=reader_max_spectral_columns,
                wide_table_mode=reader_wide_table_mode,
                confirm_wide_table=reader_confirm_wide_table,
                confirm_read_plan=reader_confirm_read_plan,
                task_type=_task_type_for_reader(goal, task_type),
                overwrite=overwrite,
                backend=backend,
            )
            blocked = _propagate_if_not_ready("reader", reader_response, backend)
            if blocked:
                return blocked
            current_package = reader_dir
            current_contract = reader_dir / "data_contract.json"
            stage_outputs["reader"] = str(current_contract)
        else:
            stage_outputs.update(_reused_stage_outputs(current_package, current_contract))

        if goal == "read":
            return _write_and_return(layout, goal, stage_outputs, str(current_contract), warnings, backend)

        if _should_run_qc(goal=goal, include_qc=include_qc, skip_qc=skip_qc, stage_outputs=stage_outputs):
            qc_dir = layout.stage_dirs["qc"]
            qc_response = _call_qc(runner, package_dir=current_package, mode=qc_mode, output_dir=qc_dir, overwrite=overwrite, backend=backend)
            blocked = _propagate_if_not_ready("qc", qc_response, backend)
            if blocked:
                return blocked
            qc_result = qc_dir / "qc_result.json"
            qc_contract = qc_dir / "qc_contract.json"
            stage_outputs["qc"] = str(qc_result if qc_result.exists() else qc_contract if qc_contract.exists() else qc_response.get("result", {}).get("qc_result", qc_dir))
            downstream_package = _qc_next_package(qc_response, qc_result)
            if downstream_package is not None:
                current_package = downstream_package
                current_contract = downstream_package / "data_contract.json"
            if goal == "qc":
                return _write_and_return(layout, goal, stage_outputs, stage_outputs["qc"], warnings, backend)

        if goal in SPLIT_PREPARATION_GOALS | {"preprocess", "feature", "classification", "regression", "modeling"} and split_contract is None:
            split_missing = _missing_split_fields(split_method=split_method, split_ratio=split_ratio)
            if "split_ratio" in split_missing:
                return _needs_confirmation(
                    "SPLIT_RATIO_REQUIRED",
                    "Please confirm the holdout split ratio, for example 8:2, 7:3, or 6:2:2.",
                    backend,
                    required_fields=["split_ratio"],
                    workflow_plan=workflow_plan,
                )
            if "split_method" in split_missing:
                return _needs_confirmation(
                    "SPLIT_METHOD_REQUIRED",
                    "Please confirm the split method. For classification with labels, stratified is usually recommended.",
                    backend,
                    required_fields=["split_method"],
                    workflow_plan=workflow_plan,
                )
            split_dir = layout.stage_dirs["splitter"]
            split_response = _call_splitter(
                runner,
                package_dir=current_package,
                output_dir=split_dir,
                method=split_method,
                ratio=split_ratio,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                n_splits=n_splits or 5,
                n_repeats=n_repeats or 100,
                shuffle=shuffle,
                confirm_incomplete_ratio=confirm_incomplete_split_ratio,
                random_seed=random_seed,
                overwrite=overwrite,
                backend=backend,
            )
            blocked = _propagate_if_not_ready("splitter", split_response, backend)
            if blocked:
                return blocked
            split_contract = split_dir / "split_contract.json"
            stage_outputs["splitter"] = str(split_contract)
        elif split_contract is not None:
            stage_outputs["splitter"] = str(split_contract)

        if goal in SPLIT_PREPARATION_GOALS:
            return _write_and_return(layout, goal, stage_outputs, str(split_contract), warnings, backend)

        preprocess_none = _is_none_selection(preprocess_methods)
        should_preprocess = goal == "preprocess" or (preprocess_methods is not None and not preprocess_none)
        if goal in {"classification", "regression", "modeling"} and preprocess_methods is None:
            return _needs_confirmation(
                "PREPROCESS_DECISION_REQUIRED",
                "Please confirm preprocessing: none for baseline, or a method sequence such as snv, msc, sg_smoothing, or sg_smoothing,snv.",
                backend,
                required_fields=["preprocess_methods"],
                workflow_plan=workflow_plan,
            )
        should_preprocess = should_preprocess or (goal in {"classification", "regression", "modeling"} and not preprocess_none)
        if should_preprocess:
            if preprocess_methods is None:
                return _needs_confirmation(
                    "PREPROCESS_METHOD_REQUIRED",
                    "Please choose preprocessing methods: none, snv, msc, sg_smoothing, first_derivative, second_derivative, mean_centering, or standardization.",
                    backend,
                    required_fields=["preprocess_methods"],
                    workflow_plan=workflow_plan,
                )
            preprocess_dir = layout.stage_dirs["preprocess"]
            preprocess_response = _call_preprocess(
                runner,
                package_dir=current_package,
                split_contract=split_contract,
                output_dir=preprocess_dir,
                methods=preprocess_methods,
                window_length=preprocess_window_length,
                polyorder=preprocess_polyorder,
                sigma=preprocess_sigma,
                poly_degree=preprocess_poly_degree,
                als_lambda=preprocess_als_lambda,
                als_p=preprocess_als_p,
                als_iter=preprocess_als_iter,
                band_range=preprocess_band_range,
                remove_band_ranges=preprocess_remove_band_ranges,
                confirm_baseline=confirm_baseline,
                confirm_absorbance=confirm_absorbance,
                confirm_band_change=confirm_band_change,
                confirm_unsplit_fit=confirm_unsplit_preprocess_fit,
                overwrite=overwrite,
                backend=backend,
            )
            blocked = _propagate_if_not_ready("preprocess", preprocess_response, backend)
            if blocked:
                return blocked
            current_package = preprocess_dir
            current_contract = _preprocess_final_contract(preprocess_dir)
            current_preprocess_contract = current_contract if current_contract.name == "preprocess_contract.json" else None
            stage_outputs["preprocess"] = str(current_contract)
        elif preprocess_none:
            stage_outputs["preprocess"] = "skipped_none"

        if goal == "preprocess":
            return _write_and_return(layout, goal, stage_outputs, str(current_contract), warnings, backend)

        inferred_feature_none = goal in {"classification", "regression", "modeling"} and feature_method is None and models is not None and preprocess_methods is not None
        feature_none = _is_none_selection(feature_method)
        should_feature = goal == "feature" or (feature_method is not None and not feature_none)
        if goal in {"classification", "regression", "modeling"} and feature_method is None and not inferred_feature_none:
            return _needs_confirmation(
                "FEATURE_DECISION_REQUIRED",
                "Please confirm feature engineering: none for baseline, or pca, variance_threshold, select_by_band_range, or select_by_band_indices.",
                backend,
                required_fields=["feature_method"],
                workflow_plan=workflow_plan,
            )
        should_feature = should_feature or (goal in {"classification", "regression", "modeling"} and not feature_none and not inferred_feature_none)
        if should_feature:
            if feature_method is None:
                return _needs_confirmation(
                    "FEATURE_METHOD_REQUIRED",
                    "Please choose a feature method such as none, pca, pls_latent_variables, vip, select_k_best, interval_pls, spa, cars, uve, or mcuve.",
                    backend,
                    required_fields=["feature_method"],
                    workflow_plan=workflow_plan,
                )
            feature_dir = layout.stage_dirs["feature"]
            feature_response = _call_feature(
                runner,
                package_dir=current_package,
                preprocess_contract=current_preprocess_contract if _contract_split_type(current_preprocess_contract) in {"cross_validation", "repeated_holdout"} else None,
                split_contract=split_contract,
                output_dir=feature_dir,
                method=feature_method,
                n_components=feature_n_components,
                explained_variance=feature_explained_variance,
                variance_threshold=feature_variance_threshold,
                band_min=feature_band_min,
                band_max=feature_band_max,
                band_indices=feature_band_indices,
                feature_names=feature_names,
                index_base=feature_index_base,
                top_k=feature_top_k,
                score_threshold=feature_score_threshold,
                n_intervals=feature_n_intervals,
                n_runs=feature_n_runs,
                sample_ratio=feature_sample_ratio,
                cv=feature_cv,
                random_state=feature_random_state if feature_random_state is not None else random_seed,
                task_type=task_type,
                correlation_method=feature_correlation_method,
                interval_mode=feature_interval_mode,
                feature_config=feature_config,
                auto_confirm_feature_defaults=auto_confirm_feature_defaults,
                confirm_unsplit_fit=confirm_unsplit_feature_fit,
                overwrite=overwrite,
                backend=backend,
            )
            blocked = _propagate_if_not_ready("feature", feature_response, backend)
            if blocked:
                return blocked
            if (feature_dir / "data_contract.json").exists():
                audit_blocked = _audit_feature_output(feature_dir, warnings, backend)
                if audit_blocked:
                    return audit_blocked
            current_package = feature_dir
            current_contract = _feature_final_contract(feature_dir)
            current_feature_contract = current_contract if current_contract.name == "feature_contract.json" else None
            stage_outputs["feature"] = str(current_contract)
        elif feature_none or inferred_feature_none:
            stage_outputs["feature"] = "skipped_none"

        if goal == "feature":
            return _write_and_return(layout, goal, stage_outputs, str(current_contract), warnings, backend)

        if goal in {"classification", "regression", "modeling"}:
            selected_task = _task_type_for_modeling(goal, task_type, current_contract)
            if selected_task is None:
                return _needs_confirmation("TASK_TYPE_REQUIRED", "Please confirm task type: classification, regression, or multi_target_regression.", backend, required_fields=["task_type"], workflow_plan=workflow_plan)
            if models is None:
                return _needs_confirmation(
                    "MODEL_TYPE_REQUIRED",
                    "Please choose models. Classification baseline candidates: logistic_regression, linear_svm, svm, lda, random_forest_classifier, knn_classifier. Regression: plsr, ridge, svr, random_forest_regressor.",
                    backend,
                    required_fields=["models"],
                    workflow_plan=workflow_plan,
                )
            selected_modeling_mode = _resolve_modeling_mode(
                modeling_mode,
                task_type=selected_task,
                split_contract=Path(split_contract) if split_contract is not None else None,
                split_method=split_method,
                models=models,
            )
            model_dir = layout.stage_dirs["modeling"]
            model_response = _call_modeling(
                runner,
                package_dir=current_package,
                split_contract=split_contract,
                feature_contract=current_feature_contract,
                preprocess_contract=current_preprocess_contract if current_feature_contract is None else None,
                output_dir=model_dir,
                task_type=selected_task,
                models=models,
                model_config=model_config,
                modeling_mode=selected_modeling_mode,
                model_parameters={
                    "n_components": model_n_components,
                    "embedding_dim": model_embedding_dim,
                    "epochs": model_epochs,
                    "batch_size": model_batch_size,
                    "alpha": model_alpha,
                    "lr": model_lr,
                    "kernel": model_kernel,
                    "preprojection": model_preprojection,
                    "encoder_type": model_encoder_type,
                    "metric": model_metric,
                    "temperature": model_temperature,
                    "device": model_device,
                },
                checkpoint_per_model=checkpoint_per_model or selected_modeling_mode == "repeated_classifier_comparison",
                candidate_model_set_source=(
                    candidate_model_set_source
                    or ("workflow_auto_repeated_classifier_comparison" if selected_modeling_mode == "repeated_classifier_comparison" else None)
                ),
                auto_confirm_model_defaults=auto_confirm_model_defaults,
                evaluation_mode=("validation_only" if require_test_confirmation and not confirm_test_evaluation else "final"),
                require_test_confirmation=require_test_confirmation,
                confirm_test_evaluation=confirm_test_evaluation,
                confirm_confirmatory_test_evaluation=confirm_confirmatory_test_evaluation,
                random_seed=random_seed,
                overwrite=overwrite,
                backend=backend,
            )
            blocked = _propagate_if_not_ready("modeling", model_response, backend)
            if blocked:
                return blocked
            final_output = _modeling_final_output(model_dir)
            stage_outputs["modeling"] = str(final_output)
            if require_test_confirmation and not confirm_test_evaluation:
                return _write_test_confirmation(layout, selected_task, stage_outputs, str(final_output), warnings, backend)
            return _write_and_return(layout, selected_task, stage_outputs, str(final_output), warnings, backend)

        return _blocked("WORKFLOW_GOAL_UNSUPPORTED", "Unsupported workflow goal.", backend, task_goal=task_goal)
    except Exception as exc:
        return error_response("run_spectral_workflow", f"Workflow failed: {exc}", backend=backend, code="WORKFLOW_FAILED", result={"status": "blocked"}, details={"error": str(exc)}, warnings=warnings)


def _write_and_return(layout: Any, task_goal: str, stage_outputs: dict[str, str], final_output: str, warnings: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    root = layout.run_dir
    relative_outputs = relative_stage_outputs(stage_outputs, root)
    relative_final = relative_stage_outputs({"final": final_output}, root).get("final", final_output)
    result = update_workflow_result(
        result_path=root / "workflow_result.json",
        task_goal=task_goal,
        stage_outputs=stage_outputs,
        stage_outputs_relative=relative_outputs,
        final_output=final_output,
        final_output_relative=relative_final,
        workflow_plan=str(root / "workflow_plan.json") if (root / "workflow_plan.json").exists() else None,
        run_id=layout.run_id,
        dataset_name=layout.dataset_name,
        run_dir=str(root.resolve()),
        output_root=str(layout.output_root.resolve()) if layout.output_root else None,
        warnings=warnings,
    )
    write_run_manifest(layout, task_goal=task_goal, status=result["workflow_status"], parameters={"final_output": final_output})
    update_runs_index(layout, task_goal=task_goal, status=result["workflow_status"], stage_outputs=stage_outputs, final_output=final_output)
    return ok_response("run_spectral_workflow", result, backend=backend, warnings=warnings)


def _write_test_confirmation(layout: Any, task_goal: str, stage_outputs: dict[str, str], final_output: str, warnings: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    root = layout.run_dir
    confirmation_required = [
        {
            "field": "test_evaluation",
            "reason": "Validation-only model selection is complete; the isolated test split requires explicit final-evaluation confirmation.",
            "options": ["rerun the same workflow with --confirm-test-evaluation"],
        }
    ]
    result = update_workflow_result(
        result_path=root / "workflow_result.json",
        task_goal=task_goal,
        stage_outputs=stage_outputs,
        stage_outputs_relative=relative_stage_outputs(stage_outputs, root),
        final_output=final_output,
        final_output_relative=relative_stage_outputs({"final": final_output}, root).get("final", final_output),
        workflow_plan=str(root / "workflow_plan.json") if (root / "workflow_plan.json").exists() else None,
        workflow_status="needs_confirmation",
        confirmation_required=confirmation_required,
        run_id=layout.run_id,
        dataset_name=layout.dataset_name,
        run_dir=str(root.resolve()),
        output_root=str(layout.output_root.resolve()) if layout.output_root else None,
        warnings=warnings,
    )
    result["status"] = "needs_confirmation"
    write_run_manifest(layout, task_goal=task_goal, status="needs_confirmation", parameters={"final_output": final_output, "test_accessed": False})
    update_runs_index(layout, task_goal=task_goal, status="needs_confirmation", stage_outputs=stage_outputs, final_output=final_output)
    return error_response(
        "run_spectral_workflow",
        "Validation selection is complete. Confirm before accessing the isolated test split.",
        backend=backend,
        code="TEST_EVALUATION_CONFIRMATION_REQUIRED",
        result=result,
        warnings=warnings,
    )


def _write_workflow_plan(
    root: Path,
    *,
    task_goal: str | None,
    input_path: str | Path | None,
    package_dir: str | Path | None,
    data_contract: str | Path | None,
    split_contract: str | Path | None,
    include_qc: bool,
    qc_mode: str,
    split_ratio: str | None,
    split_method: str | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    n_splits: int | None,
    n_repeats: int | None,
    shuffle: bool,
    preprocess_methods: str | list[str] | None,
    preprocess_parameters: dict[str, Any] | None,
    feature_method: str | None,
    feature_parameters: dict[str, Any] | None,
    models: str | list[str] | None,
    random_seed: int,
) -> str:
    return create_workflow_plan(
        output_dir=root,
        task_goal=task_goal,
        input_path=input_path,
        package_dir=package_dir,
        data_contract=data_contract,
        split_contract=split_contract,
        include_qc=include_qc,
        qc_mode=qc_mode,
        split_ratio=split_ratio,
        split_method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_splits=n_splits,
        n_repeats=n_repeats,
        shuffle=shuffle,
        preprocess_methods=preprocess_methods,
        preprocess_parameters=_non_empty_params(preprocess_parameters),
        feature_method=feature_method,
        feature_parameters=_non_empty_params(feature_parameters),
        models=models,
        random_seed=random_seed,
    )["workflow_plan"]


def _stage_plan(
    stage: str,
    status: str,
    *,
    parameters: dict[str, Any] | None = None,
    required_fields: list[str] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "parameters": parameters or {},
        "confirmation": {
            "required": status == "pending_user_decision",
            "status": "pending_user_decision" if status == "pending_user_decision" else "not_required",
            "required_fields": required_fields or [],
        },
    }
    if skip_reason:
        payload["skip_reason"] = skip_reason
    return payload


def _call_reader(
    runner: StageRunner | None,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    sample_orientation: str | None,
    label_column: str | None,
    target_columns: list[str] | None,
    sample_id_column: str | None,
    sample_id_column_index: int | None,
    spectral_start_column: str | int | None,
    spectral_end_column: str | int | None,
    band_type: str | None,
    band_unit: str | None,
    max_auto_columns: int,
    max_spectral_columns: int,
    wide_table_mode: str,
    confirm_wide_table: bool,
    confirm_read_plan: bool,
    task_type: str | None,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.reader.workflow import read_spectral_dataset

        return read_spectral_dataset(
            input_path=input_path,
            output_dir=output_dir,
            sample_orientation=sample_orientation,
            label_column=label_column,
            target_columns=target_columns,
            sample_id_column=sample_id_column,
            sample_id_column_index=sample_id_column_index,
            spectral_start_column=spectral_start_column,
            spectral_end_column=spectral_end_column,
            band_type=band_type,
            band_unit=band_unit,
            max_auto_columns=max_auto_columns,
            max_spectral_columns=max_spectral_columns,
            wide_table_mode=wide_table_mode,
            confirm_wide_table=confirm_wide_table,
            confirm_read_plan=confirm_read_plan,
            task_type=task_type,
            overwrite=overwrite,
            backend=backend,
        )
    args = ["--input", str(input_path), "--output-dir", str(output_dir)]
    _add_optional(args, "--sample-orientation", sample_orientation)
    _add_optional(args, "--label-column", label_column)
    _add_optional(args, "--target-columns", _join_csv(target_columns))
    _add_optional(args, "--sample-id-column", sample_id_column)
    _add_optional(args, "--sample-id-column-index", sample_id_column_index)
    _add_column_boundary(args, "--spectral-start-column", "--spectral-start-column-index", spectral_start_column)
    _add_column_boundary(args, "--spectral-end-column", "--spectral-end-column-index", spectral_end_column)
    _add_optional(args, "--band-type", band_type)
    _add_optional(args, "--band-unit", band_unit)
    _add_optional(args, "--max-auto-columns", max_auto_columns)
    _add_optional(args, "--max-spectral-columns", max_spectral_columns)
    _add_optional(args, "--wide-table-mode", wide_table_mode)
    _add_flag(args, "--confirm-wide-table", confirm_wide_table)
    _add_flag(args, "--confirm-read-plan", confirm_read_plan)
    _add_optional(args, "--task-type", task_type)
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("reader", args)


def _call_qc(
    runner: StageRunner | None,
    *,
    package_dir: str | Path,
    mode: str,
    output_dir: str | Path,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.qc.workflow import qc_spectral_package

        return qc_spectral_package(package_dir=package_dir, mode=mode, output_dir=output_dir, overwrite=overwrite, backend=backend)
    args = ["--package-dir", str(package_dir), "--mode", mode, "--output-dir", str(output_dir)]
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("qc", args)


def _call_splitter(
    runner: StageRunner | None,
    *,
    package_dir: str | Path,
    output_dir: str | Path,
    method: str | None,
    ratio: str | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    n_splits: int,
    n_repeats: int,
    shuffle: bool,
    confirm_incomplete_ratio: bool,
    random_seed: int,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.splitter.workflow import split_spectral_package

        return split_spectral_package(
            package_dir=package_dir,
            output_dir=output_dir,
            method=method,
            ratio=ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_splits=n_splits,
            n_repeats=n_repeats,
            shuffle=shuffle,
            confirm_incomplete_ratio=confirm_incomplete_ratio,
            random_seed=random_seed,
            overwrite=overwrite,
            backend=backend,
        )
    args = ["--package-dir", str(package_dir), "--output-dir", str(output_dir), "--random-seed", str(random_seed)]
    _add_optional(args, "--method", method)
    _add_optional(args, "--ratio", ratio)
    _add_optional(args, "--train-ratio", train_ratio)
    _add_optional(args, "--val-ratio", val_ratio)
    _add_optional(args, "--test-ratio", test_ratio)
    _add_optional(args, "--n-splits", n_splits)
    _add_optional(args, "--n-repeats", n_repeats)
    _add_flag(args, "--confirm-incomplete-ratio", confirm_incomplete_ratio)
    _add_flag(args, "--no-shuffle", not shuffle)
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("splitter", args)


def _call_preprocess(
    runner: StageRunner | None,
    *,
    package_dir: str | Path,
    split_contract: str | Path,
    output_dir: str | Path,
    methods: str | list[str],
    window_length: int | None,
    polyorder: int | None,
    sigma: float | None,
    poly_degree: int | None,
    als_lambda: float | None,
    als_p: float | None,
    als_iter: int | None,
    band_range: str | None,
    remove_band_ranges: str | None,
    confirm_baseline: bool,
    confirm_absorbance: bool,
    confirm_band_change: bool,
    confirm_unsplit_fit: bool,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.preprocess.workflow import preprocess_spectral_package

        return preprocess_spectral_package(
            package_dir=package_dir,
            split_contract=split_contract,
            output_dir=output_dir,
            methods=methods,
            window_length=window_length,
            polyorder=polyorder,
            sigma=sigma,
            poly_degree=poly_degree,
            als_lambda=als_lambda,
            als_p=als_p,
            als_iter=als_iter,
            band_range=band_range,
            remove_band_ranges=remove_band_ranges,
            confirm_baseline=confirm_baseline,
            confirm_absorbance=confirm_absorbance,
            confirm_band_change=confirm_band_change,
            confirm_unsplit_fit=confirm_unsplit_fit,
            overwrite=overwrite,
            backend=backend,
        )
    args = [
        "--package-dir",
        str(package_dir),
        "--split-contract",
        str(split_contract),
        "--output-dir",
        str(output_dir),
        "--methods",
        _join_csv(methods),
    ]
    _add_optional(args, "--window-length", window_length)
    _add_optional(args, "--polyorder", polyorder)
    _add_optional(args, "--sigma", sigma)
    _add_optional(args, "--poly-degree", poly_degree)
    _add_optional(args, "--als-lambda", als_lambda)
    _add_optional(args, "--als-p", als_p)
    _add_optional(args, "--als-iter", als_iter)
    _add_optional(args, "--band-range", band_range)
    _add_optional(args, "--remove-band-ranges", remove_band_ranges)
    _add_flag(args, "--confirm-baseline", confirm_baseline)
    _add_flag(args, "--confirm-absorbance", confirm_absorbance)
    _add_flag(args, "--confirm-band-change", confirm_band_change)
    _add_flag(args, "--confirm-unsplit-fit", confirm_unsplit_fit)
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("preprocess", args)


def _call_feature(
    runner: StageRunner | None,
    *,
    package_dir: str | Path,
    preprocess_contract: str | Path | None,
    split_contract: str | Path,
    output_dir: str | Path,
    method: str,
    n_components: int | None,
    explained_variance: float | None,
    variance_threshold: float | None,
    band_min: float | None,
    band_max: float | None,
    band_indices: str | None,
    feature_names: str | None,
    index_base: int,
    top_k: int | None,
    score_threshold: float | None,
    n_intervals: int | None,
    n_runs: int | None,
    sample_ratio: float | None,
    cv: int | None,
    random_state: int | None,
    task_type: str | None,
    correlation_method: str | None,
    interval_mode: str | None,
    feature_config: str | Path | None,
    auto_confirm_feature_defaults: bool,
    confirm_unsplit_fit: bool,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.feature.workflow import feature_spectral_package

        return feature_spectral_package(
            package_dir=package_dir,
            preprocess_contract=preprocess_contract,
            split_contract=split_contract,
            output_dir=output_dir,
            method=method,
            n_components=n_components,
            explained_variance=explained_variance,
            variance_threshold=variance_threshold,
            band_min=band_min,
            band_max=band_max,
            band_indices=band_indices,
            feature_names=feature_names,
            index_base=index_base,
            top_k=top_k,
            score_threshold=score_threshold,
            n_intervals=n_intervals,
            n_runs=n_runs,
            sample_ratio=sample_ratio,
            cv=cv,
            random_state=random_state,
            task_type=task_type,
            correlation_method=correlation_method,
            interval_mode=interval_mode,
            feature_config=feature_config,
            auto_confirm_feature_defaults=auto_confirm_feature_defaults,
            confirm_unsplit_fit=confirm_unsplit_fit,
            overwrite=overwrite,
            backend=backend,
        )
    args = []
    if preprocess_contract is not None:
        args.extend(["--preprocess-contract", str(preprocess_contract)])
    else:
        args.extend(["--package-dir", str(package_dir)])
    args.extend([
        "--split-contract",
        str(split_contract),
        "--output-dir",
        str(output_dir),
        "--method",
        method,
    ])
    _add_optional(args, "--n-components", n_components)
    _add_optional(args, "--explained-variance", explained_variance)
    _add_optional(args, "--variance-threshold", variance_threshold)
    _add_optional(args, "--band-min", band_min)
    _add_optional(args, "--band-max", band_max)
    _add_optional(args, "--band-indices", band_indices)
    _add_optional(args, "--feature-names", feature_names)
    _add_optional(args, "--index-base", index_base)
    _add_optional(args, "--top-k", top_k)
    _add_optional(args, "--score-threshold", score_threshold)
    _add_optional(args, "--n-intervals", n_intervals)
    _add_optional(args, "--n-runs", n_runs)
    _add_optional(args, "--sample-ratio", sample_ratio)
    _add_optional(args, "--cv", cv)
    _add_optional(args, "--random-state", random_state)
    _add_optional(args, "--task-type", task_type)
    _add_optional(args, "--correlation-method", correlation_method)
    _add_optional(args, "--interval-mode", interval_mode)
    _add_optional(args, "--feature-config", feature_config)
    _add_flag(args, "--auto-confirm-feature-defaults", auto_confirm_feature_defaults)
    _add_flag(args, "--confirm-unsplit-fit", confirm_unsplit_fit)
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("feature", args)


def _call_modeling(
    runner: StageRunner | None,
    *,
    package_dir: str | Path,
    split_contract: str | Path,
    feature_contract: str | Path | None,
    preprocess_contract: str | Path | None,
    output_dir: str | Path,
    task_type: str,
    models: str | list[str],
    model_config: str | Path | None,
    modeling_mode: str,
    model_parameters: dict[str, Any] | None,
    checkpoint_per_model: bool,
    candidate_model_set_source: str | None,
    auto_confirm_model_defaults: bool,
    evaluation_mode: str,
    require_test_confirmation: bool,
    confirm_test_evaluation: bool,
    confirm_confirmatory_test_evaluation: bool,
    random_seed: int,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    if runner is None:
        from spectral_core.modeling.workflow import model_spectral_package

        return model_spectral_package(
            package_dir=package_dir,
            split_contract=split_contract,
            feature_contract=feature_contract,
            preprocess_contract=preprocess_contract,
            output_dir=output_dir,
            task_type=task_type,
            models=models,
            model_config=model_config,
            mode=modeling_mode,
            model_parameters=model_parameters,
            checkpoint_per_model=checkpoint_per_model,
            candidate_model_set_source=candidate_model_set_source,
            auto_confirm_model_defaults=auto_confirm_model_defaults,
            evaluation_mode=evaluation_mode,
            require_test_confirmation=require_test_confirmation,
            confirm_test_evaluation=confirm_test_evaluation,
            confirm_confirmatory_test_evaluation=confirm_confirmatory_test_evaluation,
            random_seed=random_seed,
            overwrite=overwrite,
            backend=backend,
        )
    args = []
    if feature_contract is not None:
        args.extend(["--feature-contract", str(feature_contract)])
    elif preprocess_contract is not None:
        args.extend(["--preprocess-contract", str(preprocess_contract)])
    else:
        args.extend(["--package-dir", str(package_dir), "--split-contract", str(split_contract)])
    args.extend([
        "--output-dir",
        str(output_dir),
        "--task-type",
        task_type,
        "--mode",
        modeling_mode,
        "--models",
        _join_csv(models),
        "--random-seed",
        str(random_seed),
    ])
    _add_optional(args, "--model-config", model_config)
    _add_optional(args, "--candidate-model-set-source", candidate_model_set_source)
    if model_parameters:
        _add_optional(args, "--model-n-components", model_parameters.get("n_components"))
        _add_optional(args, "--model-embedding-dim", model_parameters.get("embedding_dim"))
        _add_optional(args, "--model-epochs", model_parameters.get("epochs"))
        _add_optional(args, "--model-batch-size", model_parameters.get("batch_size"))
        _add_optional(args, "--model-alpha", model_parameters.get("alpha"))
        _add_optional(args, "--model-lr", model_parameters.get("lr"))
        _add_optional(args, "--model-kernel", model_parameters.get("kernel"))
        _add_optional(args, "--model-preprojection", model_parameters.get("preprojection"))
        _add_optional(args, "--model-encoder-type", model_parameters.get("encoder_type"))
        _add_optional(args, "--model-metric", model_parameters.get("metric"))
        _add_optional(args, "--model-temperature", model_parameters.get("temperature"))
        _add_optional(args, "--model-device", model_parameters.get("device"))
    _add_flag(args, "--auto-confirm-model-defaults", auto_confirm_model_defaults)
    _add_optional(args, "--evaluation-mode", evaluation_mode)
    _add_flag(args, "--disable-model-selection", modeling_mode == "repeated_classifier_comparison")
    _add_flag(args, "--checkpoint-per-model", checkpoint_per_model)
    _add_flag(args, "--require-test-confirmation", require_test_confirmation)
    _add_flag(args, "--confirm-test-evaluation", confirm_test_evaluation)
    _add_flag(args, "--confirm-confirmatory-test-evaluation", confirm_confirmatory_test_evaluation)
    _add_flag(args, "--overwrite", overwrite)
    args.append("--json")
    return runner("modeling", args)


def _run_stage_script(stage: str, args: list[str], log_dir: str | Path | None = None) -> dict[str, Any]:
    script = _stage_script(stage)
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_dir is not None:
        _write_stage_log(log_dir, stage, args, completed.stdout, completed.stderr, completed.returncode)
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return error_response(
                "run_spectral_workflow",
                f"{stage} emitted non-JSON output: {exc}",
                backend="skills",
                code="STAGE_OUTPUT_NOT_JSON",
                result={"status": "blocked", "blocked_stage": stage},
                details={"stdout": completed.stdout, "stderr": completed.stderr},
            )
    else:
        payload = error_response(
            "run_spectral_workflow",
            f"{stage} emitted no JSON output.",
            backend="skills",
            code="STAGE_OUTPUT_EMPTY",
            result={"status": "blocked", "blocked_stage": stage},
            details={"stderr": completed.stderr},
        )
    if completed.returncode != 0 and payload.get("ok"):
        return error_response(
            "run_spectral_workflow",
            f"{stage} exited with code {completed.returncode}.",
            backend="skills",
            code="STAGE_EXIT_FAILED",
            result={"status": "blocked", "blocked_stage": stage, "stage_result": payload.get("result") or {}},
            details={"stderr": completed.stderr},
            warnings=payload.get("warnings", []),
        )
    return payload


def _write_stage_log(log_dir: str | Path, stage: str, args: list[str], stdout: str, stderr: str, returncode: int) -> None:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_stage = stage.replace("/", "_").replace("\\", "_")
    (root / f"{safe_stage}.log").write_text(stdout or "", encoding="utf-8")
    if stderr:
        (root / f"{safe_stage}.stderr.log").write_text(stderr, encoding="utf-8")
    (root / f"{safe_stage}.command.json").write_text(
        json.dumps({"stage": stage, "args": args, "returncode": returncode}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stage_script(stage: str) -> Path:
    scripts = {
        "reader": SKILLS_DIR / "spectral-reader" / "scripts" / "read_spectral_dataset.py",
        "qc": SKILLS_DIR / "spectral-check" / "scripts" / "qc_spectral_package.py",
        "splitter": SKILLS_DIR / "spectral-splitter" / "scripts" / "split_spectral_package.py",
        "preprocess": SKILLS_DIR / "spectral-preprocess" / "scripts" / "preprocess_spectral_package.py",
        "feature": SKILLS_DIR / "spectral-feature" / "scripts" / "feature_spectral_package.py",
        "modeling": SKILLS_DIR / "spectral-modeling" / "scripts" / "model_spectral_package.py",
    }
    try:
        script = scripts[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow stage: {stage}") from exc
    if not script.exists():
        raise FileNotFoundError(f"Missing primary entry for {stage}: {script}")
    return script


def _add_optional(args: list[str], name: str, value: Any) -> None:
    if value is not None and value != "":
        args.extend([name, str(value)])


def _add_column_boundary(args: list[str], column_name: str, index_name: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, int) and not isinstance(value, bool):
        args.extend([index_name, str(value)])
    else:
        args.extend([column_name, str(value)])


def _add_flag(args: list[str], name: str, enabled: bool) -> None:
    if enabled:
        args.append(name)


def _non_empty_params(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if not values:
        return None
    filtered = {key: value for key, value in values.items() if value is not None and value is not False and value != ""}
    return filtered or None


def _join_csv(value: str | list[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _resolve_package_dir(*, package_dir: str | Path | None, data_contract: str | Path | None, input_path: str | Path | None) -> Path | None:
    if package_dir is not None:
        return Path(package_dir)
    if data_contract is not None:
        return Path(data_contract).parent
    if input_path is None:
        return None
    path = Path(input_path)
    if path.name == "data_contract.json":
        return path.parent
    if path.is_dir() and (path / "data_contract.json").exists():
        return path
    return None


def _contract_path(package_dir: Path | None) -> Path | None:
    return package_dir / "data_contract.json" if package_dir else None


def _reused_stage_outputs(package_dir: Path, contract_path: Path | None) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if contract_path is not None:
        stage = _stage_name(contract_path)
        if stage in {"reader", "read", "input"}:
            outputs["reader"] = f"reused_from:{contract_path}"
        else:
            outputs[stage] = f"reused_from:{contract_path}"
    qc_result = package_dir.parent / "qc_output" / "qc_result.json"
    if qc_result.exists():
        outputs["qc"] = f"reused_from:{qc_result}"
    return outputs or {"input": str(contract_path) if contract_path is not None else str(package_dir)}


def _default_qc_for_goal(goal: str | None, skip_qc: bool) -> bool:
    if skip_qc or goal is None:
        return False
    return goal in {"qc", *SPLIT_PREPARATION_GOALS, "preprocess", "feature", "classification", "regression", "modeling"}


def _should_run_qc(*, goal: str | None, include_qc: bool, skip_qc: bool, stage_outputs: dict[str, str]) -> bool:
    if goal == "qc":
        return True
    if skip_qc:
        return False
    if "qc" in stage_outputs:
        return False
    return include_qc or _default_qc_for_goal(goal, skip_qc=False)


def _stage_name(contract_path: Path | None) -> str:
    if contract_path is None or not contract_path.exists():
        return "input"
    try:
        contract = load_json_file(contract_path)
    except Exception:
        return "input"
    return str(contract.get("processing_stage") or "input")


def _contract_split_type(contract_path: Path | None) -> str | None:
    if contract_path is None or not contract_path.exists():
        return None
    try:
        return str(load_json_file(contract_path).get("split_type") or "")
    except Exception:
        return None


def _resolve_modeling_mode(
    requested: str | None,
    *,
    task_type: str | None,
    split_contract: Path | None,
    split_method: str | None,
    models: str | list[str] | None,
) -> str:
    normalized = str(requested or "auto").strip().lower().replace("-", "_")
    if normalized in {"standard", "repeated_classifier_comparison"}:
        return normalized
    if normalized not in {"", "auto"}:
        return normalized
    if task_type == "classification" and _is_repeated_or_cv_split(split_contract, split_method) and _models_imply_classifier_comparison(models):
        return "repeated_classifier_comparison"
    return "standard"


def _is_repeated_or_cv_split(split_contract: Path | None, split_method: str | None) -> bool:
    split_type = (_contract_split_type(split_contract) or "").strip().lower().replace("-", "_")
    if split_type in {"repeated_holdout", "cross_validation"}:
        return True
    return _split_type_for_method(split_method) in {"repeated_holdout", "cross_validation"}


def _models_imply_classifier_comparison(models: str | list[str] | None) -> bool:
    if models is None:
        return False
    if isinstance(models, list):
        return len([item for item in models if str(item).strip()]) > 1
    text = str(models).strip().lower().replace("-", "_")
    if text in {"regular", "regular_fast", "regular_full", "compact", "spectral_modeling"}:
        return True
    return "," in text


def _modeling_final_output(model_dir: Path) -> Path:
    for name in [
        "classifier_comparison_contract.json",
        "classifier_metric_summary.csv",
        "modeling_contract.json",
        "cv_modeling_result.json",
        "repeated_modeling_result.json",
        "metric_summary.json",
    ]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return model_dir / "modeling_contract.json"


def _preprocess_final_contract(preprocess_dir: Path) -> Path:
    for name in ["preprocess_contract.json", "data_contract.json"]:
        candidate = preprocess_dir / name
        if candidate.exists():
            return candidate
    return preprocess_dir / "data_contract.json"


def _feature_final_contract(feature_dir: Path) -> Path:
    for name in ["feature_contract.json", "data_contract.json"]:
        candidate = feature_dir / name
        if candidate.exists():
            return candidate
    return feature_dir / "data_contract.json"


def _is_none_selection(value: str | list[str] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        parts = [str(item).strip().lower() for item in value if str(item).strip()]
    return bool(parts) and all(item in {"none", "skip", "no", "no_preprocess", "no_feature"} for item in parts)


def _split_type_for_method(method: str | None) -> str | None:
    if method is None:
        return None
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"kfold", "stratified_kfold", "leave_one_out"}:
        return "cross_validation"
    if normalized in {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"}:
        return "repeated_holdout"
    if normalized == "predefined_split":
        return "predefined"
    return "holdout"


def _missing_split_fields(*, split_method: str | None, split_ratio: str | None) -> list[str]:
    if split_method is None:
        return ["split_method"]
    normalized = str(split_method).strip().lower().replace("-", "_")
    split_type = _split_type_for_method(normalized)
    if split_type == "holdout" and normalized in {"auto", "random", "stratified"} and split_ratio is None:
        return ["split_ratio"]
    return []


def _qc_next_package(response: dict[str, Any], qc_result_path: Path) -> Path | None:
    result = response.get("result") or {}
    candidate = result.get("next_package_for_downstream")
    if candidate is None and qc_result_path.exists():
        try:
            candidate = load_json_file(qc_result_path).get("next_package_for_downstream")
        except Exception:
            candidate = None
    if not candidate:
        return None
    path = Path(str(candidate))
    if (path / "data_contract.json").exists():
        return path
    return None


def _task_type_for_reader(goal: str, task_type: str | None) -> str | None:
    if task_type:
        return task_type
    if goal == "classification":
        return "classification"
    if goal == "regression":
        return "regression"
    return None


def _task_type_for_modeling(goal: str, task_type: str | None, contract_path: Path | None) -> str | None:
    if goal in {"classification", "regression"}:
        return goal
    if task_type:
        return task_type
    if contract_path and contract_path.exists():
        hint = str((load_json_file(contract_path).get("task_hint") or "")).lower()
        if "class" in hint:
            return "classification"
        if "regression" in hint:
            return "regression"
    return None


def _propagate_if_not_ready(stage: str, response: dict[str, Any], backend: str) -> dict[str, Any] | None:
    result = response.get("result") or {}
    status = result.get("status")
    if response.get("ok") and status in {None, "ready", "passed", "warning", "cleaned"}:
        return None
    errors = response.get("errors") or []
    first_error = errors[0] if errors else {}
    code = first_error.get("code") or f"{stage.upper()}_NOT_READY"
    message = first_error.get("message") or result.get("reason") or f"{stage} did not complete."
    out_status = result.get("status") or ("blocked" if not response.get("ok") else "needs_confirmation")
    return error_response(
        "run_spectral_workflow",
        str(message),
        backend=backend,
        code=code,
        result={"status": out_status, "blocked_stage": stage, "stage_result": result},
        details=first_error.get("details", {}),
        warnings=response.get("warnings", []),
    )


def _audit_feature_output(feature_dir: Path, warnings: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    from spectral_core.feature.audit import audit_feature_package

    audit = audit_feature_package(feature_dir, repair=True)
    if audit["ok"]:
        if audit.get("repaired"):
            warnings.append(
                {
                    "code": "FEATURE_CONTRACT_REPAIRED",
                    "message": "Feature data_contract.json had repairable count inconsistencies and was repaired before downstream modeling.",
                    "severity": "warning",
                    "details": {"counts": audit.get("counts")},
                }
            )
        return None
    return error_response(
        "run_spectral_workflow",
        "Feature output contract is inconsistent; downstream modeling is blocked until the contract is repaired.",
        backend=backend,
        code="FEATURE_CONTRACT_INCONSISTENT",
        result={"status": "blocked", "blocked_stage": "feature", "stage_result": audit},
        details={"issues": audit.get("issues"), "counts": audit.get("counts")},
        warnings=warnings,
    )


def _needs_confirmation(
    code: str,
    message: str,
    backend: str,
    *,
    required_fields: list[str],
    workflow_plan: str | None = None,
    confirmation_required: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return error_response(
        "run_spectral_workflow",
        message,
        backend=backend,
        code=code,
        result={
            "status": "needs_confirmation",
            "required_fields": required_fields,
            "workflow_plan": workflow_plan,
            "confirmation_required": confirmation_required or [],
        },
    )


def _incomplete_split_ratio_confirmation(ratio: str | None) -> dict[str, Any] | None:
    if ratio is None:
        return None
    normalized = str(ratio).strip().replace("：", ":")
    if not normalized:
        return None
    parts = normalized.split(":")
    if len(parts) >= 3 and any(part.strip() == "" for part in parts):
        return {
            "field": "split_ratio",
            "provided": ratio,
            "reason": "The ratio has a missing segment; workflow must not infer the test ratio silently.",
            "recommended_interpretation": "train:val:test = 6:2:2" if normalized.startswith("6:2:") else None,
            "required_reply": "confirm the complete ratio before executing the split",
        }
    return None


def _workflow_feature_confirmation(
    method: str | None,
    *,
    n_components: int | None,
    top_k: int | None,
    score_threshold: float | None,
    n_intervals: int | None,
    n_runs: int | None,
    sample_ratio: float | None,
    cv: int | None,
    random_state: int | None,
) -> list[dict[str, Any]]:
    normalized = normalize_feature_method(method)
    values = {
        "n_components": n_components,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "n_intervals": n_intervals,
        "n_runs": n_runs,
        "sample_ratio": sample_ratio,
        "cv": cv,
        "random_state": random_state,
    }
    missing = []
    for field in missing_critical_parameters(normalized, values):
        if field == "selection_rule":
            missing.append(
                {
                    "field": "selection_rule",
                    "reason": f"{normalized} requires choosing top_k or score_threshold.",
                    "options": ["top_k=50", "score_threshold=1.0"],
                }
            )
        else:
            recommended = _workflow_feature_parameter_recommendation(normalized, field)
            item = {"field": field, "reason": f"{field} is a key parameter for {normalized}."}
            if recommended is not None:
                item["recommended"] = recommended
            missing.append(item)
    return missing


def _workflow_feature_parameter_recommendation(method: str, field: str) -> Any:
    if field == "n_components":
        return 10
    if field == "random_state":
        return 42
    if field == "n_runs":
        return 100 if method == "mcuve" else 50
    if field == "sample_ratio":
        return 0.8
    if field == "cv":
        return 5
    if field == "n_intervals":
        return 20
    if field == "top_k":
        return 50
    return None


def _blocked(code: str, message: str, backend: str, **details: Any) -> dict[str, Any]:
    return error_response("run_spectral_workflow", message, backend=backend, code=code, result={"status": "blocked"}, details=details)

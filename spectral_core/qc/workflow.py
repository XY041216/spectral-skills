"""Unified compact QC workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.reader.response import error_response, ok_response
from spectral_core.reader.io_utils import write_json_file

from .apply_actions import QCActionError, apply_confirmed_actions
from .basic_checks import run_basic_checks
from .io import QCInputError, load_standard_package
from .detector_registry import list_methods
from .outlier_methods import QCMethodError, run_outlier_methods, summarize_resampling_outlier_control
from .package_writer import QCPackageWriteError, write_standard_package


def qc_spectral_package(
    *,
    package_dir: str | Path,
    mode: str = "check",
    methods: list[str] | None = None,
    output_dir: str | Path | None = None,
    confirm_action: bool = False,
    remove_sample_ids: list[str] | None = None,
    remove_sample_indices: list[int] | None = None,
    remove_band_indices: list[int] | None = None,
    impute_missing: str | None = None,
    cleaning_action: str | None = None,
    cleaning_method: str | None = None,
    cleaning_strategy: str | None = None,
    threshold: Any | None = None,
    n_resamples: int | None = None,
    sample_fraction: float | None = None,
    train_ratio: float | None = None,
    base_model: str | None = None,
    outlier_metric: str | None = None,
    detail_level: str = "full",
    export_details: bool = False,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    try:
        selected_mode = _normalize_mode(mode)
        if mode == "methods":
            return ok_response(
                "qc_spectral_package",
                {
                    "status": "ready",
                    "methods": list_methods(),
                    "standard_scheme_methods": [
                        "contract_consistency_check",
                        "missing_check",
                        "non_numeric_check",
                        "band_axis_check",
                        "constant_band_check",
                        "low_variance_band_check",
                        "label_distribution_check",
                        "sample_intensity_quality",
                        "roughness_check",
                        "spike_check",
                        "baseline_drift_check",
                        "similarity_to_mean",
                        "classwise_similarity",
                        "near_duplicate_check",
                        "pca_hotelling_t2",
                        "pca_q_residual",
                        "mahalanobis_on_pca",
                    ],
                },
                backend=backend,
            )
        package = load_standard_package(package_dir)
        if selected_mode == "check":
            result = run_basic_checks(package, mode="check")
            result = _write_observation_outputs(package, mode=mode, result=result, methods=methods, output_dir=output_dir, detail_level=detail_level, export_details=export_details, backend=backend)
            return ok_response("qc_spectral_package", result, backend=backend)
        if selected_mode == "mark":
            result = run_basic_checks(package, mode="mark")
            if methods:
                method_result = run_outlier_methods(package, methods, parameters=_outlier_parameters(n_resamples=n_resamples, sample_fraction=sample_fraction, train_ratio=train_ratio, base_model=base_model, outlier_metric=outlier_metric, threshold=threshold))
                result["method_pool_results"] = method_result.get("methods", [])
                resampling_summary = summarize_resampling_outlier_control(result["method_pool_results"])
                if resampling_summary:
                    result["resampling_outlier_control"] = resampling_summary
            result = _write_observation_outputs(package, mode=mode, result=result, methods=methods, output_dir=output_dir, detail_level=detail_level, export_details=export_details, backend=backend)
            return ok_response("qc_spectral_package", result, backend=backend)
        if selected_mode == "clean":
            return _apply_mode(
                package,
                package_dir=package_dir,
                methods=methods,
                output_dir=output_dir,
                confirm_action=confirm_action,
                remove_sample_ids=remove_sample_ids,
                remove_sample_indices=remove_sample_indices,
                remove_band_indices=remove_band_indices,
                impute_missing=impute_missing,
                cleaning_action=cleaning_action,
                cleaning_method=cleaning_method,
                cleaning_strategy=cleaning_strategy,
                threshold=threshold,
                n_resamples=n_resamples,
                sample_fraction=sample_fraction,
                train_ratio=train_ratio,
                base_model=base_model,
                outlier_metric=outlier_metric,
                detail_level=detail_level,
                export_details=export_details,
                overwrite=overwrite,
                backend=backend,
            )
        return error_response("qc_spectral_package", f"Unsupported QC mode: {mode}", backend=backend, code="QC_MODE_UNSUPPORTED", result={"status": "blocked"})
    except QCInputError as exc:
        return error_response("qc_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details)
    except QCMethodError as exc:
        return error_response("qc_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details)
    except QCActionError as exc:
        return error_response("qc_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "needs_confirmation"}, details=exc.details)
    except QCPackageWriteError as exc:
        return error_response("qc_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details)


def _write_observation_outputs(
    package: Any,
    *,
    mode: str,
    result: dict[str, Any],
    methods: list[str] | None,
    output_dir: str | Path | None,
    detail_level: str,
    export_details: bool,
    backend: str,
) -> dict[str, Any]:
    detail_level = _normalize_detail_level(detail_level)
    response_payload = result if detail_level == "full" else _compact_qc_result(result)
    if output_dir is None:
        return response_payload
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["output_package"] = None
    payload["next_package_for_downstream"] = str(package.root)
    if detail_level == "full":
        write_json_file(root / "qc_result.json", payload, ensure_ascii=False)
        payload["qc_result"] = str(root / "qc_result.json")
        payload["handoff_ready"] = result.get("status") in {"passed", "warning"}
        return payload
    compact = _compact_qc_result(payload)
    if export_details:
        details_path = root / "qc_details.json"
        write_json_file(details_path, payload, ensure_ascii=False)
        compact["details_ref"] = str(details_path)
    write_json_file(root / "qc_result.json", compact, ensure_ascii=False)
    compact["qc_result"] = str(root / "qc_result.json")
    compact["handoff_ready"] = result.get("status") in {"passed", "warning"}
    return compact


def _apply_mode(
    package: Any,
    *,
    package_dir: str | Path,
    methods: list[str] | None,
    output_dir: str | Path | None,
    confirm_action: bool,
    remove_sample_ids: list[str] | None,
    remove_sample_indices: list[int] | None,
    remove_band_indices: list[int] | None,
    impute_missing: str | None,
    cleaning_action: str | None,
    cleaning_method: str | None,
    cleaning_strategy: str | None,
    threshold: Any | None,
    n_resamples: int | None,
    sample_fraction: float | None,
    train_ratio: float | None,
    base_model: str | None,
    outlier_metric: str | None,
    detail_level: str,
    export_details: bool,
    overwrite: bool,
    backend: str,
) -> dict[str, Any]:
    observation = run_basic_checks(package, mode="clean")
    method_result = run_outlier_methods(package, methods, parameters=_outlier_parameters(n_resamples=n_resamples, sample_fraction=sample_fraction, train_ratio=train_ratio, base_model=base_model, outlier_metric=outlier_metric, threshold=threshold)) if methods else {"methods": []}
    resampling_summary = summarize_resampling_outlier_control(method_result.get("methods", []))
    if resampling_summary:
        observation["resampling_outlier_control"] = resampling_summary
    method_sample_indices = [
        int(item["sample_index"])
        for method in method_result.get("methods", [])
        for item in method.get("outlier_sample_candidates", [])
    ]
    derived_sample_indices, derived_details = _derive_sample_removals(
        package,
        observation=observation,
        cleaning_action=cleaning_action,
        cleaning_strategy=cleaning_strategy,
    )
    combined_indices = sorted(set((remove_sample_indices or []) + method_sample_indices + derived_sample_indices))
    if not (combined_indices or remove_sample_ids or remove_band_indices or (impute_missing and impute_missing != "none")):
        pending = _pending_clean_confirmation(package, observation, action="cleaning_action_missing")
        _write_pending_qc_result(pending, output_dir)
        return error_response(
            "qc_spectral_package",
            "Clean mode requires a confirmed data-changing action and method.",
            backend=backend,
            code="CLEANING_ACTION_NOT_SPECIFIED",
            result=pending,
        )
    if not confirm_action:
        pending = _pending_clean_confirmation(
            package,
            observation,
            action=cleaning_action or ("drop_outlier_samples" if combined_indices or remove_sample_ids else "drop_bands_or_impute"),
            candidate_remove_sample_indices=combined_indices,
            candidate_remove_band_indices=remove_band_indices or [],
            impute_missing=impute_missing or "none",
            cleaning_method=cleaning_method,
            cleaning_strategy=cleaning_strategy,
            threshold=threshold,
            outlier_methods=method_result.get("methods", []),
        )
        _write_pending_qc_result(pending, output_dir)
        return error_response(
            "qc_spectral_package",
            "Data-changing QC actions require explicit confirmation.",
            backend=backend,
            code="ACTION_CONFIRMATION_REQUIRED",
            result=pending,
        )
    if output_dir is None:
        return error_response("qc_spectral_package", "output_dir is required when applying QC actions.", backend=backend, code="OUTPUT_DIR_REQUIRED", result={"status": "blocked"})
    updated, action_summary = apply_confirmed_actions(
        package,
        remove_sample_ids=remove_sample_ids,
        remove_sample_indices=combined_indices,
        remove_band_indices=remove_band_indices,
        impute_missing=impute_missing,
        cleaning_action=cleaning_action,
        cleaning_method=cleaning_method,
        cleaning_strategy=cleaning_strategy,
        threshold=threshold,
        confirmed=confirm_action,
    )
    action_summary["methods_used"] = [method["method_id"] for method in method_result.get("methods", [])]
    if resampling_summary:
        action_summary["resampling_outlier_control"] = resampling_summary
    action_summary["derived_removal_details"] = derived_details
    if derived_details:
        derived_by_id = {item["sample_id"]: item for item in derived_details}
        action_summary["removed_samples"] = [
            {**item, **derived_by_id.get(item["sample_id"], {})}
            for item in action_summary.get("removed_samples", [])
        ]
    root = Path(output_dir)
    cleaned_package = root / "cleaned_package"
    written = write_standard_package(
        updated,
        output_dir=cleaned_package,
        parent_contract=str(Path(package_dir) / "data_contract.json"),
        qc_summary=action_summary,
        overwrite=overwrite,
    )
    input_shape = {"n_samples": package.n_samples, "n_features": package.n_features}
    output_shape = {"n_samples": updated.n_samples, "n_features": updated.n_features}
    cleaning_action_name = action_summary.get("action") or cleaning_action or "confirmed_qc_cleaning"
    log = {
        "stage": "spectral-qc-clean",
        "mode": "clean",
        "status": "cleaned",
        "input_package": str(package.root),
        "output_package": str(cleaned_package),
        "cleaning_actions": [
            {
                "action": cleaning_action_name,
                "method": action_summary.get("method"),
                "threshold": action_summary.get("threshold"),
                "strategy": action_summary.get("strategy"),
                "methods_used": action_summary["methods_used"],
                "decision_source": "user_confirmed_recommendation" if methods else "user_specified",
                "removed_samples_count": action_summary["removed_sample_count"],
                "removed_samples": action_summary.get("removed_samples", []),
                "removed_band_count": action_summary["removed_band_count"],
                "removed_bands": action_summary.get("removed_bands", []),
                "impute_missing": action_summary["impute_missing"],
                "resampling_outlier_control": resampling_summary,
            }
        ],
        "input_shape": input_shape,
        "output_shape": output_shape,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json_file(root / "qc_cleaning_log.json", log, ensure_ascii=False)
    result = dict(observation)
    result.update(
        {
            "mode": "clean",
            "status": "cleaned",
            "output_package": str(cleaned_package),
            "next_package_for_downstream": str(cleaned_package),
            "requires_user_confirmation": False,
            "confirmed_actions": log["cleaning_actions"],
            "cleaned_package": str(cleaned_package),
            "qc_cleaning_log": str(root / "qc_cleaning_log.json"),
            "input_shape": input_shape,
            "output_shape": output_shape,
        }
    )
    if resampling_summary:
        result["resampling_outlier_control"] = {**resampling_summary, "recommended_action": "confirmed_clean"}
    detail_level = _normalize_detail_level(detail_level)
    output_result = result if detail_level == "full" else _compact_qc_result(result)
    if export_details and detail_level != "full":
        details_path = root / "qc_details.json"
        write_json_file(details_path, result, ensure_ascii=False)
        output_result["details_ref"] = str(details_path)
    write_json_file(root / "qc_result.json", output_result, ensure_ascii=False)
    written["qc_result"] = str(root / "qc_result.json")
    written["qc_cleaning_log"] = str(root / "qc_cleaning_log.json")
    written["cleaned_package"] = str(cleaned_package)
    written["output_package"] = str(cleaned_package)
    written["next_package_for_downstream"] = str(cleaned_package)
    written["outlier_methods"] = method_result.get("methods", [])
    return ok_response("qc_spectral_package", written, backend=backend)


def _normalize_mode(mode: str) -> str:
    aliases = {"outliers": "mark", "apply": "clean"}
    return aliases.get(mode, mode)


def _normalize_detail_level(value: str | None) -> str:
    return "full" if str(value or "").lower() in {"full", "verbose", "details"} else "summary"


def _compact_qc_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "schema_version": result.get("schema_version"),
        "stage": result.get("stage", "spectral-qc"),
        "mode": result.get("mode"),
        "status": result.get("status"),
        "input_package": result.get("input_package"),
        "output_package": result.get("output_package"),
        "next_package_for_downstream": result.get("next_package_for_downstream"),
        "package_dir": result.get("package_dir"),
        "data_shape": result.get("data_shape") or result.get("shape"),
        "shape": result.get("shape") or result.get("data_shape"),
        "task_type": result.get("task_type"),
        "summary": result.get("summary", {}),
        "warnings_count": len(result.get("warnings") or []),
        "warnings_preview": (result.get("warnings") or [])[:10],
        "blocked_reasons": result.get("blocked_reasons", []),
        "duplicate_check": result.get("duplicate_check"),
        "global_similarity_risk": result.get("global_similarity_risk"),
        "outlier_detection": _compact_outlier_detection(result.get("outlier_detection") or {}),
        "outlier_groups": result.get("outlier_groups"),
        "resampling_outlier_control": result.get("resampling_outlier_control"),
        "method_pool_results": _compact_method_pool_results(result.get("method_pool_results") or []),
        "pending_confirmation": result.get("pending_confirmation"),
        "recommended_actions": result.get("recommended_actions", []),
        "requires_user_confirmation": result.get("requires_user_confirmation", False),
        "next_step_recommendation": result.get("next_step_recommendation"),
        "confirmed_actions": result.get("confirmed_actions"),
        "cleaned_package": result.get("cleaned_package"),
        "qc_cleaning_log": result.get("qc_cleaning_log"),
    }
    return {key: value for key, value in compact.items() if value is not None}


def _compact_outlier_detection(outlier_detection: dict[str, Any]) -> dict[str, Any]:
    if not outlier_detection:
        return {}
    return {
        "strategy": outlier_detection.get("strategy"),
        "methods_run": outlier_detection.get("methods_run", []),
        "advanced_methods_not_run_by_default": outlier_detection.get("advanced_methods_not_run_by_default", []),
        "recommended_action": outlier_detection.get("recommended_action"),
        "high_confidence_outliers": (outlier_detection.get("high_confidence_outliers") or [])[:20],
        "medium_confidence_outliers": (outlier_detection.get("medium_confidence_outliers") or [])[:20],
        "low_confidence_count": len(outlier_detection.get("low_confidence_outliers") or []),
        "method_selection_options": outlier_detection.get("method_selection_options", []),
    }


def _compact_method_pool_results(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for method in methods:
        item = {
            "method_id": method.get("method_id"),
            "display_name": method.get("display_name"),
            "parameters": method.get("parameters"),
            "threshold": method.get("threshold"),
            "outlier_sample_count": method.get("outlier_sample_count"),
            "outlier_sample_candidates": (method.get("outlier_sample_candidates") or [])[:20],
            "classification_instability_candidates": (method.get("classification_instability_candidates") or [])[:20],
            "resampling_risk_samples": (method.get("resampling_risk_samples") or [])[:20],
            "score_preview": (method.get("outlier_scores") or [])[:20],
            "score_count": len(method.get("outlier_scores") or []),
            "risk_semantics": method.get("risk_semantics"),
            "input_pipeline": method.get("input_pipeline"),
            "evaluation_summary": method.get("evaluation_summary"),
            "confirmation_required_for_removal": method.get("confirmation_required_for_removal"),
        }
        compacted.append({key: value for key, value in item.items() if value is not None})
    return compacted


def _pending_clean_confirmation(
    package: Any,
    observation: dict[str, Any],
    *,
    action: str,
    candidate_remove_sample_indices: list[int] | None = None,
    candidate_remove_band_indices: list[int] | None = None,
    impute_missing: str = "none",
    cleaning_method: str | None = None,
    cleaning_strategy: str | None = None,
    threshold: float | None = None,
    outlier_methods: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = dict(observation)
    result.update(
        {
            "mode": "clean",
            "status": "blocked",
            "blocked_reason": "cleaning_method_not_confirmed" if action != "cleaning_action_missing" else "cleaning_action_not_specified",
            "requires_user_confirmation": True,
            "output_package": None,
            "next_package_for_downstream": str(package.root),
            "pending_confirmation": {
                "action": action,
                "required_decisions": _required_clean_decisions(action, impute_missing=impute_missing),
                "recommended_option": _recommended_clean_option(action),
                "options": _clean_options(action),
                "outlier_detection_method_options": _outlier_detection_method_options() if action in {"drop_outlier_samples", "cleaning_action_missing"} else [],
                "confirmation_requires": _confirmation_requirements(action, impute_missing=impute_missing),
                "recommended_detection_method": "standard_multi_method_consensus" if action in {"drop_outlier_samples", "cleaning_action_missing"} else None,
                "recommended_threshold": "high_confidence_outliers_only" if action in {"drop_outlier_samples", "cleaning_action_missing"} else threshold,
                "recommended_output": "cleaned_package",
                "outlier_candidate_summary": _outlier_candidate_summary(observation),
                "spike_repair_summary": _spike_repair_summary(observation),
                "candidate_remove_sample_indices": candidate_remove_sample_indices or [],
                "candidate_remove_sample_ids": [package.sample_ids[idx] for idx in candidate_remove_sample_indices or []],
                "candidate_remove_band_indices": candidate_remove_band_indices or [],
                "impute_missing": impute_missing,
                "method": cleaning_method,
                "strategy": cleaning_strategy,
                "threshold": threshold,
                "outlier_methods": outlier_methods or [],
            },
            "next_step_recommendation": "confirm_qc_cleaning_or_mark_only",
        }
    )
    return result


def _required_clean_decisions(action: str, *, impute_missing: str) -> list[str]:
    if action == "drop_outlier_samples":
        return ["outlier_detection_method", "threshold_strategy", "deletion_scope", "output_cleaned_package"]
    if action in {"remove_exact_duplicates", "remove_near_duplicates"}:
        return ["duplicate_detection_method", "retention_strategy", "label_conflict_review"]
    if action == "remove_bad_bands":
        return ["bad_band_rule", "threshold_strategy", "deletion_scope"]
    if action in {"repair_spikes", "remove_spike_samples"}:
        return ["spike_detection_method", "window_and_threshold", "repair_or_delete_scope"]
    if impute_missing and impute_missing != "none":
        return ["missing_value_strategy", "imputation_scope", "output_cleaned_package"]
    return ["cleaning_method", "threshold_strategy", "output_cleaned_package"]


def _confirmation_requirements(action: str, *, impute_missing: str) -> list[str]:
    if action in {"drop_outlier_samples", "cleaning_action_missing"}:
        return ["method", "threshold", "deletion_scope", "output_cleaned_package"]
    return _required_clean_decisions(action, impute_missing=impute_missing)


def _recommended_clean_option(action: str) -> str:
    if action == "remove_exact_duplicates":
        return "remove_exact_duplicates_keep_first_when_labels_match"
    if action == "remove_near_duplicates":
        return "group_aware_split_or_keep_representative_after_confirmation"
    if action == "remove_bad_bands":
        return "remove_only_confirmed_high_risk_bands"
    if action in {"repair_spikes", "remove_spike_samples"}:
        return "mark_spikes_first_or_hampel_repair_after_confirmation"
    return "conservative_multi_method_high_confidence_only"


def _clean_options(action: str) -> list[str]:
    if action in {"remove_exact_duplicates", "remove_near_duplicates"}:
        return [
            "mark_only",
            "remove_exact_duplicates_keep_first",
            "remove_near_duplicates_keep_representative",
            "merge_replicate_scans_by_mean",
            "emit_group_recommendations_for_splitter",
            "custom",
        ]
    if action == "remove_bad_bands":
        return [
            "constant_and_severe_low_variance_bands",
            "high_missing_rate_bands",
            "high_spike_frequency_bands",
            "edge_noise_bands",
            "user_specified_band_range",
            "custom",
        ]
    if action in {"repair_spikes", "remove_spike_samples"}:
        return [
            "mark_only_recommended",
            "hampel_filter_after_confirmation",
            "moving_median_replace_after_confirmation",
            "route_to_spectral_preprocess_sg_smoothing",
            "custom",
        ]
    if action in {"drop_outlier_samples", "cleaning_action_missing"}:
        return [
            "standard_multi_method_consensus_high_confidence_only_recommended",
            "pca_t2_q_residual_pca_md_high_confidence_only",
            "mahalanobis_on_pca_with_confirmed_threshold",
            "robust_zscore_mad_iqr_with_confirmed_threshold",
            "half_resampling_outlier_mark_or_clean_after_confirmation",
            "mccv_outlier_mark_or_clean_after_confirmation",
            "intersection_spectral_outlier_and_resampling_risk_recommended_over_mccv_only",
            "mark_only_recommended",
            "custom_method_threshold_and_scope",
        ]
    return [
        "conservative_multi_method_high_confidence_only",
        "pca_t2_q_residual",
        "half_resampling_outlier",
        "mccv_outlier",
        "robust_statistics",
        "classwise_outlier_detection",
        "custom",
    ]


def _outlier_detection_method_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "standard_multi_method_consensus",
            "label": "Default comprehensive strategy",
            "methods": ["robust_zscore", "pca_hotelling_t2", "pca_q_residual", "mahalanobis_on_pca", "similarity_to_mean", "baseline_drift_score", "spike_detection"],
            "recommended_scope": "high_confidence_outliers_only",
        },
        {
            "id": "pca_t2_q_residual_pca_md",
            "label": "PCA chemometric strategy",
            "methods": ["pca_hotelling_t2", "pca_q_residual", "mahalanobis_on_pca"],
            "recommended_scope": "high_confidence_outliers_only",
        },
        {
            "id": "mahalanobis_on_pca",
            "label": "Mahalanobis distance in PCA score space",
            "methods": ["mahalanobis_on_pca"],
            "recommended_scope": "confirmed_threshold_only",
        },
        {
            "id": "robust_zscore_mad_iqr",
            "label": "Robust statistics",
            "methods": ["robust_zscore", "mad", "iqr"],
            "recommended_scope": "confirmed_threshold_only",
        },
        {
            "id": "half_resampling_outlier",
            "label": "HR half resampling stability",
            "methods": ["half_resampling_outlier"],
            "recommended_parameters": {"n_resamples": 100, "sample_fraction": 0.5, "threshold": "percentile_95"},
        },
        {
            "id": "mccv_outlier",
            "label": "MCCV classification stability risk",
            "methods": ["mccv_outlier"],
            "recommended_parameters": {"n_resamples": 100, "train_ratio": 0.7, "threshold": "percentile_95"},
            "recommended_scope": "mark_only_or_intersection_after_manual_review",
            "cleaning_caution": "Do not directly delete MCCV-only candidates; they may be boundary, overlapped, or pipeline-sensitive samples.",
        },
        {
            "id": "intersection_spectral_outlier_and_resampling_risk",
            "label": "Intersection of spectral QC outliers and HR/MCCV stability risk",
            "methods": ["standard_multi_method_consensus", "half_resampling_outlier_or_mccv_outlier"],
            "recommended_scope": "only samples confirmed by both spectral-shape QC and resampling risk",
        },
        {"id": "custom", "label": "Custom method and threshold", "methods": ["custom"]},
    ]


def _outlier_candidate_summary(observation: dict[str, Any]) -> dict[str, Any]:
    groups = observation.get("outlier_groups") or {}
    high = list(groups.get("high_confidence_outliers") or [])
    medium = list(groups.get("medium_confidence_outliers") or [])
    low = list(groups.get("low_confidence_outliers") or [])
    return {
        "high_confidence_count": len(high),
        "high_confidence_sample_ids": high,
        "medium_confidence_count": len(medium),
        "medium_confidence_sample_ids": medium[:20],
        "low_confidence_count": len(low),
        "recommended_options": ["remove_high_confidence_only", "mark_only"],
        "not_recommended": ["remove_medium_confidence_by_default", "remove_low_confidence_by_default"],
    }


def _spike_repair_summary(observation: dict[str, Any]) -> dict[str, Any]:
    spike = _find_check(observation, "spike_check")
    return {
        "spike_candidate_count": int(spike.get("spike_sample_count") or 0),
        "minor_spike_sample_count": int(spike.get("minor_spike_sample_count") or 0),
        "moderate_spike_sample_count": int(spike.get("moderate_spike_sample_count") or 0),
        "severe_spike_sample_count": int(spike.get("severe_spike_sample_count") or 0),
        "recommended_options": ["mark_only", "route_to_spectral_preprocess_if_smoothing_is_needed"],
        "not_recommended": ["delete_or_repair_all_spike_candidates_by_default"],
    }


def _outlier_parameters(
    *,
    n_resamples: int | None,
    sample_fraction: float | None,
    train_ratio: float | None,
    base_model: str | None,
    outlier_metric: str | None,
    threshold: Any | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if n_resamples is not None:
        params["n_resamples"] = n_resamples
    if sample_fraction is not None:
        params["sample_fraction"] = sample_fraction
    if train_ratio is not None:
        params["train_ratio"] = train_ratio
    if base_model:
        params["base_model"] = base_model
    if outlier_metric:
        params["outlier_metric"] = outlier_metric
    if threshold is not None:
        params["threshold"] = threshold
    return params


def _derive_sample_removals(
    package: Any,
    *,
    observation: dict[str, Any],
    cleaning_action: str | None,
    cleaning_strategy: str | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    if cleaning_action not in {"remove_exact_duplicates", "remove_near_duplicates"}:
        return [], []
    duplicate_check = _find_check(observation, "near_duplicate_check")
    exact_conflicts = duplicate_check.get("exact_duplicate_label_conflicts") or []
    strict_conflicts = duplicate_check.get("strict_near_duplicate_label_conflicts") or []
    conflicts = exact_conflicts if cleaning_action == "remove_exact_duplicates" else exact_conflicts + strict_conflicts
    if conflicts:
        raise QCActionError(
            "DUPLICATE_LABEL_CONFLICT_REQUIRES_REVIEW",
            "Duplicate spectra with conflicting labels require manual review before cleaning.",
            label_conflict_pairs=conflicts,
        )
    groups = duplicate_check.get("exact_duplicate_groups") if cleaning_action == "remove_exact_duplicates" else duplicate_check.get("strict_near_duplicate_groups")
    groups = groups or []
    if not groups:
        return [], []
    remove_indices: list[int] = []
    details: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups, start=1):
        indices = sorted(int(idx) for idx in group.get("indices", []) if isinstance(idx, int) or str(idx).isdigit())
        if len(indices) < 2:
            continue
        kept = indices[0]
        for idx in indices[1:]:
            remove_indices.append(idx)
            details.append(
                {
                    "sample_id": package.sample_ids[idx],
                    "sample_index": idx,
                    "group_id": f"dup_group_{group_idx:03d}",
                    "kept_sample_id": package.sample_ids[kept],
                    "reason": cleaning_action,
                    "strategy": cleaning_strategy or "keep_first",
                }
            )
    return sorted(set(remove_indices)), details


def _find_check(result: dict[str, Any], name: str) -> dict[str, Any]:
    for check in result.get("checks", []):
        if check.get("check") == name:
            return check
    return {}


def _write_pending_qc_result(result: dict[str, Any], output_dir: str | Path | None) -> None:
    if output_dir is None:
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json_file(root / "qc_result.json", result, ensure_ascii=False)

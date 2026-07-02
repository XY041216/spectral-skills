"""Unified leakage-safe spectral feature workflow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file
from spectral_core.reader.response import error_response, ok_response

from .io import FeatureInputError, SplitInfo, load_feature_package, load_preprocess_contract_feature_inputs, load_split_info
from .methods import DEEP_EMBEDDING_METHODS, FeatureMethodError, apply_feature_method, parse_method, requires_train_fit
from .parameter_policy import missing_critical_parameters
from .writer import FeatureWriteError, write_feature_iteration_outputs, write_feature_package


def feature_spectral_package(
    *,
    package_dir: str | Path | None = None,
    preprocess_contract: str | Path | None = None,
    split_contract: str | Path | None = None,
    output_dir: str | Path | None = None,
    method: str | None = None,
    n_components: int | None = None,
    explained_variance: float | None = None,
    variance_threshold: float | None = None,
    band_min: float | None = None,
    band_max: float | None = None,
    band_indices: str | None = None,
    feature_names: str | None = None,
    index_base: int = 0,
    top_k: int | None = None,
    score_threshold: float | None = None,
    n_intervals: int | None = None,
    n_runs: int | None = None,
    sample_ratio: float | None = None,
    cv: int | None = None,
    random_state: int | None = None,
    task_type: str | None = None,
    correlation_method: str | None = None,
    interval_mode: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    noise_std: float | None = None,
    mask_ratio: float | None = None,
    temperature: float | None = None,
    patch_size: int | None = None,
    device: str | None = None,
    feature_config: str | Path | dict[str, Any] | None = None,
    auto_confirm_feature_defaults: bool = False,
    confirm_unsplit_fit: bool = False,
    confirm_deep_embedding_training: bool = False,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    try:
        config = _load_feature_config(feature_config)
        method = method or config.get("method")
        params = dict(config.get("params") or {})
        provided = {
            "n_components": n_components is not None or "n_components" in params,
            "top_k": top_k is not None or "top_k" in params,
            "score_threshold": score_threshold is not None or "score_threshold" in params,
            "n_intervals": n_intervals is not None or "n_intervals" in params,
            "n_runs": n_runs is not None or "n_runs" in params,
            "sample_ratio": sample_ratio is not None or "sample_ratio" in params,
            "cv": cv is not None or "cv" in params,
            "random_state": random_state is not None or "random_state" in params,
            "epochs": epochs is not None or "epochs" in params,
            "batch_size": batch_size is not None or "batch_size" in params,
            "learning_rate": learning_rate is not None or "learning_rate" in params,
            "weight_decay": weight_decay is not None or "weight_decay" in params,
            "noise_std": noise_std is not None or "noise_std" in params,
            "mask_ratio": mask_ratio is not None or "mask_ratio" in params,
            "temperature": temperature is not None or "temperature" in params,
            "patch_size": patch_size is not None or "patch_size" in params,
            "device": device is not None or "device" in params,
        }
        feature_params = {
            "n_components": n_components if n_components is not None else params.get("n_components"),
            "top_k": top_k if top_k is not None else params.get("top_k"),
            "score_threshold": score_threshold if score_threshold is not None else params.get("score_threshold"),
            "n_intervals": n_intervals if n_intervals is not None else params.get("n_intervals"),
            "n_runs": n_runs if n_runs is not None else params.get("n_runs"),
            "sample_ratio": sample_ratio if sample_ratio is not None else params.get("sample_ratio"),
            "cv": cv if cv is not None else params.get("cv"),
            "random_state": random_state if random_state is not None else params.get("random_state"),
            "epochs": epochs if epochs is not None else params.get("epochs"),
            "batch_size": batch_size if batch_size is not None else params.get("batch_size"),
            "learning_rate": learning_rate if learning_rate is not None else params.get("learning_rate"),
            "weight_decay": weight_decay if weight_decay is not None else params.get("weight_decay"),
            "noise_std": noise_std if noise_std is not None else params.get("noise_std"),
            "mask_ratio": mask_ratio if mask_ratio is not None else params.get("mask_ratio"),
            "temperature": temperature if temperature is not None else params.get("temperature"),
            "patch_size": patch_size if patch_size is not None else params.get("patch_size"),
            "device": device if device is not None else params.get("device"),
        }
        task_type = task_type or params.get("task_type")
        correlation_method = correlation_method or params.get("correlation_method")
        interval_mode = interval_mode or params.get("mode")
        selected_method = parse_method(method)
        if selected_method in DEEP_EMBEDDING_METHODS and not confirm_deep_embedding_training:
            return error_response(
                "feature_spectral_package",
                "Deep spectral embedding training requires explicit confirmation of the training protocol and experimental small-sample risk.",
                backend=backend,
                code="DEEP_EMBEDDING_TRAINING_CONFIRMATION_REQUIRED",
                result={
                    "status": "needs_confirmation",
                    "method": selected_method,
                    "confirmation_required": [
                        {
                            "field": "confirm_deep_embedding_training",
                            "reason": "Confirm train-only self-supervised fitting, runtime budget, and comparison against simpler baselines.",
                            "option": "--confirm-deep-embedding-training",
                        }
                    ],
                },
                warnings=warnings,
            )
        package = None
        split_info = None
        if preprocess_contract is not None:
            package, split_info, iteration_inputs = load_preprocess_contract_feature_inputs(preprocess_contract, split_contract=split_contract)
            if iteration_inputs:
                feature_params, provenance = _prepare_feature_parameters(
                    selected_method,
                    feature_params,
                    provided,
                    n_features=package.n_features,
                    n_train=len(iteration_inputs[0]["partition"].train_indices),
                    auto_confirm=auto_confirm_feature_defaults,
                )
                iteration_results = []
                for item in iteration_inputs:
                    partition = item["partition"]
                    partition_package = item["package"]
                    transformed, feature_names_out, band_axis_rows, state = _apply(
                        partition_package, selected_method, partition.train_indices,
                        n_components=feature_params["n_components"], explained_variance=explained_variance,
                        variance_threshold=variance_threshold, band_min=band_min, band_max=band_max,
                        band_indices=band_indices, feature_names=feature_names, index_base=index_base,
                        top_k=feature_params["top_k"], score_threshold=feature_params["score_threshold"], n_intervals=feature_params["n_intervals"],
                        n_runs=feature_params["n_runs"], sample_ratio=feature_params["sample_ratio"], cv=feature_params["cv"], random_state=feature_params["random_state"],
                        task_type=task_type, correlation_method=correlation_method, interval_mode=interval_mode,
                        **_deep_training_kwargs(feature_params),
                    )
                    _annotate_parameter_provenance(state, provenance, auto_confirm_feature_defaults)
                    warnings.extend(state.get("warnings") or [])
                    iteration_results.append(
                        {
                            "partition": partition,
                            "X": transformed,
                            "feature_names": feature_names_out,
                            "band_axis_rows": band_axis_rows,
                            "state": state,
                        }
                    )
                preview = {
                    "status": "ready",
                    "methods": [selected_method],
                    "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
                    "split_type": split_info.split_type,
                    "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
                    "iteration_count": len(iteration_results),
                    "fit_scope": "train_only_per_partition",
                    "transform_scope": "partition_train_val_test",
                    "input_contract": str(preprocess_contract),
                    "warnings": warnings,
                }
                if output_dir is None:
                    preview["handoff_ready"] = False
                    preview["message"] = "No output_dir was provided; returning fold/repeat-wise feature preview."
                    return ok_response("feature_spectral_package", preview, backend=backend, warnings=warnings)
                written = write_feature_iteration_outputs(
                    package,
                    output_dir=output_dir,
                    method=selected_method,
                    iteration_results=iteration_results,
                    split_info=split_info,
                    warnings=warnings,
                    overwrite=overwrite,
                    backend=backend,
                )
                return ok_response("feature_spectral_package", written, backend=backend, warnings=warnings)

        if package is None:
            if package_dir is None:
                return error_response(
                    "feature_spectral_package",
                    "Provide package_dir for standard packages or preprocess_contract for fold/repeat-wise preprocess outputs.",
                    backend=backend,
                    code="FEATURE_INPUT_REQUIRED",
                    result={"status": "needs_confirmation"},
                    warnings=warnings,
                )
            package = load_feature_package(package_dir)
        if split_info is None:
            split_info = load_split_info(split_contract, package) if split_contract is not None else SplitInfo(path=None, contract=None, assignments={"train": list(range(package.n_samples)), "val": [], "test": []})
        if split_contract is None and preprocess_contract is None and requires_train_fit(selected_method) and not confirm_unsplit_fit:
            return error_response(
                "feature_spectral_package",
                "This feature method requires train-only fitted parameters. Provide split_contract.json, a preprocess_contract with split_contract, or confirm an exploratory whole-dataset transform.",
                backend=backend,
                code="SPLIT_CONTRACT_REQUIRED_FOR_FIT",
                result={
                    "status": "needs_confirmation",
                    "method": selected_method,
                    "confirmation_required": [
                        {
                            "field": "split_contract",
                            "reason": "Train-fitted feature methods require train-only fitting.",
                            "options": ["provide split_contract.json", "rerun with --confirm-unsplit-fit"],
                        }
                    ],
                },
                warnings=warnings,
            )
        if split_info.path is None:
            warnings.append(
                {
                    "code": "UNSPLIT_FEATURE_CONFIRMED" if confirm_unsplit_fit else "UNSPLIT_FEATURE",
                    "message": "No split_contract.json was provided; feature fit scope is all samples.",
                    "severity": "warning",
                    "details": {"fit_scope": "all_samples"},
                }
            )

        first_train = split_info.train_indices
        if split_info.split_type != "holdout" and split_info.partitions:
            first_train = split_info.partitions[0].train_indices
        feature_params, provenance = _prepare_feature_parameters(
            selected_method,
            feature_params,
            provided,
            n_features=package.n_features,
            n_train=len(first_train),
            auto_confirm=auto_confirm_feature_defaults,
        )

        if split_info.split_type != "holdout":
            iteration_results = []
            for partition in split_info.partitions or []:
                transformed, feature_names_out, band_axis_rows, state = _apply(
                    package, selected_method, partition.train_indices,
                    n_components=feature_params["n_components"], explained_variance=explained_variance,
                    variance_threshold=variance_threshold, band_min=band_min, band_max=band_max,
                    band_indices=band_indices, feature_names=feature_names, index_base=index_base,
                    top_k=feature_params["top_k"], score_threshold=feature_params["score_threshold"], n_intervals=feature_params["n_intervals"],
                    n_runs=feature_params["n_runs"], sample_ratio=feature_params["sample_ratio"], cv=feature_params["cv"], random_state=feature_params["random_state"],
                    task_type=task_type, correlation_method=correlation_method, interval_mode=interval_mode,
                    **_deep_training_kwargs(feature_params),
                )
                _annotate_parameter_provenance(state, provenance, auto_confirm_feature_defaults)
                warnings.extend(state.get("warnings") or [])
                iteration_results.append(
                    {
                        "partition": partition,
                        "X": transformed,
                        "feature_names": feature_names_out,
                        "band_axis_rows": band_axis_rows,
                        "state": state,
                    }
                )
            preview = {
                "status": "ready",
                "methods": [selected_method],
                "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
                "split_type": split_info.split_type,
                "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
                "iteration_count": len(iteration_results),
                "fit_scope": "train_only_per_partition",
                "transform_scope": "partition_train_val_test",
                "warnings": warnings,
            }
            if output_dir is None:
                preview["handoff_ready"] = False
                preview["message"] = "No output_dir was provided; returning fold/repeat-wise feature preview."
                return ok_response("feature_spectral_package", preview, backend=backend, warnings=warnings)
            written = write_feature_iteration_outputs(
                package,
                output_dir=output_dir,
                method=selected_method,
                iteration_results=iteration_results,
                split_info=split_info,
                warnings=warnings,
                overwrite=overwrite,
                backend=backend,
            )
            return ok_response("feature_spectral_package", written, backend=backend, warnings=warnings)

        transformed, feature_names_out, band_axis_rows, state = _apply(
            package, selected_method, split_info.train_indices,
            n_components=feature_params["n_components"], explained_variance=explained_variance,
            variance_threshold=variance_threshold, band_min=band_min, band_max=band_max,
            band_indices=band_indices, feature_names=feature_names, index_base=index_base,
            top_k=feature_params["top_k"], score_threshold=feature_params["score_threshold"], n_intervals=feature_params["n_intervals"],
            n_runs=feature_params["n_runs"], sample_ratio=feature_params["sample_ratio"], cv=feature_params["cv"], random_state=feature_params["random_state"],
            task_type=task_type, correlation_method=correlation_method, interval_mode=interval_mode,
            **_deep_training_kwargs(feature_params),
        )
        _annotate_parameter_provenance(state, provenance, auto_confirm_feature_defaults)
        warnings.extend(state.get("warnings") or [])
        preview = {
            "status": "ready",
            "methods": [selected_method],
            "shape": {"n_samples": package.n_samples, "n_features": len(feature_names_out)},
            "fit_scope": "train_only" if split_contract is not None else "all_samples_confirmed",
            "transform_scope": "train_val_test" if split_contract is not None else "all_samples",
            "warnings": warnings,
        }
        if output_dir is None:
            preview["handoff_ready"] = False
            preview["message"] = "No output_dir was provided; returning feature preview without writing data_contract.json."
            return ok_response("feature_spectral_package", preview, backend=backend, warnings=warnings)
        written = write_feature_package(
            package,
            output_dir=output_dir,
            X=transformed,
            feature_names=feature_names_out,
            band_axis_rows=band_axis_rows,
            method=selected_method,
            state=state,
            split_info=split_info,
            warnings=warnings,
            overwrite=overwrite,
            backend=backend,
        )
        return ok_response("feature_spectral_package", written, backend=backend, warnings=warnings)
    except FeatureInputError as exc:
        return error_response("feature_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details, warnings=warnings)
    except FeatureMethodError as exc:
        status = "needs_confirmation" if exc.code in {
            "FEATURE_METHOD_REQUIRED",
            "PCA_RETENTION_REQUIRED",
            "BAND_RANGE_REQUIRED",
            "BAND_INDICES_EMPTY",
            "TASK_TYPE_REQUIRED",
            "N_RUNS_CONFIRMATION_REQUIRED",
            "FEATURE_SELECTION_EMPTY",
            "UVE_EMPTY",
            "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED",
            "FEATURE_METHOD_EXPERIMENTAL_GATED",
            "TSNE_VISUALIZATION_ONLY_REQUIRES_UNSPLIT",
        } else "blocked"
        return error_response(
            "feature_spectral_package",
            exc.message,
            backend=backend,
            code=exc.code,
            result=_method_error_result(status, method, exc),
            details=exc.details,
            warnings=warnings,
        )
    except FeatureWriteError as exc:
        return error_response("feature_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details, warnings=warnings)


def _apply(
    package: Any,
    method: str,
    train_indices: list[int],
    **kwargs: Any,
) -> tuple[list[list[float]], list[str], list[list[Any]], dict[str, Any]]:
    return apply_feature_method(package, method=method, train_indices=train_indices, **kwargs)


def _load_feature_config(feature_config: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if feature_config is None:
        return {}
    if isinstance(feature_config, dict):
        return feature_config
    path = Path(feature_config)
    if not path.exists():
        raise FeatureInputError("FEATURE_CONFIG_MISSING", "feature_config JSON file does not exist.", path=str(path))
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        raise FeatureInputError("FEATURE_CONFIG_INVALID", "feature_config must contain a JSON object.", path=str(path))
    return payload


def _method_error_result(status: str, method: str | None, exc: FeatureMethodError) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "method": method}
    if status != "needs_confirmation":
        return result
    if exc.code == "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED":
        result["confirmation_required"] = exc.details.get("confirmation_required") or []
        result["recommended_defaults"] = exc.details.get("recommended_defaults") or {}
        return result
    options: dict[str, list[str]] = {
        "TASK_TYPE_REQUIRED": ["set --task-type classification", "set --task-type regression"],
        "N_RUNS_CONFIRMATION_REQUIRED": ["reduce --n-runs to 500 or less"],
        "FEATURE_SELECTION_EMPTY": ["lower --score-threshold", "provide --top-k"],
        "UVE_EMPTY": ["lower --score-threshold", "provide --top-k"],
        "FEATURE_METHOD_REQUIRED": ["choose one supported feature method"],
        "FEATURE_METHOD_EXPERIMENTAL_GATED": ["provide a dedicated training protocol", "use spectral-report with an existing embedding artifact"],
        "TSNE_VISUALIZATION_ONLY_REQUIRES_UNSPLIT": ["rerun without split_contract for confirmed discovery visualization", "choose a train-transformable embedding such as isomap_embedding or lle_embedding"],
    }
    field_map = {
        "TASK_TYPE_REQUIRED": "task_type",
        "N_RUNS_CONFIRMATION_REQUIRED": "n_runs",
        "FEATURE_SELECTION_EMPTY": "score_threshold",
        "UVE_EMPTY": "score_threshold",
        "FEATURE_METHOD_REQUIRED": "method",
        "FEATURE_METHOD_EXPERIMENTAL_GATED": "method",
        "TSNE_VISUALIZATION_ONLY_REQUIRES_UNSPLIT": "split_contract",
    }
    result["confirmation_required"] = [
        {
            "field": field_map.get(exc.code, "parameters"),
            "reason": exc.message,
            "options": options.get(exc.code, ["adjust parameters and rerun"]),
        }
    ]
    return result


def _prepare_feature_parameters(
    method: str,
    values: dict[str, Any],
    provided: dict[str, bool],
    *,
    n_features: int,
    n_train: int,
    auto_confirm: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    defaults = _recommended_defaults(method, n_features=n_features, n_train=n_train)
    missing = missing_critical_parameters(method, values)
    if missing and not auto_confirm:
        confirmations = []
        for field in missing:
            if field == "selection_rule":
                confirmations.append(
                    {
                        "field": "selection_rule",
                        "reason": f"{method} requires choosing top_k or score_threshold.",
                        "options": [
                            f"top_k={defaults.get('top_k')}",
                            f"score_threshold={defaults.get('score_threshold', 1.0)}",
                        ],
                    }
                )
            else:
                confirmations.append(
                    {
                        "field": field,
                        "reason": f"{field} is a key parameter for {method}.",
                        "recommended": defaults.get(field),
                    }
                )
        raise FeatureMethodError(
            "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED",
            f"Please confirm key parameters before running {method}.",
            confirmation_required=confirmations,
            recommended_defaults=defaults,
        )
    resolved = dict(values)
    sources: dict[str, str] = {
        field: "user_specified"
        for field, is_provided in provided.items()
        if is_provided and resolved.get(field) is not None
    }
    for field, default in defaults.items():
        if method in {"vip", "correlation_filter"} and field == "top_k" and resolved.get("score_threshold") is not None:
            continue
        if resolved.get(field) is None and default is not None:
            resolved[field] = default
            sources[field] = "defaulted_auto_confirmed"
    return resolved, sources


def _recommended_defaults(method: str, *, n_features: int, n_train: int) -> dict[str, Any]:
    components = max(1, min(10, n_train - 1, n_features))
    deep_components = max(2, min(16, n_train - 1, n_features))
    deep_batch = max(1, min(16, n_train))
    deep_patch = max(1, min(16, n_features))
    top_5pct = min(50, max(5, int(math.ceil(0.05 * n_features))))
    spa_top = min(30, max(5, int(math.ceil(0.03 * n_features))))
    cv = max(2, min(5, 3 if n_train < 10 else n_train - 1))
    defaults: dict[str, dict[str, Any]] = {
        "pls_latent_variables": {"n_components": components},
        "kernel_pca": {"n_components": min(2, components)},
        "sparse_pca": {"n_components": min(2, components), "random_state": 42},
        "nmf": {"n_components": min(2, components), "random_state": 42},
        "ica_embedding": {"n_components": min(2, components), "random_state": 42},
        "lda_projection": {"n_components": min(2, components)},
        "dct_features": {"n_components": min(10, n_features)},
        "fft_features": {"n_components": min(10, n_features)},
        "dictionary_learning": {"n_components": min(2, components), "random_state": 42},
        "umap_embedding": {"n_components": 2, "random_state": 42},
        "isomap_embedding": {"n_components": 2},
        "lle_embedding": {"n_components": 2, "random_state": 42},
        "tsne_embedding": {"n_components": 2, "random_state": 42},
        "vip": {"n_components": components, "top_k": min(50, n_features)},
        "correlation_filter": {"top_k": top_5pct},
        "select_k_best": {"top_k": top_5pct},
        "anova_f": {"top_k": top_5pct},
        "f_regression": {"top_k": top_5pct},
        "interval_pls": {"n_intervals": 10 if n_features < 200 else 20, "n_components": components, "cv": cv},
        "spa": {"top_k": spa_top},
        "cars": {"n_components": components, "n_runs": 50, "sample_ratio": 0.8, "cv": cv, "random_state": 42},
        "uve": {"n_components": components, "n_runs": 50, "top_k": min(50, n_features), "sample_ratio": 0.8, "random_state": 42},
        "mcuve": {"n_components": components, "n_runs": 100, "top_k": min(50, n_features), "sample_ratio": 0.8, "random_state": 42},
        # Deep defaults are method-specific and data-aware confirmation-card
        # recommendations, not claims of optimality.
        "autoencoder_embedding": {"n_components": deep_components, "epochs": 100, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-5, "random_state": 42, "device": "cpu"},
        "denoising_autoencoder_embedding": {"n_components": deep_components, "epochs": 100, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-5, "noise_std": 0.03, "random_state": 42, "device": "cpu"},
        "cnn_1d_embedding": {"n_components": deep_components, "epochs": 80, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-4, "random_state": 42, "device": "cpu"},
        "cls_former_embedding": {"n_components": deep_components, "epochs": 80, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-4, "patch_size": deep_patch, "random_state": 42, "device": "cpu"},
        "resnet1d_embedding": {"n_components": deep_components, "epochs": 60, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-4, "random_state": 42, "device": "cpu"},
        "masked_spectral_autoencoder_embedding": {"n_components": deep_components, "epochs": 100, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-4, "mask_ratio": 0.15, "patch_size": deep_patch, "random_state": 42, "device": "cpu"},
        "contrastive_spectral_embedding": {"n_components": deep_components, "epochs": 100, "batch_size": deep_batch, "learning_rate": 0.001, "weight_decay": 1e-4, "noise_std": 0.03, "mask_ratio": 0.1, "temperature": 0.2, "random_state": 42, "device": "cpu"},
    }
    return defaults.get(method, {})


def _deep_training_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "epochs": params.get("epochs"),
        "batch_size": params.get("batch_size"),
        "learning_rate": params.get("learning_rate"),
        "weight_decay": params.get("weight_decay"),
        "noise_std": params.get("noise_std"),
        "mask_ratio": params.get("mask_ratio"),
        "temperature": params.get("temperature"),
        "patch_size": params.get("patch_size"),
        "device": params.get("device"),
    }


def _annotate_parameter_provenance(state: dict[str, Any], sources: dict[str, str], auto_confirm: bool) -> None:
    state["parameter_sources"] = sources
    state["defaulted_params"] = sorted(field for field, source in sources.items() if source == "defaulted_auto_confirmed")
    state["user_specified_params"] = sorted(field for field, source in sources.items() if source == "user_specified")
    state["defaults_confirmed"] = bool(auto_confirm and state["defaulted_params"])

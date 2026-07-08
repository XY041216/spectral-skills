"""Unified leakage-safe spectral preprocessing workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectral_core.reader.response import error_response, ok_response

from .io import PreprocessInputError, SplitInfo, load_preprocess_package, load_split_info
from .methods import (
    PreprocessMethodError,
    apply_preprocess_methods,
    parse_methods,
    requires_absorbance_confirmation,
    requires_band_change_confirmation,
    requires_baseline_confirmation,
    requires_train_fit,
)
from .writer import PreprocessWriteError, write_preprocess_iteration_outputs, write_preprocess_package


def preprocess_spectral_package(
    *,
    package_dir: str | Path,
    split_contract: str | Path | None = None,
    output_dir: str | Path | None = None,
    methods: list[str] | str | None = None,
    window_length: int | None = None,
    polyorder: int | None = None,
    sigma: float | None = None,
    poly_degree: int | None = None,
    als_lambda: float | None = None,
    als_p: float | None = None,
    als_iter: int | None = None,
    band_range: str | None = None,
    remove_band_ranges: str | None = None,
    confirm_baseline: bool = False,
    confirm_absorbance: bool = False,
    confirm_band_change: bool = False,
    confirm_unsplit_fit: bool = False,
    overwrite: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    try:
        requested_methods = parse_methods(methods)
        selected_methods, order_info = _normalize_method_order(requested_methods)
        method_warnings = _method_chain_warnings(selected_methods)
        warnings.extend(method_warnings)
        if order_info["order_normalized"]:
            warnings.append(
                {
                    "code": "PREPROCESS_ORDER_NORMALIZED",
                    "message": order_info["reason"],
                    "severity": "info",
                    "details": {
                        "requested_methods": requested_methods,
                        "executed_methods": selected_methods,
                    },
                }
            )
        if requires_baseline_confirmation(selected_methods) and not confirm_baseline:
            return error_response(
                "preprocess_spectral_package",
                "Baseline correction changes each spectrum's baseline shape. Confirm before applying it.",
                backend=backend,
                code="BASELINE_CONFIRMATION_REQUIRED",
                result={"status": "needs_confirmation", "methods": selected_methods, "requested_methods": requested_methods},
                warnings=warnings,
            )
        if requires_absorbance_confirmation(selected_methods) and not confirm_absorbance:
            return error_response(
                "preprocess_spectral_package",
                "Absorbance conversion requires confirmed positive reflectance/transmittance input values.",
                backend=backend,
                code="ABSORBANCE_CONFIRMATION_REQUIRED",
                result={"status": "needs_confirmation", "methods": selected_methods, "requested_methods": requested_methods},
                warnings=warnings,
            )
        if requires_band_change_confirmation(selected_methods) and not confirm_band_change:
            return error_response(
                "preprocess_spectral_package",
                "Band range preprocessing changes feature count and band_axis. Confirm before applying it.",
                backend=backend,
                code="BAND_CHANGE_CONFIRMATION_REQUIRED",
                result={"status": "needs_confirmation", "methods": selected_methods, "requested_methods": requested_methods},
                warnings=warnings,
            )
        package = load_preprocess_package(package_dir)
        split_info = load_split_info(split_contract, package) if split_contract is not None else SplitInfo(path=None, contract=None, assignments={"train": list(range(package.n_samples)), "val": [], "test": []})
        if split_contract is None and requires_train_fit(selected_methods) and not confirm_unsplit_fit:
            return error_response(
                "preprocess_spectral_package",
                "This preprocessing requires train-only fitted parameters. Provide split_contract.json or confirm an unsupervised whole-dataset transform.",
                backend=backend,
                code="SPLIT_CONTRACT_REQUIRED_FOR_FIT",
                result={"status": "needs_confirmation", "methods": selected_methods, "requested_methods": requested_methods},
                warnings=warnings,
            )
        if split_contract is None:
            warnings.append(
                {
                    "code": "UNSPLIT_PREPROCESS_CONFIRMED" if confirm_unsplit_fit else "UNSPLIT_PREPROCESS",
                    "message": "No split_contract.json was provided; preprocessing fit scope is all samples.",
                    "severity": "warning",
                    "details": {"fit_scope": "all_samples"},
                }
            )

        if split_info.split_type != "holdout":
            iteration_results = []
            for partition in split_info.partitions or []:
                transformed, state = apply_preprocess_methods(
                    package.X,
                    methods=selected_methods,
                    train_indices=partition.train_indices,
                    window_length=window_length,
                    polyorder=polyorder,
                    sigma=sigma,
                    poly_degree=poly_degree,
                    als_lambda=als_lambda,
                    als_p=als_p,
                    als_iter=als_iter,
                    band_range=band_range,
                    remove_band_ranges=remove_band_ranges,
                    feature_names=package.feature_names,
                )
                _attach_method_order(state, requested_methods, selected_methods, order_info)
                iteration_results.append({"partition": partition, "X": transformed, "state": state})
            preview = {
                "status": "ready",
                "methods": selected_methods,
                "requested_methods": requested_methods,
                "executed_methods": selected_methods,
                "order_normalized": order_info["order_normalized"],
                "order_normalization_reason": order_info["reason"],
                "shape": {"n_samples": package.n_samples, "n_features": len(iteration_results[0]["X"][0]) if iteration_results and iteration_results[0]["X"] else package.n_features},
                "split_type": split_info.split_type,
                "execution_mode": "fold_wise" if split_info.split_type == "cross_validation" else "repeat_wise",
                "iteration_count": len(iteration_results),
                "fit_scope": "train_only_per_partition",
                "transform_scope": "partition_train_val_test",
                "warnings": warnings,
            }
            if output_dir is None:
                preview["handoff_ready"] = False
                preview["message"] = "No output_dir was provided; returning fold/repeat-wise preprocess preview."
                return ok_response("preprocess_spectral_package", preview, backend=backend, warnings=warnings)
            written = write_preprocess_iteration_outputs(
                package,
                output_dir=output_dir,
                methods=selected_methods,
                iteration_results=iteration_results,
                split_info=split_info,
                warnings=warnings,
                overwrite=overwrite,
                backend=backend,
                )
            return ok_response("preprocess_spectral_package", written, backend=backend, warnings=warnings)

        transformed, state = apply_preprocess_methods(
            package.X,
            methods=selected_methods,
            train_indices=split_info.train_indices,
            window_length=window_length,
            polyorder=polyorder,
            sigma=sigma,
            poly_degree=poly_degree,
            als_lambda=als_lambda,
            als_p=als_p,
            als_iter=als_iter,
            band_range=band_range,
            remove_band_ranges=remove_band_ranges,
            feature_names=package.feature_names,
        )
        _attach_method_order(state, requested_methods, selected_methods, order_info)
        preview = {
            "status": "ready",
            "methods": selected_methods,
            "requested_methods": requested_methods,
            "executed_methods": selected_methods,
            "order_normalized": order_info["order_normalized"],
            "order_normalization_reason": order_info["reason"],
            "shape": {"n_samples": package.n_samples, "n_features": len(transformed[0]) if transformed else package.n_features},
            "fit_scope": "train_only" if split_contract is not None else "all_samples_confirmed",
            "transform_scope": "train_val_test" if split_contract is not None else "all_samples",
            "warnings": warnings,
        }
        if output_dir is None:
            preview["handoff_ready"] = False
            preview["message"] = "No output_dir was provided; returning preprocess preview without writing data_contract.json."
            return ok_response("preprocess_spectral_package", preview, backend=backend, warnings=warnings)
        written = write_preprocess_package(
            package,
            output_dir=output_dir,
            X=transformed,
            methods=selected_methods,
            state=state,
            split_info=split_info,
            warnings=warnings,
            overwrite=overwrite,
            backend=backend,
        )
        return ok_response("preprocess_spectral_package", written, backend=backend, warnings=warnings)
    except PreprocessInputError as exc:
        return error_response("preprocess_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details, warnings=warnings)
    except PreprocessMethodError as exc:
        status = "needs_confirmation" if exc.code in {"PREPROCESS_METHOD_REQUIRED", "SG_PARAMETERS_REQUIRED", "BAND_RANGE_REQUIRED", "REMOVE_BAND_RANGES_REQUIRED"} else "blocked"
        return error_response("preprocess_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": status}, details=exc.details, warnings=warnings)
    except PreprocessWriteError as exc:
        return error_response("preprocess_spectral_package", exc.message, backend=backend, code=exc.code, result={"status": "blocked"}, details=exc.details, warnings=warnings)


def _normalize_method_order(methods: list[str]) -> tuple[list[str], dict[str, Any]]:
    band_methods = {"band_range_select", "remove_band_ranges"}
    if "none" in methods:
        return list(methods), {"order_normalized": False, "reason": None}
    requested = list(methods)
    normalized = [method for method in requested if method in band_methods] + [method for method in requested if method not in band_methods]
    normalized_changed = normalized != requested
    reason = None
    if normalized_changed:
        reason = (
            "band_range_select/remove_band_ranges are executed before other preprocessing steps so excluded bands "
            "do not influence downstream per-spectrum statistics such as SNV or MSC reference/scaling behavior."
        )
    return normalized, {"order_normalized": normalized_changed, "reason": reason}


def _attach_method_order(state: dict[str, Any], requested_methods: list[str], executed_methods: list[str], order_info: dict[str, Any]) -> None:
    state["requested_methods"] = list(requested_methods)
    state["executed_methods"] = list(executed_methods)
    state["order_normalized"] = bool(order_info.get("order_normalized"))
    state["order_normalization_reason"] = order_info.get("reason")


def _method_chain_warnings(methods: list[str]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if "snv" in methods and "msc" in methods:
        warnings.append(
            {
                "code": "OVERLAPPING_SCATTER_CORRECTIONS",
                "message": "SNV and MSC are both scatter corrections; using both is usually unnecessary unless explicitly justified.",
                "severity": "warning",
                "details": {"methods": ["snv", "msc"]},
            }
        )
    if len([method for method in methods if method != "none"]) > 5:
        warnings.append(
            {
                "code": "LONG_PREPROCESS_CHAIN",
                "message": "The preprocessing chain is long and may over-process spectra. Consider validating a simpler chain first.",
                "severity": "warning",
                "details": {"method_count": len(methods), "methods": methods},
            }
        )
    return warnings

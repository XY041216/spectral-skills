"""Execute prepared optimizer trials through official spectral-modeling."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from spectral_core.feature.workflow import feature_spectral_package
from spectral_core.modeling.workflow import model_spectral_package
from spectral_core.preprocess.workflow import preprocess_spectral_package


class OptimizerExecutionError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def execute_validation_trials(
    trials: list[dict[str, Any]],
    *,
    trial_inputs: str | Path | dict[str, Any] | None = None,
    package_dir: str | Path | None = None,
    split_contract: str | Path | None = None,
    fixed_feature_contract: str | Path | None = None,
    output_dir: str | Path,
    task_type: str,
    random_seed: int,
) -> Path:
    inputs = _load_trial_inputs(trial_inputs) if trial_inputs is not None else {}
    root = Path(output_dir)
    trial_root = root / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        trial_id = trial["trial_id"]
        source = inputs.get(trial_id) or _prepare_trial_input(
            trial,
            trial_dir=trial_root / trial_id,
            package_dir=package_dir,
            split_contract=split_contract,
            fixed_feature_contract=fixed_feature_contract,
            task_type=task_type,
            random_seed=random_seed,
        )
        if not isinstance(source, dict):
            raise OptimizerExecutionError(
                "OPTIMIZER_TRIAL_INPUT_MISSING",
                "Every planned trial requires a prepared input contract or package_dir plus split_contract so optimizer can prepare one.",
                trial_id=trial_id,
            )
        started = time.perf_counter()
        trial_output = trial_root / trial_id / "model_output"
        response = model_spectral_package(
            package_dir=source.get("package_dir"),
            split_contract=source.get("split_contract"),
            feature_contract=source.get("feature_contract"),
            preprocess_contract=source.get("preprocess_contract"),
            output_dir=trial_output,
            task_type=task_type,
            models=trial["model_method"],
            model_config={trial["model_method"]: trial.get("model_params") or {}},
            random_seed=random_seed,
            evaluation_mode="validation_only",
            disable_model_selection=True,
            save_model=False,
            overwrite=True,
            backend="optimizer",
        )
        elapsed = time.perf_counter() - started
        if not response.get("ok"):
            rows.append(_failed_row(trial, response, elapsed))
            continue
        metrics = json.loads((trial_output / "metrics.json").read_text(encoding="utf-8"))
        rows.append(_success_row(trial, metrics, elapsed, trial_output))
    path = root / "trial_results.csv"
    _write_rows(path, rows)
    return path


def _load_trial_inputs(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {str(item["trial_id"]): item for item in payload}
    if isinstance(payload, dict) and isinstance(payload.get("trials"), list):
        return {str(item["trial_id"]): item for item in payload["trials"]}
    if isinstance(payload, dict):
        return payload
    raise OptimizerExecutionError("OPTIMIZER_TRIAL_INPUTS_INVALID", "trial_inputs must be a mapping or a list of trial input records.")


def _prepare_trial_input(
    trial: dict[str, Any],
    *,
    trial_dir: Path,
    package_dir: str | Path | None,
    split_contract: str | Path | None,
    fixed_feature_contract: str | Path | None,
    task_type: str,
    random_seed: int,
) -> dict[str, Any]:
    if fixed_feature_contract is not None:
        contract_path = Path(fixed_feature_contract).resolve()
        source: dict[str, Any] = {"feature_contract": str(contract_path)}
        if package_dir is not None:
            source["package_dir"] = str(Path(package_dir).resolve())
        if split_contract is not None:
            source["split_contract"] = str(Path(split_contract).resolve())
        return source
    if package_dir is None or split_contract is None:
        raise OptimizerExecutionError(
            "OPTIMIZER_TRIAL_INPUT_MISSING",
            "Automatic trial preparation requires package_dir and split_contract.",
            trial_id=trial.get("trial_id"),
        )
    trial_dir.mkdir(parents=True, exist_ok=True)
    current_package = Path(package_dir)
    source: dict[str, Any] = {"package_dir": str(current_package.resolve()), "split_contract": str(Path(split_contract).resolve())}

    preprocess_method = str(trial.get("preprocess_method") or "none").strip()
    if preprocess_method and preprocess_method.lower() != "none":
        preprocess_dir = trial_dir / "preprocess_output"
        response = preprocess_spectral_package(
            package_dir=current_package,
            split_contract=split_contract,
            output_dir=preprocess_dir,
            methods=preprocess_method,
            overwrite=True,
            backend="optimizer",
            **_preprocess_kwargs(trial.get("preprocess_params") or {}),
        )
        if not response.get("ok"):
            raise OptimizerExecutionError(
                "OPTIMIZER_PREPROCESS_TRIAL_FAILED",
                "Failed to prepare preprocess output for optimizer trial.",
                trial_id=trial.get("trial_id"),
                response=response,
            )
        current_package = preprocess_dir
        source = {"package_dir": str(current_package.resolve()), "split_contract": str(Path(split_contract).resolve()), "preprocess_contract": str((preprocess_dir / "preprocess_contract.json").resolve())}

    feature_method = str(trial.get("feature_method") or "none").strip()
    if feature_method and feature_method.lower() != "none":
        feature_dir = trial_dir / "feature_output"
        feature_kwargs = _feature_kwargs(trial.get("feature_params") or {})
        feature_kwargs.setdefault("random_state", random_seed)
        if feature_method in {
            "autoencoder_embedding",
            "denoising_autoencoder_embedding",
            "cnn_1d_embedding",
            "resnet1d_embedding",
            "cls_former_embedding",
            "masked_spectral_autoencoder_embedding",
            "contrastive_spectral_embedding",
        }:
            feature_kwargs["confirm_deep_embedding_training"] = True
        response = feature_spectral_package(
            package_dir=current_package,
            split_contract=split_contract,
            output_dir=feature_dir,
            method=feature_method,
            task_type=task_type,
            auto_confirm_feature_defaults=True,
            overwrite=True,
            backend="optimizer",
            **feature_kwargs,
        )
        if not response.get("ok"):
            raise OptimizerExecutionError(
                "OPTIMIZER_FEATURE_TRIAL_FAILED",
                "Failed to prepare feature output for optimizer trial.",
                trial_id=trial.get("trial_id"),
                response=response,
            )
        source = {"package_dir": str(feature_dir.resolve()), "split_contract": str(Path(split_contract).resolve())}
        feature_contract = feature_dir / "feature_contract.json"
        if feature_contract.exists():
            source["feature_contract"] = str(feature_contract.resolve())
    return source


def _preprocess_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "window_length",
        "polyorder",
        "sigma",
        "poly_degree",
        "als_lambda",
        "als_p",
        "als_iter",
        "band_range",
        "remove_band_ranges",
        "confirm_baseline",
        "confirm_absorbance",
        "confirm_band_change",
        "confirm_unsplit_fit",
    }
    return {key: value for key, value in params.items() if key in allowed}


def _feature_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "n_components",
        "explained_variance",
        "variance_threshold",
        "band_min",
        "band_max",
        "band_indices",
        "feature_names",
        "index_base",
        "top_k",
        "score_threshold",
        "n_intervals",
        "n_runs",
        "sample_ratio",
        "cv",
        "random_state",
        "correlation_method",
        "interval_mode",
        "confirm_unsplit_fit",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "noise_std",
        "mask_ratio",
        "temperature",
        "patch_size",
        "device",
        "confirm_deep_embedding_training",
    }
    return {key: value for key, value in params.items() if key in allowed}


def _success_row(trial: dict[str, Any], metrics: dict[str, Any], elapsed: float, output_dir: Path) -> dict[str, Any]:
    val = metrics.get("val_metrics") or {}
    train = metrics.get("train_metrics") or {}
    return {
        "trial_id": trial["trial_id"],
        "preprocess_method": trial["preprocess_method"],
        "feature_method": trial["feature_method"],
        "model_method": trial["model_method"],
        "params": _params_json(trial),
        "val_accuracy": val.get("accuracy", ""),
        "val_macro_f1": val.get("f1", ""),
        "val_rmse": val.get("rmse", ""),
        "val_mae": val.get("mae", ""),
        "train_accuracy": train.get("accuracy", ""),
        "train_rmse": train.get("rmse", ""),
        "fit_time": round(elapsed, 6),
        "test_used_for_selection": False,
        "test_accessed": bool(metrics.get("test_accessed")),
        "official_modeling_used": True,
        "modeling_output": str(output_dir.resolve()),
        "status": "completed",
        "warnings": "",
    }


def _failed_row(trial: dict[str, Any], response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "trial_id": trial["trial_id"],
        "preprocess_method": trial["preprocess_method"],
        "feature_method": trial["feature_method"],
        "model_method": trial["model_method"],
        "params": _params_json(trial),
        "val_accuracy": "",
        "val_macro_f1": "",
        "val_rmse": "",
        "val_mae": "",
        "train_accuracy": "",
        "train_rmse": "",
        "fit_time": round(elapsed, 6),
        "test_used_for_selection": False,
        "test_accessed": False,
        "official_modeling_used": True,
        "status": "failed",
        "warnings": json.dumps(response.get("errors") or [], ensure_ascii=False),
    }


def _params_json(trial: dict[str, Any]) -> str:
    return json.dumps(
        {
            "preprocess": trial.get("preprocess_params") or {},
            "feature": trial.get("feature_params") or {},
            "modeling": trial.get("model_params") or {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

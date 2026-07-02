"""Build split contract payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION

from .io import SpectralSplitPackage


def build_split_contract(
    package: SpectralSplitPackage,
    *,
    output_dir: str | Path,
    method: str,
    split_type: str,
    random_seed: int,
    ratios: dict[str, float],
    assignments: dict[str, list[int]] | None,
    folds: list[dict[str, Any]] | None,
    repeats: list[dict[str, Any]] | None,
    label_distribution: dict[str, Any],
    diagnostics: dict[str, Any],
    warnings: list[dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    counts = {name: len((assignments or {}).get(name, [])) for name in ["train", "val", "test"]}
    contract = {
        "contract_type": "split_contract",
        "schema_version": SCHEMA_VERSION,
        "stage": "spectral-splitter",
        "split_type": split_type,
        "contract_id": f"split-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "input_data_contract_id": str(package.contract.get("contract_id") or package.contract_path),
        "input_contract": str(package.contract_path),
        "contract_status": "confirmed",
        "method": method,
        "random_seed": random_seed,
        "confirmation": {
            "required": True,
            "status": "confirmed",
            "decision_source": "user_specified",
            "question": "Confirm split method, ratio, and random seed before creating train/validation/test assignments.",
            "user_selected_option": {
                "method": method,
                "ratios": ratios,
                "random_seed": random_seed,
            },
            "alternatives": ["random", "stratified", "custom_ratio"],
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        },
        "ratios": {
            "train": ratios["train"],
            "val": ratios["val"],
            "test": ratios["test"],
        },
        "n_samples": {
            "total": package.n_samples,
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
        },
        "split_files": {
            "split_indices": "split_indices.csv",
        },
        "diagnostics": diagnostics,
        "label_distribution": label_distribution,
        "warnings": warnings,
        "handoff_ready": True,
        "next_recommended_skill": "spectral-preprocess",
        "next_step_recommendation": "spectral-preprocess",
        "execution": {
            "backend": backend,
            "server_name": None,
            "tool_chain": ["split_spectral_package"],
            "fallback_used": False,
            "fallback_reason": None,
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_executable": None,
            "working_directory": str(Path.cwd()),
            "plugin_root": None,
            "warnings": warnings,
            "errors": [],
        },
    }
    if assignments is not None:
        contract["indices"] = {name: assignments.get(name, []) for name in ["train", "val", "test"]}
        contract["sample_ids"] = {
            name: [package.sample_ids[idx] for idx in assignments.get(name, [])]
            for name in ["train", "val", "test"]
        }
        contract["splits"] = {
            name: [
                {"sample_id": package.sample_ids[idx], "index": idx}
                for idx in assignments.get(name, [])
            ]
            for name in ["train", "val", "test"]
        }
    if folds is not None:
        contract["n_splits"] = len(folds)
        contract["folds"] = [
            {
                "fold_id": fold["fold_id"],
                "train_indices": fold["train_indices"],
                "val_indices": fold["val_indices"],
                "train_sample_ids": [package.sample_ids[idx] for idx in fold["train_indices"]],
                "val_sample_ids": [package.sample_ids[idx] for idx in fold["val_indices"]],
            }
            for fold in folds
        ]
    if repeats is not None:
        contract["n_repeats"] = len(repeats)
        contract["repeats"] = [
            {
                "repeat_id": repeat["repeat_id"],
                "train_indices": repeat["train_indices"],
                "val_indices": repeat.get("val_indices", []),
                "test_indices": repeat["test_indices"],
                "train_sample_ids": [package.sample_ids[idx] for idx in repeat["train_indices"]],
                "val_sample_ids": [package.sample_ids[idx] for idx in repeat.get("val_indices", [])],
                "test_sample_ids": [package.sample_ids[idx] for idx in repeat["test_indices"]],
            }
            for repeat in repeats
        ]
    return contract

"""Confirmed data-changing QC actions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .io import SpectralQCPackage, numeric_matrix


class QCActionError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def apply_confirmed_actions(
    package: SpectralQCPackage,
    *,
    remove_sample_ids: list[str] | None = None,
    remove_sample_indices: list[int] | None = None,
    remove_band_indices: list[int] | None = None,
    impute_missing: str | None = None,
    cleaning_action: str | None = None,
    cleaning_method: str | None = None,
    cleaning_strategy: str | None = None,
    threshold: float | None = None,
    confirmed: bool = False,
) -> tuple[SpectralQCPackage, dict[str, Any]]:
    if not confirmed:
        raise QCActionError("ACTION_CONFIRMATION_REQUIRED", "Data-changing QC actions require explicit user confirmation.")
    sample_indices = set(remove_sample_indices or [])
    if remove_sample_ids:
        id_to_index = {sample_id: idx for idx, sample_id in enumerate(package.sample_ids)}
        missing_ids = [sample_id for sample_id in remove_sample_ids if sample_id not in id_to_index]
        if missing_ids:
            raise QCActionError("REMOVE_SAMPLE_ID_NOT_FOUND", "Some requested sample IDs were not found.", sample_ids=missing_ids)
        sample_indices.update(id_to_index[sample_id] for sample_id in remove_sample_ids)
    band_indices = set(remove_band_indices or [])
    _validate_indices(sample_indices, package.n_samples, "sample")
    _validate_indices(band_indices, package.n_features, "band")

    matrix = package.X
    if impute_missing and impute_missing != "none":
        matrix = numeric_matrix(package, fill=impute_missing)

    keep_samples = [idx for idx in range(package.n_samples) if idx not in sample_indices]
    keep_bands = [idx for idx in range(package.n_features) if idx not in band_indices]
    if not keep_samples:
        raise QCActionError("ALL_SAMPLES_REMOVED", "QC action would remove all samples.")
    if not keep_bands:
        raise QCActionError("ALL_BANDS_REMOVED", "QC action would remove all bands.")

    new_X = [[row[col_idx] for col_idx in keep_bands] for row_idx, row in enumerate(matrix) if row_idx in keep_samples]
    new_package = replace(
        package,
        X=new_X,
        feature_names=[package.feature_names[idx] for idx in keep_bands],
        sample_ids=[package.sample_ids[idx] for idx in keep_samples],
        band_axis=[package.band_axis[idx] for idx in keep_bands],
        y=[package.y[idx] for idx in keep_samples] if package.y is not None else None,
        metadata=[package.metadata[idx] for idx in keep_samples] if package.metadata is not None else None,
    )
    summary = {
        "applied": True,
        "action": cleaning_action or _infer_action(sample_indices, band_indices, impute_missing),
        "method": cleaning_method or "user_specified_indices",
        "strategy": cleaning_strategy or "explicit_selection",
        "threshold": threshold,
        "removed_sample_count": len(sample_indices),
        "removed_sample_ids": [package.sample_ids[idx] for idx in sorted(sample_indices)],
        "removed_samples": [
            {
                "sample_id": package.sample_ids[idx],
                "sample_index": idx,
                "reason": cleaning_action or "confirmed_qc_cleaning",
            }
            for idx in sorted(sample_indices)
        ],
        "removed_band_count": len(band_indices),
        "removed_band_indices": sorted(band_indices),
        "removed_bands": [
            {
                "band_index": idx,
                "band_label": package.feature_names[idx],
                "band_axis": package.band_axis[idx],
                "reason": cleaning_action or "confirmed_qc_cleaning",
            }
            for idx in sorted(band_indices)
        ],
        "impute_missing": impute_missing or "none",
    }
    return new_package, summary


def _validate_indices(indices: set[int], length: int, role: str) -> None:
    invalid = sorted(idx for idx in indices if idx < 0 or idx >= length)
    if invalid:
        raise QCActionError(f"REMOVE_{role.upper()}_INDEX_INVALID", f"Requested {role} indices are out of range.", indices=invalid, length=length)


def _infer_action(sample_indices: set[int], band_indices: set[int], impute_missing: str | None) -> str:
    actions = []
    if sample_indices:
        actions.append("remove_samples")
    if band_indices:
        actions.append("remove_bands")
    if impute_missing and impute_missing != "none":
        actions.append("impute_missing")
    return "+".join(actions) if actions else "confirmed_qc_cleaning"

"""Build compact contracts for QC outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .io import SpectralQCPackage


def build_qc_contract(
    package: SpectralQCPackage,
    *,
    mode: str,
    result: dict[str, Any],
    methods: list[str] | None = None,
    backend: str = "core",
) -> dict[str, Any]:
    """Build the formal observation-stage QC handoff contract."""
    now = datetime.now(timezone.utc).isoformat()
    parent = dict(package.contract)
    input_contract_path = str(package.root / "data_contract.json")
    outlier_candidates = {}
    if mode == "outliers":
        outlier_candidates = {
            str(method.get("method_id")): {
                "outlier_sample_count": int(method.get("outlier_sample_count") or 0),
                "outlier_sample_candidates": list(method.get("outlier_sample_candidates") or []),
                "confirmation_required_for_removal": bool(method.get("confirmation_required_for_removal")),
            }
            for method in result.get("methods", [])
        }
    return {
        "contract_type": "qc_contract",
        "contract_id": f"qc-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "input_data_contract": input_contract_path,
        "input_data_contract_id": str(parent.get("contract_id") or input_contract_path),
        "package_dir": str(package.root),
        "mode": mode,
        "methods": list(methods or []),
        "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
        "check_results": result if mode == "check" else {},
        "outlier_candidates": outlier_candidates,
        "recommended_actions": list(result.get("recommended_actions") or []),
        "user_confirmed_actions": [],
        "contract_status": "confirmed",
        "confirmation": {
            "required": False,
            "status": "not_required",
            "decision_source": "observation_only",
            "question": None,
            "recommended_option": "Mark and report QC findings; do not change samples, bands, or values.",
            "user_selected_option": "observation_only",
            "alternatives": ["mark_only", "confirmed_cleaning"],
            "confirmed_at": None,
        },
        "outputs": {
            "qc_contract": "qc_contract.json",
            "qc_findings": "qc_findings.json",
        },
        "execution": {
            "backend": backend,
            "tool_chain": ["qc_spectral_package"],
            "timestamp": now,
            "warnings": [],
            "errors": [],
        },
    }


def build_qc_data_contract(
    package: SpectralQCPackage,
    *,
    parent_contract: str,
    qc_summary: dict[str, Any],
) -> dict[str, Any]:
    parent = dict(package.contract)
    contract = dict(parent)
    contract["status"] = "ready"
    contract["processing_stage"] = "qc"
    contract["parent_contract"] = parent_contract
    contract["files"] = {
        "X": "X.csv",
        "sample_ids": "sample_ids.csv",
        "band_axis": "band_axis.csv",
        "y": "y.csv" if package.y is not None else None,
        "metadata": "metadata.csv" if package.metadata is not None else None,
    }
    contract["shape"] = {"n_samples": package.n_samples, "n_features": package.n_features}
    contract["n_samples"] = package.n_samples
    contract["n_features"] = package.n_features
    band_axis = dict(contract.get("band_axis") or {})
    band_axis["file"] = "band_axis.csv"
    band_axis["count"] = package.n_features
    contract["band_axis"] = band_axis
    contract["label_status"] = "present" if package.y is not None else "absent"
    contract["metadata_status"] = "present" if package.metadata is not None else "absent"
    contract["qc_summary"] = {
        "applied": bool(qc_summary.get("applied")),
        "methods_used": list(qc_summary.get("methods_used") or []),
        "removed_sample_count": int(qc_summary.get("removed_sample_count") or 0),
        "removed_samples": list(qc_summary.get("removed_samples") or []),
        "removed_band_count": int(qc_summary.get("removed_band_count") or 0),
        "removed_bands": list(qc_summary.get("removed_bands") or []),
        "impute_missing": qc_summary.get("impute_missing") or "none",
        "handoff_ready": True,
    }
    contract["confirmation"] = {
        "required": bool(qc_summary.get("applied")),
        "status": "confirmed" if qc_summary.get("applied") else "not_required",
        "decision_source": "user_confirmed" if qc_summary.get("applied") else "observation_only",
        "question": "Confirm data-changing QC actions before writing a cleaned package.",
        "user_selected_option": {
            "methods_used": list(qc_summary.get("methods_used") or []),
            "removed_sample_count": int(qc_summary.get("removed_sample_count") or 0),
            "removed_band_count": int(qc_summary.get("removed_band_count") or 0),
            "impute_missing": qc_summary.get("impute_missing") or "none",
        },
        "alternatives": ["mark_only", "remove_samples", "remove_bands", "impute_missing"],
        "confirmed_at": datetime.now(timezone.utc).isoformat() if qc_summary.get("applied") else None,
    }
    execution = dict(contract.get("execution") or {})
    execution["qc_written_at"] = datetime.now(timezone.utc).isoformat()
    execution["backend"] = execution.get("backend") or "spectral_core.qc"
    contract["execution"] = execution
    contract.pop("confidence_scores", None)
    return contract

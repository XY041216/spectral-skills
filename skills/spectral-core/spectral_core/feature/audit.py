"""Post-feature package contract audit and repair helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file, write_json_file


def audit_feature_package(package_dir: str | Path, *, repair: bool = False) -> dict[str, Any]:
    root = Path(package_dir)
    counts = _feature_counts(root)
    contract_path = root / "data_contract.json"
    state_path = root / "feature_state.json"
    contract = load_json_file(contract_path)
    state = load_json_file(state_path)
    issues = _audit_counts(contract, state, counts)
    repaired = False
    if issues and repair and _repairable(issues):
        contract = _repair_contract(contract, state, counts)
        write_json_file(contract_path, contract, ensure_ascii=False)
        repaired = True
        issues = _audit_counts(contract, state, counts)
    return {
        "ok": not issues,
        "status": "ready" if not issues else "blocked",
        "issues": issues,
        "repaired": repaired,
        "counts": counts,
    }


def _feature_counts(root: Path) -> dict[str, Any]:
    x_path = root / "X.csv"
    band_path = root / "band_axis.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing feature X.csv: {x_path}")
    if not band_path.exists():
        raise FileNotFoundError(f"Missing feature band_axis.csv: {band_path}")
    with x_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        x_rows = sum(1 for _ in reader)
    with band_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        band_header = next(reader, [])
        band_rows = list(reader)
    labels = [str(row[1]) if len(row) > 1 else str(row[0]) for row in band_rows]
    return {
        "x_n_samples": x_rows,
        "x_n_features": len(header),
        "x_feature_names": header,
        "band_axis_rows": len(band_rows),
        "band_axis_header": band_header,
        "band_axis_labels": labels,
    }


def _audit_counts(contract: dict[str, Any], state: dict[str, Any], counts: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected_features = counts["x_n_features"]
    expected_samples = counts["x_n_samples"]
    _compare(issues, "shape.n_samples", (contract.get("shape") or {}).get("n_samples"), expected_samples)
    _compare(issues, "shape.n_features", (contract.get("shape") or {}).get("n_features"), expected_features)
    _compare(issues, "n_samples", contract.get("n_samples"), expected_samples)
    _compare(issues, "n_features", contract.get("n_features"), expected_features)
    _compare(issues, "band_axis.count", (contract.get("band_axis") or {}).get("count"), expected_features)
    _compare(issues, "spectral.n_bands", (contract.get("spectral") or {}).get("n_bands"), expected_features)
    _compare(issues, "feature.output_n_features", (contract.get("feature") or {}).get("output_n_features"), expected_features)
    _compare(issues, "feature_summary.output_n_features", (contract.get("feature_summary") or {}).get("output_n_features"), expected_features)
    _compare(issues, "feature_state.output_n_features", state.get("output_n_features"), expected_features)
    if counts["band_axis_rows"] != expected_features:
        issues.append(
            {
                "code": "FEATURE_AXIS_COUNT_MISMATCH",
                "field": "band_axis.csv",
                "expected": expected_features,
                "observed": counts["band_axis_rows"],
                "repairable": False,
            }
        )
    return issues


def _compare(issues: list[dict[str, Any]], field: str, observed: Any, expected: int) -> None:
    if observed is None:
        return
    if observed != expected:
        issues.append(
            {
                "code": "FEATURE_CONTRACT_COUNT_MISMATCH",
                "field": field,
                "expected": expected,
                "observed": observed,
                "repairable": True,
            }
        )


def _repairable(issues: list[dict[str, Any]]) -> bool:
    return all(issue.get("repairable") for issue in issues)


def _repair_contract(contract: dict[str, Any], state: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(contract)
    n_samples = counts["x_n_samples"]
    n_features = counts["x_n_features"]
    method = _feature_method(state, repaired)
    original_features = _original_feature_count(repaired, state)
    repaired["shape"] = {"n_samples": n_samples, "n_features": n_features}
    repaired["n_samples"] = n_samples
    repaired["n_features"] = n_features
    band_axis = dict(repaired.get("band_axis") or {})
    band_axis.setdefault("file", (repaired.get("files") or {}).get("band_axis") or "band_axis.csv")
    band_axis["count"] = n_features
    if method == "pca":
        band_axis["type"] = "derived_feature_axis"
        band_axis["unit"] = "principal_component"
    repaired["band_axis"] = band_axis
    repaired["spectral"] = {
        "n_bands": n_features,
        "band_axis_type": band_axis.get("type") or ("derived_feature_axis" if method == "pca" else "selected_spectral_axis"),
        "band_axis_ref": band_axis.get("file") or "band_axis.csv",
        "band_axis_labels": counts["band_axis_labels"],
    }
    feature = dict(repaired.get("feature") or {})
    feature.update(
        {
            "method": method,
            "input_n_features": original_features,
            "output_n_features": n_features,
            "parameters": _feature_parameters(state),
        }
    )
    repaired["feature"] = feature
    feature_summary = dict(repaired.get("feature_summary") or {})
    feature_summary["input_n_features"] = original_features
    feature_summary["output_n_features"] = n_features
    repaired["feature_summary"] = feature_summary
    repaired.setdefault(
        "source_spectral",
        {
            "original_n_bands": original_features,
            "original_band_axis_ref": _source_band_axis_ref(contract),
        },
    )
    return repaired


def _feature_method(state: dict[str, Any], contract: dict[str, Any]) -> str:
    methods = state.get("methods") or (contract.get("feature_summary") or {}).get("methods") or []
    if methods:
        return str(methods[0])
    method_states = state.get("method_states") or []
    if method_states:
        return str(method_states[0].get("method") or "unknown")
    return str((contract.get("feature") or {}).get("method") or "unknown")


def _feature_parameters(state: dict[str, Any]) -> dict[str, Any]:
    method_states = state.get("method_states") or []
    if method_states:
        return dict(method_states[0].get("parameters") or {})
    return {}


def _original_feature_count(contract: dict[str, Any], state: dict[str, Any]) -> int | None:
    values = [
        (contract.get("source_spectral") or {}).get("original_n_bands"),
        (contract.get("feature") or {}).get("input_n_features"),
        state.get("input_n_features"),
        (contract.get("shape") or {}).get("n_features"),
        contract.get("n_features"),
        (contract.get("band_axis") or {}).get("count"),
    ]
    numeric = [int(value) for value in values if value is not None]
    return max(numeric) if numeric else None


def _source_band_axis_ref(contract: dict[str, Any]) -> str | None:
    source = contract.get("source_spectral") or {}
    if source.get("original_band_axis_ref"):
        return source.get("original_band_axis_ref")
    band_axis = contract.get("band_axis") or {}
    return band_axis.get("file") or contract.get("band_axis_ref") or (contract.get("files") or {}).get("band_axis")

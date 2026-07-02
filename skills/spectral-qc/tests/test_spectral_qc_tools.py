from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from spectral_core.qc.workflow import qc_spectral_package


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_package(root: Path, *, with_y: bool = True, with_metadata: bool = True) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200"],
            [0.10, 0.11, 5.00, 0.13],
            [0.11, 0.12, 5.00, ""],
            [0.12, 0.13, 5.00, 0.15],
            [0.13, 0.14, 5.00, 0.16],
            [4.50, 4.70, 5.00, 4.90],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"], [3, 1200, "cm-1"]])
    files = {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": None, "metadata": None}
    if with_y:
        _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["B"], ["B"], ["B"]])
        files["y"] = "y.csv"
    if with_metadata:
        _write_rows(root / "metadata.csv", [["batch"], ["B1"], ["B1"], ["B2"], ["B2"], ["B3"]])
        files["metadata"] = "metadata.csv"
    contract = {
        "status": "ready",
        "reader_version": "test",
        "source_type": "synthetic",
        "files": files,
        "shape": {"n_samples": 5, "n_features": 4},
        "sample_orientation": "rows",
        "label_status": "present" if with_y else "absent",
        "metadata_status": "present" if with_metadata else "absent",
        "task_hint": "classification" if with_y else "unsupervised",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 4},
        "warnings": [],
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_supervised_package_without_y(root: Path) -> Path:
    package = _write_package(root, with_y=False)
    contract_path = package / "data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["task_hint"] = "classification"
    contract["label_status"] = "present"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return package


def _write_duplicate_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100"],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["B"], ["B"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 4, "n_features": 3},
        "n_samples": 4,
        "n_features": 3,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 3},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_exact_label_conflict_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200"],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [2.0, 1.0, 0.5, 0.2],
            [2.1, 1.1, 0.6, 0.3],
            [2.2, 1.2, 0.7, 0.4],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"], [3, 1200, "cm-1"]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["B"], ["A"], ["B"], ["B"], ["A"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 6, "n_features": 4},
        "n_samples": 6,
        "n_features": 4,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 4},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_high_similarity_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200"],
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.2, 3.3, 4.4],
            [0.9, 1.8, 2.7, 3.6],
            [2.0, 4.0, 6.0, 8.0],
            [2.1, 4.2, 6.3, 8.4],
            [1.9, 3.8, 5.7, 7.6],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"], [3, 1200, "cm-1"]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 6, "n_features": 4},
        "n_samples": 6,
        "n_features": 4,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 4},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_single_spike_package(root: Path) -> Path:
    header = [str(900 + idx) for idx in range(20)]
    base_a = [1.0 + idx * 0.01 for idx in range(20)]
    base_b = [2.0 + idx * 0.01 for idx in range(20)]
    spiked = list(base_a)
    spiked[10] = 8.0
    _write_rows(
        root / "X.csv",
        [
            header,
            base_a,
            [value + 0.01 for value in base_a],
            [value - 0.01 for value in base_a],
            spiked,
            base_b,
            [value + 0.01 for value in base_b],
            [value - 0.01 for value in base_b],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"], ["S007"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], *[[idx, 900 + idx, "cm-1"] for idx in range(20)]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 7, "n_features": 20},
        "n_samples": 7,
        "n_features": 20,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 20},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_flat_spike_package(root: Path) -> Path:
    header = [str(900 + idx) for idx in range(20)]
    base_a = [1.0 for _ in range(20)]
    base_b = [2.0 for _ in range(20)]
    spiked = list(base_a)
    spiked[10] = 8.0
    _write_rows(
        root / "X.csv",
        [
            header,
            base_a,
            base_a,
            base_a,
            spiked,
            base_b,
            base_b,
            base_b,
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"], ["S007"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], *[[idx, 900 + idx, "cm-1"] for idx in range(20)]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 7, "n_features": 20},
        "n_samples": 7,
        "n_features": 20,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 20},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_resampling_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200"],
            [0.10, 0.11, 0.10, 0.12],
            [0.12, 0.10, 0.11, 0.13],
            [0.11, 0.12, 0.10, 0.12],
            [0.13, 0.11, 0.12, 0.14],
            [0.10, 0.13, 0.11, 0.12],
            [0.12, 0.12, 0.13, 0.11],
            [1.00, 1.05, 1.04, 1.02],
            [1.02, 1.03, 1.05, 1.04],
            [1.04, 1.01, 1.02, 1.03],
            [1.03, 1.04, 1.01, 1.05],
            [0.11, 0.12, 0.10, 0.13],
            [1.05, 1.02, 1.03, 1.04],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], *[[f"S{idx:03d}"] for idx in range(1, 13)]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"], [3, 1200, "cm-1"]])
    _write_rows(root / "y.csv", [["Class"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["B"], ["B"], ["B"]])
    contract = {
        "status": "ready",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 12, "n_features": 4},
        "n_samples": 12,
        "n_features": 4,
        "task_hint": "classification",
        "band_axis": {"file": "band_axis.csv", "unit": "cm-1", "type": "wavenumber", "count": 4},
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def test_basic_checks_find_missing_constant_class_and_intensity(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    result = response["result"]
    assert result["stage"] == "spectral-qc"
    assert result["mode"] == "check"
    assert result["status"] in {"passed", "warning", "blocked"}
    assert result["shape"] == {"n_samples": 5, "n_features": 4}
    checks = {item["check"]: item for item in result["checks"]}
    assert checks["missing_check"]["total_missing_values"] == 1
    assert checks["constant_band_check"]["constant_band_count"] == 1
    assert checks["label_distribution_check"]["class_counts"] == {"A": 2, "B": 3}
    assert {"minor_spike_sample_count", "moderate_spike_sample_count", "severe_spike_sample_count"} <= set(checks["spike_check"])
    assert result["summary"]["missing_values"] == 1
    assert "sample_intensity_quality" in result["checks_run"]
    detection = result["outlier_detection"]
    assert detection["strategy"] == "standard"
    assert "mahalanobis_on_pca" in detection["methods_run"]
    assert "half_resampling_outlier" in detection["advanced_methods_not_run_by_default"]
    for item in detection["high_confidence_outliers"] + detection["medium_confidence_outliers"]:
        assert item["sample_id"]
        assert item["triggered_by"]


def test_supervised_package_without_y_is_blocked(tmp_path: Path) -> None:
    package_dir = _write_supervised_package_without_y(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "blocked"
    assert any(item["code"] == "Y_REQUIRED_FOR_SUPERVISED_TASK" for item in result["blocked_reasons"])


def test_observation_check_writes_lightweight_qc_result_when_output_dir_is_given(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_observation"
    response = qc_spectral_package(package_dir=package_dir, mode="check", output_dir=output_dir)
    assert response["ok"] is True
    assert {path.name for path in output_dir.iterdir()} == {"qc_result.json"}
    result = json.loads((output_dir / "qc_result.json").read_text(encoding="utf-8"))
    assert result["stage"] == "spectral-qc"
    assert result["data_shape"] == {"n_samples": 5, "n_features": 4}
    assert result["output_package"] is None
    assert result["next_package_for_downstream"] == str(package_dir)
    assert result["requires_user_confirmation"] is False


def test_mark_mode_runs_standard_detection_without_writing_package(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="mark", methods=["NOE", "MD", "IQR", "MAD"])
    assert response["ok"] is True
    result = response["result"]
    methods = {item["method_id"]: item for item in result["method_pool_results"]}
    assert methods["NOE"]["outlier_sample_count"] == 0
    assert methods["MD"]["outlier_sample_count"] >= 1
    assert methods["IQR"]["outlier_sample_count"] >= 1
    assert methods["MAD"]["outlier_sample_count"] >= 1
    assert result["mode"] == "mark"
    assert "pca_outlier_check" in result["checks_run"]
    assert not (package_dir / "qc_result.json").exists()
    assert not (package_dir / "X_qc.csv").exists()


def test_mark_mode_can_run_mccv_resampling_outlier_control(tmp_path: Path) -> None:
    package_dir = _write_resampling_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_output"
    response = qc_spectral_package(
        package_dir=package_dir,
        mode="mark",
        methods=["mccv_outlier"],
        output_dir=output_dir,
        n_resamples=12,
        train_ratio=0.7,
        base_model="logistic_regression",
        outlier_metric="misclassification_frequency",
        threshold="percentile_95",
    )
    assert response["ok"] is True
    result = response["result"]
    summary = result["resampling_outlier_control"]
    assert summary["methods_run"] == ["mccv_outlier"]
    assert summary["n_resamples"] == 12
    assert summary["train_ratio"] == 0.7
    assert summary["score_type"] == "misclassification_frequency"
    method = result["method_pool_results"][0]
    assert method["confirmation_required_for_removal"] is True
    assert len(method["outlier_scores"]) == 12
    assert method["risk_semantics"] == "classification_instability_risk"
    assert "classification_instability_candidates" in method
    assert method["input_pipeline"]["base_model"] == "logistic_regression"
    assert "low_evaluation_warning" in method["evaluation_summary"]
    written = json.loads((output_dir / "qc_result.json").read_text(encoding="utf-8"))
    assert written["resampling_outlier_control"]["recommended_action"] == "mark_only"


def test_high_similarity_cross_label_pairs_warn_without_blocking(tmp_path: Path) -> None:
    package_dir = _write_high_similarity_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "warning"
    duplicate = result["duplicate_check"]
    assert duplicate["exact_duplicate_pairs"] == 0
    assert duplicate["strict_near_duplicate_pairs"] == 0
    assert "high_similarity_pairs" not in duplicate
    similarity = result["global_similarity_risk"]
    assert similarity["high_similarity_pairs"] > 0
    assert similarity["cross_label_high_similarity_pairs"] > 0
    assert not any(item["code"] == "EXACT_DUPLICATE_LABEL_CONFLICT" for item in result["blocked_reasons"])


def test_single_local_spike_is_not_severe_without_corroboration(tmp_path: Path) -> None:
    package_dir = _write_single_spike_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    spike = {item["check"]: item for item in response["result"]["checks"]}["spike_check"]
    assert spike["spike_sample_count"] >= 1
    assert spike["severe_spike_sample_count"] == 0
    assert spike["minor_spike_sample_count"] + spike["moderate_spike_sample_count"] == spike["spike_sample_count"]


def test_flat_local_spike_scores_are_capped_when_local_mad_is_too_small(tmp_path: Path) -> None:
    package_dir = _write_flat_spike_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    spike = {item["check"]: item for item in response["result"]["checks"]}["spike_check"]
    assert spike["score_cap"] == 999.0
    assert spike["score_capped_count"] >= 1
    assert spike["local_mad_too_small_count"] >= 1
    capped_samples = [item for item in spike["spike_samples"] if item.get("score_capped")]
    assert capped_samples
    assert max(item["max_spike_score"] for item in capped_samples) <= 999.0


def test_exact_duplicate_label_conflict_blocks(tmp_path: Path) -> None:
    package_dir = _write_exact_label_conflict_package(tmp_path / "reader_package")
    response = qc_spectral_package(package_dir=package_dir, mode="check")
    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "blocked"
    duplicate = result["duplicate_check"]
    assert duplicate["exact_duplicate_pairs"] >= 1
    assert duplicate["exact_duplicate_label_conflicts"] >= 1
    assert any(item["code"] == "EXACT_DUPLICATE_LABEL_CONFLICT" for item in result["blocked_reasons"])


def test_clean_requires_confirmation_before_removing_samples(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_output"
    response = qc_spectral_package(package_dir=package_dir, mode="clean", methods=["IQR"], output_dir=output_dir)
    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["result"]["requires_user_confirmation"] is True
    pending = response["result"]["pending_confirmation"]
    assert pending["action"] == "drop_outlier_samples"
    assert pending["confirmation_requires"] == ["method", "threshold", "deletion_scope", "output_cleaned_package"]
    assert "output_cleaned_package" in pending["required_decisions"]
    methods = {item["id"] for item in pending["outlier_detection_method_options"]}
    assert {"standard_multi_method_consensus", "mahalanobis_on_pca", "half_resampling_outlier", "mccv_outlier"} <= methods
    assert "intersection_spectral_outlier_and_resampling_risk" in methods
    mccv = next(item for item in pending["outlier_detection_method_options"] if item["id"] == "mccv_outlier")
    assert "Do not directly delete MCCV-only candidates" in mccv["cleaning_caution"]
    assert response["result"]["pending_confirmation"]["outlier_candidate_summary"]["high_confidence_count"] >= 0
    assert "standard_multi_method_consensus_high_confidence_only_recommended" in pending["options"]
    assert "intersection_spectral_outlier_and_resampling_risk_recommended_over_mccv_only" in pending["options"]
    assert (output_dir / "qc_result.json").exists()
    assert not (output_dir / "cleaned_package").exists()


def test_confirmed_clean_writes_cleaned_package_shape_and_log(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_output"
    response = qc_spectral_package(
        package_dir=package_dir,
        mode="clean",
        methods=["IQR"],
        output_dir=output_dir,
        confirm_action=True,
        remove_band_indices=[2],
    )
    assert response["ok"] is True
    assert {"qc_result.json", "qc_cleaning_log.json", "cleaned_package"} == {path.name for path in output_dir.iterdir()}
    cleaned = output_dir / "cleaned_package"
    assert set(path.name for path in cleaned.iterdir()) == {"X.csv", "sample_ids.csv", "band_axis.csv", "y.csv", "metadata.csv", "data_contract.json"}
    contract = json.loads((cleaned / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["processing_stage"] == "qc"
    assert contract["files"]["X"] == "X.csv"
    assert contract["shape"]["n_features"] == 3
    assert contract["qc_summary"]["methods_used"] == ["IQR"]
    assert contract["qc_summary"]["removed_sample_count"] >= 1
    assert contract["qc_summary"]["removed_band_count"] == 1
    assert "confidence_scores" not in contract
    assert len(_read_csv(cleaned / "sample_ids.csv")) - 1 == contract["shape"]["n_samples"]
    assert len(_read_csv(cleaned / "band_axis.csv")) - 1 == contract["shape"]["n_features"]
    qc_result = json.loads((output_dir / "qc_result.json").read_text(encoding="utf-8"))
    assert qc_result["status"] == "cleaned"
    assert qc_result["output_package"] == str(cleaned)
    assert qc_result["next_package_for_downstream"] == str(cleaned)
    log = json.loads((output_dir / "qc_cleaning_log.json").read_text(encoding="utf-8"))
    assert log["stage"] == "spectral-qc-clean"
    assert log["input_shape"] == {"n_samples": 5, "n_features": 4}
    assert log["output_shape"]["n_features"] == 3
    assert log["cleaning_actions"][0]["removed_band_count"] == 1


def test_confirmed_clean_can_remove_exact_duplicates_and_updates_downstream_package(tmp_path: Path) -> None:
    package_dir = _write_duplicate_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_output"
    pending = qc_spectral_package(
        package_dir=package_dir,
        mode="clean",
        output_dir=output_dir,
        cleaning_action="remove_exact_duplicates",
    )
    assert pending["ok"] is False
    assert pending["result"]["pending_confirmation"]["action"] == "remove_exact_duplicates"
    assert pending["result"]["pending_confirmation"]["candidate_remove_sample_ids"] == ["S002"]

    response = qc_spectral_package(
        package_dir=package_dir,
        mode="clean",
        output_dir=output_dir,
        confirm_action=True,
        cleaning_action="remove_exact_duplicates",
        cleaning_method="exact_match",
        cleaning_strategy="keep_first",
        threshold=1.0,
        overwrite=True,
    )
    assert response["ok"] is True
    result = response["result"]
    cleaned = Path(result["next_package_for_downstream"])
    assert cleaned.name == "cleaned_package"
    contract = json.loads((cleaned / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["shape"] == {"n_samples": 3, "n_features": 3}
    assert contract["n_samples"] == 3
    assert contract["n_features"] == 3
    assert contract["band_axis"]["count"] == 3
    log = json.loads((output_dir / "qc_cleaning_log.json").read_text(encoding="utf-8"))
    assert log["cleaning_actions"][0]["action"] == "remove_exact_duplicates"
    assert log["cleaning_actions"][0]["removed_samples"][0]["sample_id"] == "S002"


def test_qc_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "reader_package")
    for script in [
        REPO_ROOT / "skills" / "spectral-qc" / "scripts" / "qc_spectral_package.py",
        REPO_ROOT / "scripts" / "qc" / "qc_spectral_package.py",
    ]:
        completed = subprocess.run(
            [sys.executable, str(script), "--package-dir", str(package_dir), "--mode", "methods", "--json"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert {item["method_id"] for item in payload["result"]["methods"]} >= {"NOE", "MD", "PCA_DISTANCE", "ROBUST_ZSCORE", "IQR", "MAD", "half_resampling_outlier", "mccv_outlier"}
        assert {"pca_hotelling_t2", "pca_q_residual", "mahalanobis_on_pca"} <= set(payload["result"]["standard_scheme_methods"])


def test_qc_cli_summary_json_and_export_details_keep_terminal_compact(tmp_path: Path) -> None:
    package_dir = _write_resampling_package(tmp_path / "reader_package")
    output_dir = tmp_path / "qc_output"
    script = REPO_ROOT / "skills" / "spectral-qc" / "scripts" / "qc_spectral_package.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package-dir",
            str(package_dir),
            "--mode",
            "mark",
            "--methods",
            "mccv_outlier",
            "--output-dir",
            str(output_dir),
            "--n-resamples",
            "8",
            "--train-ratio",
            "0.7",
            "--base-model",
            "logistic_regression",
            "--outlier-metric",
            "misclassification_frequency",
            "--threshold",
            "percentile_95",
            "--export-details",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    result = payload["result"]
    assert payload["ok"] is True
    assert "checks" not in result
    assert result["qc_result"] == str(output_dir / "qc_result.json")
    assert result["details_ref"] == str(output_dir / "qc_details.json")
    method = result["method_pool_results"][0]
    assert method["score_count"] == 12
    assert len(method["score_preview"]) <= 20
    written = json.loads((output_dir / "qc_result.json").read_text(encoding="utf-8"))
    assert "checks" not in written
    assert "resampling_details" not in (output_dir / "qc_result.json").read_text(encoding="utf-8")
    details = json.loads((output_dir / "qc_details.json").read_text(encoding="utf-8"))
    assert "checks" in details
    assert "resampling_details" in (output_dir / "qc_details.json").read_text(encoding="utf-8")

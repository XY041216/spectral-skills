from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from spectral_core.optimizer.workflow import optimize_spectral_pipeline


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_deep_budget_preview_expands_embedding_and_classifier_space_without_writes() -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        n_classes=4,
        comparison_depth="deep",
        max_trials=30,
        preview_only=True,
    )
    assert response["ok"] is True
    result = response["result"]
    assert result["files_written"] == []
    assert result["directories_created"] == []
    assert result["budget_audit"]["expanded_trials"] > 300
    assert {"linear_svm", "svm", "lda"} <= set(result["confirmation_card"]["candidate_methods"]["modeling"])
    assert {8, 16, 32} <= set(result["confirmation_card"]["parameter_grid"]["feature.autoencoder_embedding"]["n_components"])


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_classification_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100"],
            [1, 2, 4],
            [2, 5, 3],
            [3, 4, 7],
            [4, 9, 6],
            [8, 9, 12],
            [9, 12, 10],
            [10, 11, 15],
            [12, 13, 16],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"], ["S007"], ["S008"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "nm"], [1, 1000, "nm"], [2, 1100, "nm"]])
    _write_rows(root / "y.csv", [["class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["B"], ["B"]])
    contract = {
        "contract_id": "optimizer-modeling-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 8, "n_features": 3},
        "task_hint": "classification",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_split(root: Path) -> Path:
    rows = [
        ["sample_id", "index", "split"],
        ["S001", 0, "train"],
        ["S002", 1, "train"],
        ["S003", 2, "train"],
        ["S004", 3, "train"],
        ["S005", 4, "val"],
        ["S006", 5, "val"],
        ["S007", 6, "test"],
        ["S008", 7, "test"],
    ]
    _write_rows(root / "split_indices.csv", rows)
    contract = {
        "contract_type": "split_contract",
        "contract_id": "optimizer-split-test",
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 8},
    }
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def test_recommend_from_profile_writes_optimizer_outputs(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="recommend_from_profile",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        n_classes=4,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["test_used_for_selection"] is False
    assert result["data_profile"]["p_over_n"] > 20
    assert result["recommendations"][0]["feature"] == ["pca", "pls_latent_variables", "vip"]
    output_dir = tmp_path / "optimizer"
    assert {"optimizer_contract.json", "candidate_space.json", "recommendation_report.md"} <= {path.name for path in output_dir.iterdir()}
    contract = json.loads((output_dir / "optimizer_contract.json").read_text(encoding="utf-8"))
    assert contract["test_used_for_selection"] is False


def test_tune_method_vip_builds_trial_manifest_without_test_selection(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="tune_method",
        target_step="feature",
        method="vip",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        confirm_budget=True,
        confirm_comparison_design=True,
        confirm_parameter_grid=True,
    )

    assert response["ok"] is True
    assert response["result"]["trial_count"] == 6
    assert response["result"]["selection_metric"] == "val_macro_f1"
    rows = _read_csv(tmp_path / "optimizer" / "trial_manifest.csv")
    assert rows[0][:4] == ["trial_id", "preprocess_method", "feature_method", "model_method"]
    assert rows[1][rows[0].index("test_used_for_selection")] == "False"
    assert "top_k" in rows[1][rows[0].index("params")]


def test_optimize_pipeline_requires_budget_confirmation(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        output_dir=tmp_path / "optimizer",
        max_trials=2,
    )

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "OPTIMIZER_BUDGET_CONFIRMATION_REQUIRED"
    assert response["result"]["files_written"] == []
    assert response["result"]["directories_created"] == []
    assert not (tmp_path / "optimizer").exists()


def test_compare_step_execute_requires_design_and_budget_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    candidate_space = {"preprocess": [{"method": "none"}], "feature": [{"method": "none"}], "modeling": [{"method": "svm"}]}

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="preprocess",
        candidate_space=candidate_space,
        execute_trials=True,
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=3,
        max_trials=1,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "OPTIMIZER_EXECUTION_CONFIRMATION_REQUIRED"
    required = {item["field"] for item in response["result"]["confirmation_required"]}
    assert required == {"comparison_design", "max_trials"}
    assert not (tmp_path / "optimizer" / "trial_results.csv").exists()


def test_optimize_pipeline_execute_requires_candidate_space_confirmation_even_within_budget(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    candidate_space = {"preprocess": [{"method": "none"}], "feature": [{"method": "none"}], "modeling": [{"method": "svm"}]}

    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        candidate_space=candidate_space,
        execute_trials=True,
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=3,
        max_trials=1,
        confirm_budget=True,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "OPTIMIZER_EXECUTION_CONFIRMATION_REQUIRED"
    assert [item["field"] for item in response["result"]["confirmation_required"]] == ["candidate_space"]
    assert not (tmp_path / "optimizer" / "trial_results.csv").exists()


def test_selects_best_from_validation_trial_results(tmp_path: Path) -> None:
    trial_results = tmp_path / "trial_results.csv"
    with trial_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "val_accuracy", "test_accuracy"])
        writer.writeheader()
        writer.writerow({"trial_id": "trial_0001", "val_accuracy": "0.70", "test_accuracy": "0.95"})
        writer.writerow({"trial_id": "trial_0002", "val_accuracy": "0.80", "test_accuracy": "0.60"})

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        trial_results=trial_results,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    best = response["result"]["best_pipeline"]
    assert best["trial_id"] == "trial_0002"
    assert best["selection_metric"] == "val_accuracy"
    assert best["selection_value"] == 0.8
    assert best["test_metrics_ignored_for_selection"] is True
    assert best["ignored_test_metric_columns"] == ["test_accuracy"]
    persisted = json.loads((tmp_path / "optimizer" / "best_pipeline.json").read_text(encoding="utf-8"))
    assert persisted["trial_id"] == "trial_0002"
    contract = json.loads((tmp_path / "optimizer" / "optimizer_contract.json").read_text(encoding="utf-8"))
    assert contract["test_metrics_ignored_for_selection"] is True
    persisted_results = _read_csv(tmp_path / "optimizer" / "trial_results.csv")
    assert len(persisted_results) == 3
    assert "0.80" in persisted_results[2]


def test_existing_trial_results_are_not_replaced_by_blank_manifest(tmp_path: Path) -> None:
    trial_results = tmp_path / "actual_trial_results.csv"
    with trial_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "val_accuracy", "status"])
        writer.writeheader()
        writer.writerow({"trial_id": "trial_0001", "val_accuracy": "0.91", "status": "completed"})

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        trial_results=trial_results,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    rows = _read_csv(tmp_path / "optimizer" / "trial_results.csv")
    assert rows == [["trial_id", "val_accuracy", "status"], ["trial_0001", "0.91", "completed"]]


def test_compare_feature_space_includes_no_feature_baseline(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    feature_methods = [trial["feature_method"] for trial in response["result"]["trials"]]
    assert feature_methods[0] == "none"
    assert "pca" in feature_methods
    assert "pls_latent_variables" in feature_methods
    assert len(response["result"]["trials"]) == 22


def test_compare_feature_quick_depth_keeps_single_point_budget(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        comparison_depth="quick",
    )

    assert response["ok"] is True
    assert len(response["result"]["trials"]) == 6
    assert response["result"]["trials"][2]["feature_params"] == {"n_components": 10}


def test_execute_trials_can_prepare_inputs_and_use_official_modeling_validation_only(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    candidate_space = {
        "preprocess": [{"method": "snv"}],
        "feature": [{"method": "pca", "n_components": [1]}],
        "modeling": [{"method": "logistic_regression", "C": [1.0]}],
    }

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        candidate_space=candidate_space,
        execute_trials=True,
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=2,
        max_trials=1,
        confirm_budget=True,
        confirm_comparison_design=True,
        confirm_parameter_grid=True,
    )

    assert response["ok"] is True
    rows = _read_csv(tmp_path / "optimizer" / "trial_results.csv")
    header = rows[0]
    assert rows[1][header.index("official_modeling_used")] == "True"
    assert rows[1][header.index("test_accessed")] == "False"
    metrics = json.loads((tmp_path / "optimizer" / "trials" / "trial_0001" / "model_output" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_mode"] == "validation_only"
    assert metrics["test_metrics"] == {}
    assert (tmp_path / "optimizer" / "trials" / "trial_0001" / "preprocess_output" / "preprocess_contract.json").exists()
    assert (tmp_path / "optimizer" / "trials" / "trial_0001" / "feature_output" / "feature_state.json").exists()
    assert not (tmp_path / "optimizer" / "trials" / "trial_0001" / "model_output" / "confusion_matrix.csv").exists()


def test_preview_only_returns_confirmation_card_without_writing_files(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        fixed_preprocess_methods="snv",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        max_trials=22,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "needs_confirmation"
    assert result["preview_only"] is True
    assert result["files_written"] == []
    assert result["directories_created"] == []
    assert not (tmp_path / "optimizer").exists()
    card = result["confirmation_card"]
    assert card["fixed_preprocess"] == "snv"
    assert card["validator_model"]["method"] == "svm"
    assert card["validator_model"]["status"] == "needs_user_confirmation"
    assert card["parameter_grid"]["feature.pca"]["n_components"] == [5, 10, 20, 30]
    assert card["parameter_grid"]["feature.pls_latent_variables"]["n_components"] == [3, 5, 10]
    assert card["estimated_trials"] == 22
    assert card["comparison_depth_options"][1]["option"] == "recommended"


def test_pipeline_compact_preview_includes_pls_lv_and_linear_svm_requires_budget_confirmation(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "needs_confirmation"
    assert result["trial_count"] == 72
    assert result["budget_audit"]["budget_exceeded"] is True
    policy = result["candidate_space_policy"]
    assert "pls_latent_variables" in policy["included_methods"]["feature"]
    assert "linear_svm" in policy["included_methods"]["modeling"]
    assert "pls_latent_variables" not in {item["method"] for item in policy["excluded_methods"]}
    assert not (tmp_path / "optimizer").exists()


def test_optimize_pipeline_model_addons_cross_all_preprocess_and_feature_trials() -> None:
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        model_candidates="cls_former_classifier,proto_spectral_classifier,spectral_dkl_gp_classifier",
        max_trials=126,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["trial_count"] == 126
    trials = result["trials"]
    for method in {"cls_former_classifier", "proto_spectral_classifier", "spectral_dkl_gp_classifier"}:
        addon_trials = [trial for trial in trials if trial["model_method"] == method]
        assert len(addon_trials) == 18
        assert {trial["preprocess_method"] for trial in addon_trials} == {"none", "snv", "msc"}
        assert {"none", "pca", "pls_latent_variables", "vip"} <= {trial["feature_method"] for trial in addon_trials}


def test_optimize_pipeline_bcd_deep_model_addons_have_executable_locked_params() -> None:
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        model_candidates="proto_spectral_classifier,spectral_dkl_gp_classifier,cls_former_embedding_svm",
        max_trials=126,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["trial_count"] == 126
    assert result["locked_parameter_check"]["status"] == "passed"
    trials = result["trials"]
    for method in {"proto_spectral_classifier", "spectral_dkl_gp_classifier", "cls_former_embedding_svm"}:
        addon_trials = [trial for trial in trials if trial["model_method"] == method]
        assert len(addon_trials) == 18
        assert {trial["preprocess_method"] for trial in addon_trials} == {"none", "snv", "msc"}
        assert {"none", "pca", "pls_latent_variables", "vip"} <= {trial["feature_method"] for trial in addon_trials}
        assert all(trial["model_params"] for trial in addon_trials)

    params = {trial["model_method"]: trial["model_params"] for trial in trials if trial["preprocess_method"] == "none" and trial["feature_method"] == "none"}
    assert params["proto_spectral_classifier"]["embedding_dim"] == 16
    assert params["proto_spectral_classifier"]["batch_size"] == 8
    assert params["spectral_dkl_gp_classifier"]["embedding_dim"] == 16
    assert params["spectral_dkl_gp_classifier"]["preprojection"] == "pca"
    assert params["cls_former_embedding_svm"]["feature_dim"] == 16
    assert params["cls_former_embedding_svm"]["svm_C"] == 1.0
    assert params["cls_former_embedding_svm"]["svm_gamma"] == "scale"


def test_optimize_pipeline_deep_feature_addon_crosses_preprocess_and_model_axes() -> None:
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        feature_candidates="cls_former_embedding",
        max_trials=84,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["trial_count"] == 84
    cls_trials = [trial for trial in result["trials"] if trial["feature_method"] == "cls_former_embedding"]
    assert len(cls_trials) == 12
    assert {trial["preprocess_method"] for trial in cls_trials} == {"none", "snv", "msc"}
    assert {trial["model_method"] for trial in cls_trials} == {"svm", "linear_svm", "pls_da"}
    assert all(trial["feature_params"]["n_components"] == 16 for trial in cls_trials)


def test_compare_step_execute_requires_parameter_grid_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    candidate_space = {"preprocess": [{"method": "none"}], "feature": [{"method": "pca", "n_components": [5, 10]}], "modeling": [{"method": "svm"}]}

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        candidate_space=candidate_space,
        execute_trials=True,
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=3,
        max_trials=2,
        confirm_budget=True,
        confirm_comparison_design=True,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "OPTIMIZER_EXECUTION_CONFIRMATION_REQUIRED"
    required = {item["field"] for item in response["result"]["confirmation_required"]}
    assert required == {"parameter_grid"}
    assert not (tmp_path / "optimizer" / "trial_results.csv").exists()


def test_trial_inputs_json_accepts_utf8_bom(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    trial_inputs = tmp_path / "trial_inputs.json"
    trial_inputs.write_text(
        "\ufeff"
        + json.dumps(
            {
                "trial_0001": {
                    "package_dir": str(package_dir),
                    "split_contract": str(split_contract),
                }
            }
        ),
        encoding="utf-8",
    )
    candidate_space = {"preprocess": [{"method": "none"}], "feature": [{"method": "none"}], "modeling": [{"method": "logistic_regression", "C": [1.0]}]}

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        candidate_space=candidate_space,
        execute_trials=True,
        trial_inputs=trial_inputs,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=2,
        max_trials=1,
        confirm_budget=True,
        confirm_comparison_design=True,
        confirm_parameter_grid=True,
    )

    assert response["ok"] is True
    assert (tmp_path / "optimizer" / "trial_results.csv").exists()


def test_metric_tie_uses_simpler_fewer_feature_pipeline(tmp_path: Path) -> None:
    trial_results = tmp_path / "trial_results.csv"
    with trial_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_id", "preprocess_method", "feature_method", "model_method", "params", "val_accuracy", "output_n_features"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "trial_0001",
                "preprocess_method": "snv",
                "feature_method": "none",
                "model_method": "svm",
                "params": json.dumps({"preprocess": {}, "feature": {}, "modeling": {}}),
                "val_accuracy": "0.75",
                "output_n_features": "3401",
            }
        )
        writer.writerow(
            {
                "trial_id": "trial_0002",
                "preprocess_method": "snv",
                "feature_method": "pca",
                "model_method": "svm",
                "params": json.dumps({"preprocess": {}, "feature": {"n_components": 5}, "modeling": {}}),
                "val_accuracy": "0.75",
                "output_n_features": "5",
            }
        )

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        trial_results=trial_results,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    best = response["result"]["best_pipeline"]
    assert best["trial_id"] == "trial_0002"
    assert best["tie_breaker"]["tie_count"] == 2
    assert "fewer" in best["tie_breaker"]["selected_reason"]


def test_metric_tie_prefers_snv_over_msc_by_predefined_priority(tmp_path: Path) -> None:
    trial_results = tmp_path / "trial_results.csv"
    with trial_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_id", "preprocess_method", "feature_method", "model_method", "params", "val_macro_f1", "val_accuracy", "output_n_features"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "trial_msc",
                "preprocess_method": "msc",
                "feature_method": "none",
                "model_method": "linear_svm",
                "params": json.dumps({"preprocess": {}, "feature": {}, "modeling": {}}),
                "val_macro_f1": "0.8709",
                "val_accuracy": "0.875",
                "output_n_features": "3401",
            }
        )
        writer.writerow(
            {
                "trial_id": "trial_snv",
                "preprocess_method": "snv",
                "feature_method": "none",
                "model_method": "linear_svm",
                "params": json.dumps({"preprocess": {}, "feature": {}, "modeling": {}}),
                "val_macro_f1": "0.8709",
                "val_accuracy": "0.875",
                "output_n_features": "3401",
            }
        )

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="preprocess",
        trial_results=trial_results,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    best = response["result"]["best_pipeline"]
    assert best["trial_id"] == "trial_snv"
    assert best["tie_breaker"]["primary_metric"] == "val_macro_f1"
    assert "SNV is preferred" in best["tie_breaker"]["selected_reason"]


def test_validator_model_metadata_can_record_inherited_confirmation(tmp_path: Path) -> None:
    candidate_space = {
        "preprocess": [{"method": "snv"}],
        "feature": [{"method": "pca", "n_components": 10}],
        "modeling": [{"method": "svm", "C": [1.0], "gamma": ["scale"]}],
    }

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        candidate_space=candidate_space,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        confirm_budget=True,
        confirm_comparison_design=True,
        validator_model_source="inherited_from_previous_user_confirmed_comparison",
        previous_confirmation_stage="compare_preprocess",
    )

    assert response["ok"] is True
    validator = response["result"]["validator_model"]
    assert validator["method"] == "svm"
    assert validator["source"] == "inherited_from_previous_user_confirmed_comparison"
    assert validator["previous_confirmation_stage"] == "compare_preprocess"
    plan = json.loads((tmp_path / "optimizer" / "optimization_plan.json").read_text(encoding="utf-8"))
    assert plan["validator_model"]["source"] == "inherited_from_previous_user_confirmed_comparison"


def test_compare_preprocess_space_records_sg_parameter_sources(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="preprocess",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        confirm_budget=True,
        confirm_comparison_design=True,
        confirm_parameter_grid=True,
    )

    assert response["ok"] is True
    sg_trials = [trial for trial in response["result"]["trials"] if trial["preprocess_method"] == "sg_smoothing,snv"]
    assert sg_trials
    assert sg_trials[0]["preprocess_params"]["window_length"] == 11
    assert sg_trials[0]["preprocess_params"]["polyorder"] == 2
    assert sg_trials[0]["preprocess_params"]["parameter_source"] == "optimizer_confirmed_default"


def test_compare_preprocess_candidates_parameter_avoids_agent_candidate_space_file(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="preprocess",
        preprocess_candidates="none,snv,msc,detrend,snv_detrend,sg_smoothing,first_derivative",
        model_candidates="svm",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        max_trials=7,
        preview_only=True,
        output_dir=tmp_path / "optimizer",
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "needs_confirmation"
    assert result["trial_count"] == 7
    assert result["files_written"] == []
    assert not (tmp_path / "optimizer").exists()
    card = result["confirmation_card"]
    assert card["candidate_methods"]["preprocess"] == ["none", "snv", "msc", "detrend", "snv_detrend", "sg_smoothing", "first_derivative"]
    assert card["parameter_grid"]["preprocess.sg_smoothing"]["window_length"] == [11]
    assert card["parameter_grid"]["preprocess.first_derivative"]["polyorder"] == [2]


def test_compare_preprocess_validator_params_avoid_agent_input_json(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="preprocess",
        preprocess_candidates="none,snv",
        validator_model="svm",
        validator_params=["C=1.0", "gamma=scale"],
        task_type="classification",
        n_samples=120,
        n_features=3401,
        max_trials=2,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["trial_count"] == 2
    assert result["files_written"] == []
    card = result["confirmation_card"]
    assert card["validator_model"]["method"] == "svm"
    assert card["validator_model"]["parameter_grid"]["svm"] == {"C": [1.0], "gamma": ["scale"]}
    assert all(trial["model_params"] == {"C": 1.0, "gamma": "scale"} for trial in result["trials"])


def test_model_param_grid_applies_to_named_candidates_without_json(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        model_candidates="svm,linear_svm",
        model_param_grid=["svm.C=1|10", "svm.gamma=scale", "linear_svm.C=1"],
        task_type="classification",
        n_samples=120,
        n_features=10,
        max_trials=3,
        preview_only=True,
    )

    assert response["ok"] is True
    trials = response["result"]["trials"]
    assert [trial["model_params"] for trial in trials if trial["model_method"] == "svm"] == [
        {"C": 1, "gamma": "scale"},
        {"C": 10, "gamma": "scale"},
    ]
    assert [trial["model_params"] for trial in trials if trial["model_method"] == "linear_svm"] == [{"C": 1}]


def test_model_compare_preview_supplies_executable_fixed_parameter_card(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        model_candidates="logistic_regression,linear_svm,svm,lda,qda,gaussian_nb,knn_classifier,random_forest_classifier",
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3,
        max_trials=8,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["trial_count"] == 8
    assert result["locked_parameter_check"]["status"] == "passed"
    assert result["missing_locked_params"] == {}
    assert result["files_written"] == []
    assert not (tmp_path / "optimizer").exists()
    params = {trial["model_method"]: trial["model_params"] for trial in result["trials"]}
    assert params["svm"] == {"C": 1.0, "gamma": "scale"}
    assert params["logistic_regression"] == {"C": 1.0}
    assert params["linear_svm"] == {"C": 1.0}
    assert params["qda"] == {"reg_param": 0.1}
    assert params["knn_classifier"] == {"n_neighbors": 5}
    assert params["random_forest_classifier"] == {"max_depth": 5, "n_estimators": 100}


def test_incomplete_custom_model_space_is_blocked_before_materialization(tmp_path: Path) -> None:
    candidate_space = {
        "preprocess": [{"method": "none"}],
        "feature": [{"method": "none"}],
        "modeling": [{"method": "svm", "gamma": ["scale"]}, {"method": "qda"}],
    }
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        candidate_space=candidate_space,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3,
        max_trials=2,
        confirm_budget=True,
        confirm_comparison_design=True,
        confirm_parameter_grid=True,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "OPTIMIZER_LOCKED_MODEL_PARAMS_CONFIRMATION_REQUIRED"
    result = response["result"]
    assert result["status"] == "needs_confirmation"
    assert result["missing_locked_params"] == {"svm": ["C"], "qda": ["reg_param"]}
    assert result["recommended_locked_params"] == {"svm": {"C": 1.0}, "qda": {"reg_param": 0.1}}
    assert result["files_written"] == []
    assert result["directories_created"] == []
    assert not (tmp_path / "optimizer").exists()


def test_incomplete_model_preview_reports_all_missing_locked_params_without_side_effects(tmp_path: Path) -> None:
    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        candidate_space={
            "preprocess": [{"method": "none"}],
            "feature": [{"method": "none"}],
            "modeling": [
                {"method": "svm"},
                {"method": "logistic_regression"},
                {"method": "linear_svm"},
                {"method": "qda"},
                {"method": "knn_classifier"},
                {"method": "random_forest_classifier"},
            ],
        },
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3,
        max_trials=6,
        preview_only=True,
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "needs_confirmation"
    assert result["missing_locked_params"] == {
        "svm": ["C", "gamma"],
        "logistic_regression": ["C"],
        "linear_svm": ["C"],
        "qda": ["reg_param"],
        "knn_classifier": ["n_neighbors"],
        "random_forest_classifier": ["n_estimators", "max_depth"],
    }
    assert result["files_written"] == []
    assert result["directories_created"] == []
    assert not (tmp_path / "optimizer").exists()


def test_small_sample_best_pipeline_records_repeated_holdout_followup(tmp_path: Path) -> None:
    trial_results = tmp_path / "trial_results.csv"
    _write_rows(
        trial_results,
        [
            ["trial_id", "preprocess_method", "feature_method", "model_method", "params", "val_macro_f1", "val_accuracy", "train_accuracy", "status"],
            ["trial_0001", "none", "pls_latent_variables", "svm", '{"feature":{"n_components":10},"modeling":{"C":1.0,"gamma":"scale"}}', 0.875, 0.875, 1.0, "completed"],
        ],
    )
    response = optimize_spectral_pipeline(
        mode="optimize_pipeline",
        trial_results=trial_results,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
        max_trials=72,
        confirm_budget=True,
        confirm_candidate_space=True,
    )

    assert response["ok"] is True
    followup = response["result"]["stability_followup"]
    assert followup["protocol"] == "stratified_repeated_holdout"
    assert followup["n_repeats_options"] == [5, 10]
    assert followup["lock_selected_pipeline"] is True
    assert followup["selection_metric"] == "macro_f1"
    assert "near_perfect_train_accuracy_overfit_signal" in followup["trigger_reasons"]
    contract = json.loads((tmp_path / "optimizer" / "optimizer_contract.json").read_text(encoding="utf-8"))
    best = json.loads((tmp_path / "optimizer" / "best_pipeline.json").read_text(encoding="utf-8"))
    assert contract["stability_followup"] == followup
    assert best["stability_followup"] == followup


def test_compare_model_candidates_with_fixed_feature_contract_executes_without_candidate_space_file(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    feature_contract = tmp_path / "feature_contract.json"
    feature_contract.write_text(
        json.dumps(
            {
                "contract_type": "feature_contract",
                "stage": "spectral-feature",
                "output_package": str(package_dir / "data_contract.json"),
                "split_contract": str(split_contract),
                "execution_mode": "holdout",
                "feature_method": "none",
                "feature_mode": "unchanged",
                "requires_y": False,
                "upstream_preprocess": {"applied": False, "methods": ["none"]},
            }
        ),
        encoding="utf-8",
    )

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="modeling",
        fixed_feature_contract=feature_contract,
        model_candidates="logistic_regression,linear_svm,svm,lda",
        execute_trials=True,
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=8,
        n_features=3,
        max_trials=4,
        confirm_budget=True,
        confirm_comparison_design=True,
    )

    assert response["ok"] is True
    assert response["result"]["trial_count"] == 4
    rows = _read_csv(tmp_path / "optimizer" / "trial_results.csv")
    header = rows[0]
    assert {row[header.index("model_method")] for row in rows[1:]} == {"logistic_regression", "linear_svm", "svm", "lda"}
    assert all(row[header.index("official_modeling_used")] == "True" for row in rows[1:])
    assert all(row[header.index("test_accessed")] == "False" for row in rows[1:])


def test_optimizer_records_post_test_evaluation_context_in_outputs(tmp_path: Path) -> None:
    trial_results = tmp_path / "trial_results.csv"
    with trial_results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "val_macro_f1"])
        writer.writeheader()
        writer.writerow({"trial_id": "trial_0001", "val_macro_f1": "0.8"})
    test_log = tmp_path / "test_access_log.json"
    test_log.write_text(
        json.dumps({"test_set_accessed": True, "first_access_time": "2026-06-26T10:59:26+08:00"}),
        encoding="utf-8",
    )

    response = optimize_spectral_pipeline(
        mode="compare_step",
        target_step="feature",
        trial_results=trial_results,
        test_access_log=test_log,
        output_dir=tmp_path / "optimizer",
        task_type="classification",
        n_samples=120,
        n_features=3401,
    )

    assert response["ok"] is True
    context = response["result"]["evaluation_context"]
    assert context["evaluation_context"] == "post_test_exploratory"
    contract = json.loads((tmp_path / "optimizer" / "optimizer_contract.json").read_text(encoding="utf-8"))
    assert contract["evaluation_context"]["blind_test_available"] is False
    best = json.loads((tmp_path / "optimizer" / "best_pipeline.json").read_text(encoding="utf-8"))
    assert best["evaluation_context"]["prior_test_access_detected"] is True


def test_optimizer_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-optimizer" / "scripts" / "optimize_spectral_pipeline.py",
            REPO_ROOT / "scripts" / "optimizer" / "optimize_spectral_pipeline.py",
        ]
    ):
        output_dir = tmp_path / f"optimizer_{idx}"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--mode",
                "tune_method",
                "--target-step",
                "feature",
                "--method",
                "vip",
                "--task-type",
                "classification",
                "--n-samples",
                "120",
                "--n-features",
                "3401",
                "--output-dir",
                str(output_dir),
                "--confirm-budget",
                "--confirm-comparison-design",
                "--confirm-parameter-grid",
                "--json",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert payload["result"]["trial_count"] == 6
        assert (output_dir / "optimizer_contract.json").exists()


def test_optimizer_cli_accepts_candidate_list_arguments_without_candidate_space_file(tmp_path: Path) -> None:
    script = REPO_ROOT / "skills" / "spectral-optimizer" / "scripts" / "optimize_spectral_pipeline.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "compare_step",
            "--target-step",
            "preprocess",
            "--preprocess-candidates",
            "none,snv,msc,detrend,snv_detrend,sg_smoothing,first_derivative",
            "--model-candidates",
            "svm",
            "--validator-model",
            "svm",
            "--validator-param",
            "C=1.0",
            "--validator-param",
            "gamma=scale",
            "--task-type",
            "classification",
            "--n-samples",
            "120",
            "--n-features",
            "3401",
            "--max-trials",
            "7",
            "--output-dir",
            str(tmp_path / "optimizer"),
            "--preview-only",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["result"]["trial_count"] == 7
    assert payload["result"]["confirmation_card"]["validator_model"]["parameter_grid"]["svm"] == {"C": [1.0], "gamma": ["scale"]}
    assert not (tmp_path / "optimizer").exists()

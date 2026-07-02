from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from spectral_core.modeling import workflow as modeling_workflow
from spectral_core.feature.workflow import feature_spectral_package
from spectral_core.modeling.methods import _classification_estimator, _locked_parameter_grid, _optional_boosting_grid, _select_candidates, parse_models
from spectral_core.modeling.workflow import model_spectral_package
from spectral_core.preprocess.workflow import preprocess_spectral_package


REPO_ROOT = Path(__file__).resolve().parents[3]
LONG_SPLIT_HEADER = ["split_type", "method", "fold_id", "repeat_id", "role", "sample_index", "sample_id", "label", "group_id"]


def test_regular_classification_grids_cover_key_overfit_controls() -> None:
    _, svm = _classification_estimator("svm", random_seed=42, n_train=72, n_features=3401)
    assert {"model__C", "model__gamma", "model__class_weight"} <= set(svm)
    assert len(svm["model__C"]) >= 4 and len(svm["model__gamma"]) >= 3

    _, knn = _classification_estimator("knn_classifier", random_seed=42, n_train=72, n_features=3401)
    assert {"model__n_neighbors", "model__weights", "model__metric"} <= set(knn)
    assert 9 in knn["model__n_neighbors"]

    _, forest = _classification_estimator("random_forest_classifier", random_seed=42, n_train=72, n_features=3401)
    assert {"n_estimators", "max_depth", "max_features", "min_samples_leaf"} <= set(forest)

    _, lda = _classification_estimator("lda", random_seed=42, n_train=72, n_features=3401)
    assert "model__shrinkage" in lda and "auto" in lda["model__shrinkage"]


def test_classification_candidate_selection_uses_macro_f1() -> None:
    X = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [0.15], [1.15]])
    y = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
    estimator, _ = _classification_estimator("linear_svm", random_seed=42, n_train=6, n_features=1)
    candidates = _select_candidates(
        estimator,
        {"model__C": [0.1, 1.0]},
        "linear_svm",
        X,
        y,
        [0, 1, 2, 3, 4, 5],
        [6, 7],
        task_type="classification",
        cv_folds=2,
    )
    assert candidates
    assert {item["selection_metric"] for item in candidates} == {"macro_f1"}


def test_feature_contract_modeling_gate_blocks_tsne_and_gates_umap(tmp_path: Path) -> None:
    contract_path = tmp_path / "feature_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_type": "feature_contract",
                "feature_method": "tsne_embedding",
                "intended_use": "visualization",
                "out_of_sample_transform": "unsupported",
                "handoff": {"spectral_modeling": {"ready": False, "blocked": True}},
            }
        ),
        encoding="utf-8",
    )
    blocked = modeling_workflow._feature_contract_modeling_gate(contract_path, confirmed=True)
    assert blocked["status"] == "blocked"
    assert blocked["code"] == "FEATURE_NOT_MODELING_COMPATIBLE"

    contract_path.write_text(
        json.dumps(
            {
                "contract_type": "feature_contract",
                "feature_method": "umap_embedding",
                "intended_use": "visualization_or_confirmed_modeling",
                "out_of_sample_transform": "supported",
                "modeling_requires_confirmation": True,
                "handoff": {"spectral_modeling": {"ready": False, "requires_confirmation": True}},
            }
        ),
        encoding="utf-8",
    )
    gated = modeling_workflow._feature_contract_modeling_gate(contract_path, confirmed=False)
    assert gated["status"] == "needs_confirmation"
    assert gated["code"] == "GATED_FEATURE_MODELING_CONFIRMATION_REQUIRED"
    assert modeling_workflow._feature_contract_modeling_gate(contract_path, confirmed=True)["status"] == "ready"


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_classification_package(root: Path, *, with_y: bool = True) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100"],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [8, 9, 10],
            [9, 10, 11],
            [10, 11, 12],
            [11, 12, 13],
            [12, 13, 14],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"], ["S007"], ["S008"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "nm"], [1, 1000, "nm"], [2, 1100, "nm"]])
    if with_y:
        _write_rows(root / "y.csv", [["class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["B"], ["B"]])
    contract = {
        "contract_id": "classification-modeling-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv" if with_y else None, "metadata": None},
        "shape": {"n_samples": 8, "n_features": 3},
        "task_hint": "classification",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_regression_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["PC1", "PC2"],
            [1, 0.5],
            [2, 1.4],
            [3, 1.1],
            [4, 2.8],
            [5, 2.2],
            [6, 3.7],
            [7, 3.1],
            [8, 4.4],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["R001"], ["R002"], ["R003"], ["R004"], ["R005"], ["R006"], ["R007"], ["R008"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, "PC1", "principal_component"], [1, "PC2", "principal_component"]])
    _write_rows(root / "y.csv", [["target"], [1.5], [2.9], [4.4], [6.1], [7.6], [9.1], [10.4], [12.0]])
    contract = {
        "contract_id": "regression-modeling-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 8, "n_features": 2},
        "task_hint": "regression",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_split(root: Path, *, prefix: str = "S", no_val: bool = False, no_test: bool = False) -> Path:
    ids = [f"{prefix}{idx:03d}" for idx in range(1, 9)]
    if no_val:
        rows = [["sample_id", "index", "split"], [ids[0], 0, "train"], [ids[1], 1, "train"], [ids[2], 2, "train"], [ids[3], 3, "train"], [ids[4], 4, "train"], [ids[5], 5, "train"], [ids[6], 6, "test"], [ids[7], 7, "test"]]
    elif no_test:
        rows = [["sample_id", "index", "split"], [ids[0], 0, "train"], [ids[1], 1, "train"], [ids[2], 2, "train"], [ids[3], 3, "train"], [ids[4], 4, "train"], [ids[5], 5, "train"], [ids[6], 6, "val"], [ids[7], 7, "val"]]
    else:
        rows = [["sample_id", "index", "split"], [ids[0], 0, "train"], [ids[1], 1, "train"], [ids[2], 2, "train"], [ids[3], 3, "train"], [ids[4], 4, "val"], [ids[5], 5, "val"], [ids[6], 6, "test"], [ids[7], 7, "test"]]
    _write_rows(root / "split_indices.csv", rows)
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-modeling-test",
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 8},
    }
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_new_holdout_split(root: Path, *, prefix: str = "S") -> Path:
    ids = [f"{prefix}{idx:03d}" for idx in range(1, 9)]
    assignments = {"train": [0, 1, 2, 3], "val": [4, 5], "test": [6, 7]}
    rows = [LONG_SPLIT_HEADER]
    for role, indices in assignments.items():
        for idx in indices:
            rows.append(["holdout", "stratified", "", "", role, idx, ids[idx], "", ""])
    _write_rows(root / "split_indices.csv", rows)
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-new-holdout-modeling-test",
        "split_type": "holdout",
        "method": "stratified",
        "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
        "indices": assignments,
        "sample_ids": {role: [ids[idx] for idx in indices] for role, indices in assignments.items()},
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 8, "train": 4, "val": 2, "test": 2},
    }
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_cv_split(root: Path) -> Path:
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-cv-modeling-test",
        "split_type": "cross_validation",
        "method": "stratified_kfold",
        "n_splits": 4,
        "folds": [{"fold_id": 1, "train_indices": [0, 1, 2, 3, 4, 5], "val_indices": [6, 7]}],
    }
    path = root / "split_contract.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def test_best_pipeline_lock_replays_optimizer_feature_contract(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    trial_dir = tmp_path / "optimizer" / "trials" / "trial_0001"
    feature_dir = trial_dir / "feature_output"
    feature_response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=feature_dir,
        method="pca",
        n_components=2,
        overwrite=True,
    )
    assert feature_response["ok"] is True
    best_pipeline = tmp_path / "optimizer" / "best_pipeline.json"
    best_pipeline.parent.mkdir(parents=True, exist_ok=True)
    best_pipeline.write_text(
        json.dumps(
            {
                "trial_id": "trial_0001",
                "selection_metric": "val_macro_f1",
                "selection_value": 0.75,
                "row": {
                    "trial_id": "trial_0001",
                    "preprocess_method": "none",
                    "feature_method": "pca",
                    "model_method": "linear_svm",
                    "params": json.dumps({"preprocess": {}, "feature": {"n_components": 2}, "modeling": {"C": 1.0}}, sort_keys=True),
                    "modeling_output": str(trial_dir / "model_output"),
                    "val_macro_f1": "0.75",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "confirmatory_best",
        task_type="classification",
        best_pipeline=best_pipeline,
        lock_best_pipeline_params=True,
        evaluation_mode="final",
        require_test_confirmation=True,
        confirm_test_evaluation=True,
        overwrite=True,
    )

    assert response["ok"] is True
    contract = json.loads((tmp_path / "confirmatory_best" / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["experiment_conditions"]["feature"]["methods"] == ["pca"]
    assert contract["experiment_conditions"]["n_features"] == 2
    reproduction = contract["best_pipeline_reproduction"]
    assert reproduction["trial_id"] == "trial_0001"
    assert reproduction["input_stage"] == "feature"
    assert reproduction["full_pipeline_reproduced"] is True
    assert Path(reproduction["resolved_input_contract"]).resolve() == (feature_dir / "feature_contract.json").resolve()


def test_best_pipeline_lock_blocks_missing_feature_contract(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    best_pipeline = tmp_path / "optimizer" / "best_pipeline.json"
    best_pipeline.parent.mkdir(parents=True, exist_ok=True)
    best_pipeline.write_text(
        json.dumps(
            {
                "trial_id": "trial_0002",
                "row": {
                    "trial_id": "trial_0002",
                    "preprocess_method": "none",
                    "feature_method": "pca",
                    "model_method": "linear_svm",
                    "params": json.dumps({"feature": {"n_components": 2}, "modeling": {"C": 1.0}}, sort_keys=True),
                    "modeling_output": str(tmp_path / "optimizer" / "trials" / "trial_0002" / "model_output"),
                    "val_macro_f1": "0.75",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "should_not_write",
        task_type="classification",
        best_pipeline=best_pipeline,
        lock_best_pipeline_params=True,
        evaluation_mode="final",
        require_test_confirmation=True,
        confirm_test_evaluation=True,
        overwrite=True,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "BEST_PIPELINE_UPSTREAM_CONTRACT_MISSING"


def _write_repeated_split(root: Path) -> Path:
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-repeated-modeling-test",
        "split_type": "repeated_holdout",
        "method": "stratified_monte_carlo_cv",
        "n_repeats": 2,
        "repeats": [
            {"repeat_id": 1, "train_indices": [0, 1, 3, 4], "val_indices": [2, 5], "test_indices": [6, 7]},
            {"repeat_id": 2, "train_indices": [2, 3, 6, 7], "val_indices": [0, 4], "test_indices": [1, 5]},
        ],
    }
    path = root / "split_contract.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _read_numeric_X(path: Path) -> list[list[float]]:
    return [[float(value) for value in row] for row in _read_csv(path)[1:]]


def test_classification_modeling_writes_metrics_predictions_contract_and_confusion_matrix(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="classification", models="logistic_regression,random_forest_classifier", random_seed=7)

    assert response["ok"] is True
    assert {"modeling_contract.json", "modeling_summary.json", "metrics.json", "predictions.csv", "model_artifact.pkl", "confusion_matrix.csv", "pipeline_bundle"} <= {path.name for path in output_dir.iterdir()}
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["contract_type"] == "modeling_contract"
    assert contract["selection_strategy"]["test_used_for_selection"] is False
    assert contract["selection_strategy"]["tuning_split"] == "val"
    assert contract["experiment_conditions"]["candidate_models"] == ["logistic_regression", "random_forest_classifier"]
    assert "condition_statement" in contract["experiment_conditions"]
    summary = json.loads((output_dir / "modeling_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_model"] == contract["model_type"]
    assert any(note["type"] == "class_level_review" for note in summary["interpretation_notes"])
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics["test_metrics"]
    rows = _read_csv(output_dir / "predictions.csv")
    assert rows[0][:4] == ["sample_id", "split", "y_true", "y_pred"]
    assert len(rows) == 9
    test_log = json.loads((output_dir / "test_access_log.json").read_text(encoding="utf-8"))
    assert test_log["test_set_accessed"] is True
    assert test_log["first_access_stage"] == "final_modeling"
    assert (output_dir / "pipeline_bundle" / "pipeline_artifact.pkl").exists()
    manifest = json.loads((output_dir / "pipeline_bundle" / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contracts"]["modeling_contract"] == "modeling_contract.json"
    assert manifest["contracts"]["split_contract"] == "split_contract.json"
    assert manifest["metrics"] == "metrics.json"


def test_validation_only_modeling_does_not_read_or_write_test_outputs(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        task_type="classification",
        models="logistic_regression",
        random_seed=7,
        evaluation_mode="validation_only",
    )

    assert response["ok"] is True
    assert not (output_dir / "confusion_matrix.csv").exists()
    assert not (output_dir / "model_artifact.pkl").exists()
    assert not (output_dir / "test_access_log.json").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_mode"] == "validation_only"
    assert metrics["test_accessed"] is False
    assert metrics["test_metrics"] == {}
    predictions = _read_csv(output_dir / "predictions.csv")
    split_column = predictions[0].index("split")
    assert {row[split_column] for row in predictions[1:]} == {"train", "val"}
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["selection_strategy"]["evaluation_mode"] == "validation_only"
    assert contract["selection_strategy"]["test_evaluated"] is False


def test_validation_only_multiclassifier_writes_full_classifier_summary(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        task_type="classification",
        models="lda,linear_svm",
        random_seed=7,
        evaluation_mode="validation_only",
    )

    assert response["ok"] is True
    summary_path = output_dir / "classifier_validation_summary.csv"
    assert summary_path.exists()
    rows = _read_csv(summary_path)
    header = rows[0]
    assert len(rows) == 3
    assert {"model_method", "train_accuracy", "train_balanced_accuracy", "train_macro_f1", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "model_parameters_json", "test_accessed"} <= set(header)
    model_column = header.index("model_method")
    assert {row[model_column] for row in rows[1:]} == {"lda", "linear_svm"}
    test_access_column = header.index("test_accessed")
    assert {row[test_access_column] for row in rows[1:]} == {"False"}
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_accessed"] is False
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["outputs"]["classifier_validation_summary"] == "classifier_validation_summary.csv"


def test_final_holdout_test_can_require_explicit_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    blocked = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model_blocked",
        task_type="classification",
        models="logistic_regression",
        require_test_confirmation=True,
    )

    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "TEST_EVALUATION_CONFIRMATION_REQUIRED"
    assert not (tmp_path / "model_blocked").exists()

    confirmed = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model_confirmed",
        task_type="classification",
        models="logistic_regression",
        require_test_confirmation=True,
        confirm_test_evaluation=True,
    )

    assert confirmed["ok"] is True
    assert (tmp_path / "model_confirmed" / "test_access_log.json").exists()


def test_new_holdout_split_contract_indices_are_supported(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="classification", models="random_forest_classifier", random_seed=7)

    assert response["ok"] is True
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["experiment_conditions"]["split"]["split_type"] == "holdout"
    assert contract["experiment_conditions"]["split"]["method"] == "stratified"


def test_cv_split_contract_runs_fold_wise_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_cv_split(tmp_path / "split")

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "model", task_type="classification", models="random_forest_classifier")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "cv_modeling_result.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "cross_validation"
    assert contract["execution_mode"] == "fold_wise"
    assert contract["leakage_guard"]["fit_on"] == "train_only_for_each_partition"
    assert (tmp_path / "model" / "fold_metrics.csv").exists()
    assert (tmp_path / "model" / "fold_predictions.csv").exists()


def test_feature_contract_runs_fold_wise_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_cv_split(tmp_path / "split")
    feature_response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "feature", method="pca", n_components=2)
    assert feature_response["ok"] is True

    response = model_spectral_package(feature_contract=tmp_path / "feature" / "feature_contract.json", output_dir=tmp_path / "model", task_type="classification", models="random_forest_classifier")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "cv_modeling_result.json").read_text(encoding="utf-8"))
    assert contract["input_contract"].endswith("feature_contract.json")
    assert contract["split_type"] == "cross_validation"
    assert (tmp_path / "model" / "fold_metrics.csv").exists()


def test_preprocess_contract_runs_fold_wise_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_cv_split(tmp_path / "split")
    preprocess_response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="snv")
    assert preprocess_response["ok"] is True

    response = model_spectral_package(preprocess_contract=tmp_path / "preprocess" / "preprocess_contract.json", output_dir=tmp_path / "model", task_type="classification", models="random_forest_classifier")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "cv_modeling_result.json").read_text(encoding="utf-8"))
    assert contract["input_contract"].endswith("preprocess_contract.json")
    assert contract["split_type"] == "cross_validation"
    assert contract["hyperparameter_selection"]["strategy"] == "inner_cv"
    assert contract["hyperparameter_selection"]["selection_scope"] == "outer_train_only"
    assert contract["hyperparameter_selection"]["outer_validation_used_for_selection"] is False
    rows = _read_csv(tmp_path / "model" / "fold_metrics.csv")
    assert "outer_validation_used_for_selection" in rows[0]
    assert rows[1][rows[0].index("outer_validation_used_for_selection")] == "False"


def test_holdout_preprocess_contract_runs_standard_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    preprocess_response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="snv")
    assert preprocess_response["ok"] is True

    response = model_spectral_package(preprocess_contract=tmp_path / "preprocess" / "preprocess_contract.json", output_dir=tmp_path / "model", task_type="classification", models="random_forest_classifier")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["input_contract"].endswith("preprocess_contract.json")
    assert contract["selection_strategy"]["test_used_for_selection"] is False
    conditions = contract["experiment_conditions"]
    assert conditions["input_source"] == "preprocess_contract"
    assert conditions["preprocess"]["methods"] == ["snv"]
    assert conditions["feature"]["methods"] == ["none"]
    assert "preprocess=snv" in conditions["condition_statement"]
    assert "preprocessing=none unless" not in conditions["condition_statement"]
    summary = json.loads((tmp_path / "model" / "modeling_summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_conditions"]["preprocess"]["methods"] == ["snv"]
    assert (tmp_path / "model" / "metrics.json").exists()


def test_holdout_feature_contract_pipeline_bundle_copies_feature_and_preprocess_contracts(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    preprocess_response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="none")
    assert preprocess_response["ok"] is True
    feature_response = feature_spectral_package(
        preprocess_contract=tmp_path / "preprocess" / "preprocess_contract.json",
        output_dir=tmp_path / "feature",
        method="pca",
        n_components=2,
        task_type="classification",
    )
    assert feature_response["ok"] is True

    response = model_spectral_package(
        feature_contract=tmp_path / "feature" / "feature_contract.json",
        output_dir=tmp_path / "model",
        task_type="classification",
        models="svm",
        random_seed=7,
    )

    assert response["ok"] is True
    bundle = tmp_path / "model" / "pipeline_bundle"
    assert (bundle / "pipeline_artifact.pkl").exists()
    assert (bundle / "feature_contract.json").exists()
    assert (bundle / "preprocess_contract.json").exists()
    assert (bundle / "modeling_contract.json").exists()
    manifest = json.loads((bundle / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contracts"]["feature_contract"] == "feature_contract.json"
    assert manifest["contracts"]["preprocess_contract"] == "preprocess_contract.json"
    contract = json.loads((tmp_path / "model" / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["experiment_conditions"]["preprocess"]["methods"] == ["none"]
    assert contract["experiment_conditions"]["feature"]["methods"] == ["pca"]


def test_locked_optimizer_best_pipeline_params_disable_final_model_selection(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    best_pipeline = tmp_path / "best_pipeline.json"
    best_pipeline.write_text(
        json.dumps(
            {
                "trial_id": "trial_linear_svm_c1",
                "row": {
                    "model_method": "linear_svm",
                    "params": json.dumps({"modeling": {"C": 1.0}}),
                },
            }
        ),
        encoding="utf-8",
    )

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model",
        task_type="classification",
        best_pipeline=best_pipeline,
        lock_best_pipeline_params=True,
    )

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["model_type"] == "linear_svm"
    assert contract["model_parameters"] == {"model__C": 1.0}
    assert contract["model_selection_mode"] == "locked_from_optimizer_best"
    assert contract["param_search_enabled"] is False
    assert contract["model_params_source"] == "best_pipeline.json"
    assert contract["selection_strategy"]["candidate_count"] == 1
    assert contract["selection_strategy"]["hyperparameter_selection"]["strategy"] == "locked_from_optimizer_best"


def test_locked_optimizer_best_pipeline_missing_tunable_param_blocks(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    best_pipeline = tmp_path / "best_pipeline.json"
    best_pipeline.write_text(
        json.dumps({"row": {"model_method": "linear_svm", "params": json.dumps({"modeling": {}})}}),
        encoding="utf-8",
    )

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model",
        task_type="classification",
        best_pipeline=best_pipeline,
        lock_best_pipeline_params=True,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "LOCKED_MODEL_PARAMETER_MISSING"
    assert not (tmp_path / "model").exists()


def test_prior_test_access_requires_confirmatory_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_access_log.json").write_text(
        json.dumps({"test_set_accessed": True, "first_access_time": "2026-06-26T10:59:26+08:00"}),
        encoding="utf-8",
    )

    blocked = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=run_dir / "model_output",
        task_type="classification",
        models="svm",
        confirm_test_evaluation=True,
    )

    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "CONFIRMATORY_TEST_EVALUATION_REQUIRED"
    assert not (run_dir / "model_output").exists()

    confirmed = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=run_dir / "model_output",
        task_type="classification",
        models="svm",
        confirm_test_evaluation=True,
        confirm_confirmatory_test_evaluation=True,
    )
    assert confirmed["ok"] is True


def test_pipeline_artifact_predicts_from_raw_spectra_through_snv_pls_and_model(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    _write_rows(
        package_dir / "X.csv",
        [
            ["900", "1000", "1100"],
            [1.0, 2.0, 4.0],
            [1.4, 2.6, 3.6],
            [2.0, 2.1, 5.0],
            [4.2, 3.1, 1.0],
            [4.7, 3.8, 1.4],
            [5.1, 2.9, 2.2],
            [2.2, 6.0, 1.8],
            [2.8, 6.4, 2.4],
        ],
    )
    split_contract = _write_new_holdout_split(tmp_path / "split")
    preprocess_response = preprocess_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "preprocess",
        methods="snv",
    )
    assert preprocess_response["ok"] is True
    feature_response = feature_spectral_package(
        preprocess_contract=tmp_path / "preprocess" / "preprocess_contract.json",
        output_dir=tmp_path / "feature",
        method="pls_latent_variables",
        n_components=1,
        task_type="classification",
    )
    assert feature_response["ok"] is True
    response = model_spectral_package(
        feature_contract=tmp_path / "feature" / "feature_contract.json",
        output_dir=tmp_path / "model",
        task_type="classification",
        models="svm",
        random_seed=7,
    )
    assert response["ok"] is True

    with (tmp_path / "model" / "pipeline_bundle" / "pipeline_artifact.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    raw_X = _read_numeric_X(package_dir / "X.csv")
    predictions = artifact.predict_raw(raw_X[:2])
    transformed = artifact.transform_raw(raw_X[:2])
    manifest = json.loads((tmp_path / "model" / "pipeline_bundle" / "pipeline_manifest.json").read_text(encoding="utf-8"))
    smoke = json.loads((tmp_path / "model" / "pipeline_bundle" / "smoke_test.json").read_text(encoding="utf-8"))

    assert len(predictions) == 2
    assert transformed.shape == (2, 1)
    assert manifest["raw_prediction"]["supported"] is True
    assert manifest["input_schema"]["raw_n_features"] == 3
    assert smoke["status"] == "passed"
    assert smoke["raw_spectra_input_shape"] == [3, 3]
    assert smoke["prediction_success"] is True


def test_repeated_split_contract_runs_repeat_wise_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_repeated_split(tmp_path / "split")

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "model", task_type="classification", models="random_forest_classifier")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "repeated_modeling_result.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "repeated_holdout"
    assert contract["execution_mode"] == "repeat_wise"
    assert contract["eval_role"] == "test"
    assert (tmp_path / "model" / "repeat_metrics.csv").exists()
    assert (tmp_path / "model" / "repeat_predictions.csv").exists()


def test_model_set_aliases_distinguish_regular_fast_and_full() -> None:
    fast = parse_models("regular-fast", "classification")
    full = parse_models("regular-full", "classification")

    assert "gradient_boosting_classifier" not in fast
    assert "gradient_boosting_classifier" in full
    assert parse_models("compact", "classification") == ["svm", "linear_svm", "pls_da"]


def test_optional_boosting_grids_include_regularization_and_sampling_controls() -> None:
    xgb_grid = _optional_boosting_grid("xgboost_classifier")
    assert {"n_estimators", "max_depth", "learning_rate", "subsample", "reg_lambda"} <= set(xgb_grid)
    assert min(xgb_grid["n_estimators"]) <= 50
    assert min(xgb_grid["max_depth"]) <= 2
    assert max(xgb_grid["reg_lambda"]) >= 5.0
    assert len(xgb_grid["n_estimators"]) * len(xgb_grid["max_depth"]) > 4

    lgbm_grid = _optional_boosting_grid("lightgbm_classifier")
    assert {"n_estimators", "num_leaves", "learning_rate", "reg_lambda"} <= set(lgbm_grid)
    assert 7 in lgbm_grid["num_leaves"]

    cat_grid = _optional_boosting_grid("catboost_classifier")
    assert {"iterations", "depth", "learning_rate", "l2_leaf_reg"} <= set(cat_grid)
    assert max(cat_grid["l2_leaf_reg"]) >= 10.0


def test_locked_experimental_model_parameters_allow_empty_grid() -> None:
    params = {
        "embedding_dim": 16,
        "epochs": 80,
        "batch_size": 16,
        "lr": 0.001,
        "metric": "euclidean",
        "device": "auto",
    }

    assert _locked_parameter_grid("proto_spectral_classifier", {}, params) == params

    with pytest.raises(Exception):
        _locked_parameter_grid("linear_svm", {"model__C": [0.1, 1.0]}, {"embedding_dim": 16})


def test_repeated_classifier_comparison_writes_per_model_repeat_tables(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_repeated_split(tmp_path / "split")
    output_dir = tmp_path / "classifier_compare"

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        task_type="classification",
        models="lda,linear_svm",
        mode="repeated_classifier_comparison",
        checkpoint_per_model=True,
        candidate_model_set_source="user_custom",
        overwrite=True,
    )

    assert response["ok"] is True
    metrics = _read_csv(output_dir / "classifier_repeat_metrics.csv")
    header = metrics[0]
    rows = metrics[1:]
    assert len(rows) == 4
    assert "model_method" in header
    assert "accuracy" in header
    assert "balanced_accuracy" in header
    assert "macro_f1" in header
    model_column = header.index("model_method")
    assert {row[model_column] for row in rows} == {"lda", "linear_svm"}
    contract = json.loads((output_dir / "classifier_comparison_contract.json").read_text(encoding="utf-8"))
    assert contract["comparison_mode"] == "per_classifier_repeated_evaluation"
    assert contract["model_selection_enabled"] is False
    assert contract["same_repeats_across_models"] is True
    assert contract["candidate_model_set_source"] == "user_custom"
    summary = _read_csv(output_dir / "classifier_metric_summary.csv")
    assert "accuracy_mean_percent" in summary[0]
    assert "balanced_accuracy_mean_percent" in summary[0]
    assert "macro_f1_mean_percent" in summary[0]
    assert (output_dir / "checkpoints" / "lda" / "repeated_modeling_result.json").exists()
    assert (output_dir / "checkpoints" / "linear_svm" / "repeated_modeling_result.json").exists()


def test_repeated_holdout_locked_best_pipeline_params_apply_to_every_repeat(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_repeated_split(tmp_path / "split")
    best_pipeline = tmp_path / "best_pipeline.json"
    best_pipeline.write_text(
        json.dumps(
            {
                "trial_id": "trial_svm_c1_scale",
                "row": {
                    "model_method": "svm",
                    "params": json.dumps({"modeling": {"C": 1.0, "gamma": "scale"}}),
                },
            }
        ),
        encoding="utf-8",
    )

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model",
        task_type="classification",
        best_pipeline=best_pipeline,
        lock_best_pipeline_params=True,
    )

    assert response["ok"] is True
    contract = json.loads((tmp_path / "model" / "repeated_modeling_result.json").read_text(encoding="utf-8"))
    assert contract["model_selection_mode"] == "locked_from_optimizer_best"
    assert contract["param_search_enabled"] is False
    assert contract["model_params_source"] == "best_pipeline.json"
    for item in contract["iterations"]:
        assert item["model_type"] == "svm"
        assert item["model_parameters"] == {"model__C": 1.0, "model__gamma": "scale"}
        assert item["model_selection_mode"] == "locked_from_optimizer_best"
        assert item["param_search_enabled"] is False
        assert item["model_params_source"] == "best_pipeline.json"
        metrics = json.loads((tmp_path / "model" / item["metrics"]).read_text(encoding="utf-8"))
        assert metrics["model_parameters"] == {"model__C": 1.0, "model__gamma": "scale"}
        assert metrics["param_search_enabled"] is False


def test_regression_modeling_writes_regression_metrics_and_residuals(tmp_path: Path) -> None:
    package_dir = _write_regression_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split", prefix="R")
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="regression", models="plsr,ridge", random_seed=7)

    assert response["ok"] is True
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "rmse" in metrics["test_metrics"]
    assert metrics["selection_strategy"]["test_used_for_selection"] is False
    header = _read_csv(output_dir / "predictions.csv")[0]
    assert "residual" in header
    assert "absolute_error" in header


def test_no_validation_uses_train_cv_not_test_selection(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split", no_val=True)
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="classification", models="random_forest_classifier", cv_folds=2)

    assert response["ok"] is True
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["selection_strategy"]["tuning_split"].startswith("train_cv_")
    assert contract["selection_strategy"]["test_used_for_selection"] is False


def test_missing_model_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "model", task_type="classification")

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "MODEL_TYPE_REQUIRED"
    assert not (tmp_path / "model").exists()


def test_linear_svm_and_lda_are_supported_classification_candidates(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="classification", models="linear_svm,lda", random_seed=7)

    assert response["ok"] is True
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["selection_strategy"]["candidate_models"] == ["linear_svm", "lda"]


def test_extended_classification_and_chemometric_models_are_supported(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "model"

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        task_type="classification",
        models="gaussian_nb,pls_da,extra_trees_classifier",
        random_seed=7,
    )

    assert response["ok"] is True
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["selection_strategy"]["candidate_models"] == ["gaussian_nb", "pls_da", "extra_trees_classifier"]
    assert contract["model_family"] in {"traditional_ml", "chemometrics"}


def test_gpr_regression_writes_uncertainty_outputs(tmp_path: Path) -> None:
    package_dir = _write_regression_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split", prefix="R")
    output_dir = tmp_path / "model"

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, task_type="regression", models="gpr", random_seed=7)

    assert response["ok"] is True
    assert (output_dir / "prediction_std.csv").exists()
    assert (output_dir / "uncertainty_summary.json").exists()
    header = _read_csv(output_dir / "predictions.csv")[0]
    assert "y_pred_std" in header
    assert "lower_95" in header
    contract = json.loads((output_dir / "modeling_contract.json").read_text(encoding="utf-8"))
    assert contract["model_family"] == "traditional_ml"


def test_experimental_model_missing_parameters_need_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model",
        task_type="classification",
        models="cls_former_classifier",
    )

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "MODEL_PARAMETERS_CONFIRMATION_REQUIRED"
    fields = set(response["result"]["confirmation_required"][0]["fields"])
    assert {"epochs", "batch_size", "alpha", "device"} <= fields


def test_experimental_model_auto_confirm_then_checks_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    monkeypatch.setattr(modeling_workflow, "missing_dependencies", lambda models: {"cls_former_classifier": ["torch"]})

    response = model_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "model",
        task_type="classification",
        models="cls_former_classifier",
        auto_confirm_model_defaults=True,
    )

    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["errors"][0]["code"] == "MODEL_DEPENDENCY_MISSING"
    assert response["result"]["missing_dependencies"]["cls_former_classifier"] == ["torch"]


def test_missing_y_blocks_modeling(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package", with_y=False)
    split_contract = _write_split(tmp_path / "split")

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "model", task_type="classification", models="svm")

    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["errors"][0]["code"] == "Y_REQUIRED"


def test_no_test_split_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split", no_test=True)

    response = model_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "model", task_type="classification", models="svm")

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "TEST_SPLIT_REQUIRED"


def test_modeling_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_classification_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-modeling" / "scripts" / "model_spectral_package.py",
            REPO_ROOT / "scripts" / "modeling" / "model_spectral_package.py",
        ]
    ):
        output_dir = tmp_path / f"model_{idx}"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--package-dir",
                str(package_dir),
                "--split-contract",
                str(split_contract),
                "--output-dir",
                str(output_dir),
                "--task-type",
                "classification",
                "--models",
                "random_forest_classifier",
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
        assert payload["result"]["modeling_contract"] == "modeling_contract.json"
        assert (output_dir / "metrics.json").exists()

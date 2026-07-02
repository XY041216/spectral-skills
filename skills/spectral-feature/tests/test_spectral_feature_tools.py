from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from spectral_core.feature.audit import audit_feature_package
from spectral_core.feature.workflow import _recommended_defaults, feature_spectral_package
from spectral_core.preprocess.workflow import preprocess_spectral_package


REPO_ROOT = Path(__file__).resolve().parents[3]
LONG_SPLIT_HEADER = ["split_type", "method", "fold_id", "repeat_id", "role", "sample_index", "sample_id", "label", "group_id"]


def test_deep_recommended_defaults_are_method_specific_and_data_aware() -> None:
    dae = _recommended_defaults("denoising_autoencoder_embedding", n_features=3401, n_train=72)
    cnn = _recommended_defaults("cnn_1d_embedding", n_features=3401, n_train=72)
    resnet = _recommended_defaults("resnet1d_embedding", n_features=3401, n_train=72)
    cls = _recommended_defaults("cls_former_embedding", n_features=3401, n_train=72)
    masked = _recommended_defaults("masked_spectral_autoencoder_embedding", n_features=3401, n_train=72)
    contrastive = _recommended_defaults("contrastive_spectral_embedding", n_features=3401, n_train=72)

    assert {item["n_components"] for item in [dae, cnn, resnet, cls, masked, contrastive]} == {16}
    assert {item["batch_size"] for item in [dae, cnn, resnet, cls, masked, contrastive]} == {16}
    assert dae["noise_std"] == 0.03 and dae["epochs"] == 100
    assert cnn["epochs"] == 80 and cnn["weight_decay"] == 1e-4
    assert resnet["epochs"] == 60
    assert cls["patch_size"] == 16 and cls["weight_decay"] == 1e-4
    assert masked["mask_ratio"] == 0.15
    assert contrastive["temperature"] == 0.2


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200", "1300"],
            [1, 2, 3, 10, 5],
            [2, 2, 4, 11, 6],
            [3, 2, 5, 12, 7],
            [10, 2, 12, 19, 14],
            [11, 2, 13, 20, 15],
            [12, 2, 14, 21, 16],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "nm"], [1, 1000, "nm"], [2, 1100, "nm"], [3, 1200, "nm"], [4, 1300, "nm"]])
    _write_rows(root / "y.csv", [["class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    _write_rows(root / "metadata.csv", [["batch"], ["B1"], ["B1"], ["B1"], ["B2"], ["B2"], ["B2"]])
    contract = {
        "contract_id": "data-feature-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": "metadata.csv"},
        "shape": {"n_samples": 6, "n_features": 5},
        "task_hint": "classification",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_split(root: Path) -> Path:
    _write_rows(
        root / "split_indices.csv",
        [
            ["sample_id", "index", "split"],
            ["S001", 0, "train"],
            ["S002", 1, "train"],
            ["S003", 2, "train"],
            ["S004", 3, "val"],
            ["S005", 4, "test"],
            ["S006", 5, "test"],
        ],
    )
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-test",
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 6, "train": 3, "val": 1, "test": 2},
    }
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_new_holdout_split(root: Path) -> Path:
    ids = [f"S{idx:03d}" for idx in range(1, 7)]
    assignments = {"train": [0, 1, 2], "val": [3], "test": [4, 5]}
    rows = [LONG_SPLIT_HEADER]
    for role, indices in assignments.items():
        for idx in indices:
            rows.append(["holdout", "stratified", "", "", role, idx, ids[idx], "A" if idx < 3 else "B", ""])
    _write_rows(root / "split_indices.csv", rows)
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-new-holdout-test",
        "split_type": "holdout",
        "method": "stratified",
        "indices": assignments,
        "sample_ids": {role: [ids[idx] for idx in indices] for role, indices in assignments.items()},
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 6, "train": 3, "val": 1, "test": 2},
    }
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_cv_split(root: Path) -> Path:
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-cv-test",
        "split_type": "cross_validation",
        "method": "stratified_kfold",
        "n_splits": 3,
        "folds": [{"fold_id": 1, "train_indices": [0, 1, 2, 3], "val_indices": [4, 5]}],
    }
    path = root / "split_contract.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_numeric_package(root: Path, *, task: str = "regression", n_samples: int = 24, n_features: int = 12) -> Path:
    rng_rows = []
    y_rows = []
    for sample_idx in range(n_samples):
        latent = sample_idx / max(n_samples - 1, 1)
        row = [
            latent * (feature_idx + 1)
            + ((sample_idx + feature_idx * 3) % 5) * 0.03
            + (1.5 if task == "classification" and sample_idx >= n_samples // 2 and feature_idx < 3 else 0.0)
            for feature_idx in range(n_features)
        ]
        rng_rows.append(row)
        y_rows.append(["B" if sample_idx >= n_samples // 2 else "A"] if task == "classification" else [2.0 * latent + 0.1 * row[0]])
    names = [str(900 + idx * 10) for idx in range(n_features)]
    _write_rows(root / "X.csv", [names, *rng_rows])
    _write_rows(root / "sample_ids.csv", [["sample_id"], *[[f"S{idx:03d}"] for idx in range(n_samples)]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], *[[idx, names[idx], "nm"] for idx in range(n_features)]])
    _write_rows(root / "y.csv", [["class" if task == "classification" else "target"], *y_rows])
    contract = {
        "contract_id": f"feature-{task}",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv"},
        "shape": {"n_samples": n_samples, "n_features": n_features},
        "task_hint": task,
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_numeric_split(root: Path, *, n_samples: int = 24) -> Path:
    assignments = {
        "train": list(range(0, int(n_samples * 0.7))),
        "val": list(range(int(n_samples * 0.7), int(n_samples * 0.85))),
        "test": list(range(int(n_samples * 0.85), n_samples)),
    }
    contract = {
        "contract_type": "split_contract",
        "contract_id": "numeric-holdout",
        "split_type": "holdout",
        "method": "random",
        "indices": assignments,
        "n_samples": {"total": n_samples, **{role: len(indices) for role, indices in assignments.items()}},
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "split_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _read_X(path: Path) -> list[list[float]]:
    return [[float(value) for value in row] for row in _read_csv(path)[1:]]


def test_none_writes_standard_package_and_state(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="none")

    assert response["ok"] is True
    assert set(path.name for path in output_dir.iterdir()) == {"X.csv", "sample_ids.csv", "band_axis.csv", "y.csv", "metadata.csv", "data_contract.json", "feature_state.json", "feature_contract.json"}
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["processing_stage"] == "feature"
    assert contract["feature_summary"]["methods"] == ["none"]
    assert contract["feature_summary"]["fit_scope"] == "train_only"
    assert contract["shape"] == {"n_samples": 6, "n_features": 5}
    assert _read_csv(output_dir / "sample_ids.csv")[1][0] == "S001"
    feature_contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert feature_contract["execution_mode"] == "holdout"
    assert feature_contract["output_package"].endswith("data_contract.json")


def test_preprocess_contract_with_explicit_relative_split_resolves_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    preprocess_dir = tmp_path / "preprocess"
    preprocess_response = preprocess_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=preprocess_dir,
        methods="snv",
        overwrite=True,
    )
    assert preprocess_response["ok"] is True

    monkeypatch.chdir(tmp_path)
    response = feature_spectral_package(
        preprocess_contract=Path("preprocess") / "preprocess_contract.json",
        split_contract=Path("split") / "split_contract.json",
        output_dir=Path("feature_from_relative_split"),
        method="pca",
        n_components=2,
        overwrite=True,
    )

    assert response["ok"] is True
    contract = json.loads((tmp_path / "feature_from_relative_split" / "feature_contract.json").read_text(encoding="utf-8"))
    assert Path(contract["split_contract"]).resolve() == split_contract.resolve()
    assert Path(contract["resolved_paths"]["split_contract"]).resolve() == split_contract.resolve()


def test_new_holdout_split_contract_indices_are_supported(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="variance_threshold")

    assert response["ok"] is True
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert state["split"]["split_type"] == "holdout"
    assert contract["split"]["method"] == "stratified"


def test_cv_split_contract_runs_partition_wise(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_cv_split(tmp_path / "split")

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "feature", method="variance_threshold")

    assert response["ok"] is True
    contract = json.loads((tmp_path / "feature" / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "cross_validation"
    assert contract["execution_mode"] == "fold_wise"
    assert contract["leakage_guard"]["fit_on"] == "train_only_for_each_partition"
    assert contract["iterations"][0]["params_path"] == "iterations/fold_001/feature_params.json"


def test_preprocess_contract_runs_partition_wise_pca(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_cv_split(tmp_path / "split")
    preprocess_dir = tmp_path / "preprocess"

    preprocess_response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=preprocess_dir, methods="snv")
    assert preprocess_response["ok"] is True

    response = feature_spectral_package(
        preprocess_contract=preprocess_dir / "preprocess_contract.json",
        output_dir=tmp_path / "feature",
        method="pca",
        n_components=2,
    )

    assert response["ok"] is True
    contract = json.loads((tmp_path / "feature" / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["input_package"].endswith("preprocess_contract.json")
    assert contract["split_type"] == "cross_validation"
    iteration = contract["iterations"][0]
    assert iteration["role_files"]["X_train"].endswith("X_train_features.csv")
    assert (tmp_path / "feature" / iteration["role_files"]["X_train"]).exists()
    params = json.loads((tmp_path / "feature" / iteration["params_path"]).read_text(encoding="utf-8"))
    assert params["method"] == "pca"
    assert params["output_n_features"] == 2


def test_holdout_preprocess_contract_runs_standard_feature_output(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    preprocess_dir = tmp_path / "preprocess"
    preprocess_response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=preprocess_dir, methods="snv")
    assert preprocess_response["ok"] is True

    response = feature_spectral_package(
        preprocess_contract=preprocess_dir / "preprocess_contract.json",
        output_dir=tmp_path / "feature",
        method="pls_latent_variables",
        n_components=2,
        task_type="classification",
    )

    assert response["ok"] is True
    assert response["result"]["feature_contract"] == "feature_contract.json"
    feature_contract = json.loads((tmp_path / "feature" / "feature_contract.json").read_text(encoding="utf-8"))
    assert feature_contract["execution_mode"] == "holdout"
    assert feature_contract["upstream_preprocess"]["methods"] == ["snv"]
    assert feature_contract["upstream_preprocess"]["preprocess_contract"].endswith("preprocess_contract.json")
    assert feature_contract["leakage_guard"]["val_test_y_used_for_fit"] is False
    assert _read_csv(tmp_path / "feature" / "X.csv")[0] == ["PLS_LV_001", "PLS_LV_002"]


def test_variance_threshold_fits_train_only_and_drops_constant_train_band(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="variance_threshold")
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))

    assert response["ok"] is True
    assert _read_csv(output_dir / "X.csv")[0] == ["900", "1100", "1200", "1300"]
    assert [row[1] for row in _read_csv(output_dir / "band_axis.csv")[1:]] == ["900", "1100", "1200", "1300"]
    fitted = state["method_states"][0]["fitted"]
    assert fitted["selected_band_indices"] == [0, 2, 3, 4]
    assert fitted["fit_sample_count"] == 3


def test_pca_fits_train_only_and_updates_axis(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="pca", n_components=2)
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    X = _read_X(output_dir / "X.csv")

    assert response["ok"] is True
    assert _read_csv(output_dir / "X.csv")[0] == ["PC1", "PC2"]
    assert [row[1] for row in _read_csv(output_dir / "band_axis.csv")[1:]] == ["PC1", "PC2"]
    assert len(X) == 6
    assert len(X[0]) == 2
    fitted = state["method_states"][0]["fitted"]
    assert fitted["mean"] == [2, 2, 4, 11, 6]
    assert fitted["fit_sample_count"] == 3
    assert len(fitted["components"]) == 2
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["shape"]["n_features"] == 2
    assert contract["n_features"] == 2
    assert contract["band_axis"]["count"] == 2
    assert contract["band_axis"]["type"] == "derived_feature_axis"
    assert contract["spectral"]["n_bands"] == 2
    assert contract["spectral"]["band_axis_labels"] == ["PC1", "PC2"]
    assert contract["feature"]["input_n_features"] == 5
    assert contract["feature"]["output_n_features"] == 2
    assert contract["source_spectral"]["original_n_bands"] == 5
    assert state["input_n_features"] == 5
    assert state["output_n_features"] == 2


def test_feature_audit_repairs_stale_contract_counts(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="pca", n_components=2)
    assert response["ok"] is True
    contract_path = output_dir / "data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["n_features"] = 5
    contract["band_axis"]["count"] = 5
    contract["spectral"]["n_bands"] = 5
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    audit = audit_feature_package(output_dir, repair=True)

    assert audit["ok"] is True
    assert audit["repaired"] is True
    repaired = json.loads(contract_path.read_text(encoding="utf-8"))
    assert repaired["n_features"] == 2
    assert repaired["band_axis"]["count"] == 2
    assert repaired["spectral"]["n_bands"] == 2
    assert repaired["source_spectral"]["original_n_bands"] == 5


def test_pca_explained_variance_retention(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "feature"

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, method="pca", explained_variance=0.95)

    assert response["ok"] is True
    assert json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))["shape"]["n_features"] >= 1
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    assert state["method_states"][0]["parameters"] == {"explained_variance": 0.95}


def test_extended_projection_and_signal_methods_execute(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)

    kpca_dir = tmp_path / "kpca"
    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=kpca_dir,
        method="kernel_pca",
        n_components=2,
        random_state=7,
    )
    assert response["ok"] is True
    assert _read_csv(kpca_dir / "X.csv")[0] == ["KPCA_001", "KPCA_002"]
    state = json.loads((kpca_dir / "feature_state.json").read_text(encoding="utf-8"))
    assert state["method"] == "kernel_pca"
    assert state["output_features"]["feature_mode"] == "modeling_embedding"
    assert state["intended_use"] == "modeling"
    assert state["allowed_for_optimizer_default"] is False
    assert json.loads((kpca_dir / "data_contract.json").read_text(encoding="utf-8"))["band_axis"]["type"] == "derived_feature_axis"

    dct_dir = tmp_path / "dct"
    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=dct_dir,
        method="dct_features",
        n_components=3,
    )
    assert response["ok"] is True
    assert _read_csv(dct_dir / "X.csv")[0] == ["DCT_001", "DCT_002", "DCT_003"]
    state = json.loads((dct_dir / "feature_state.json").read_text(encoding="utf-8"))
    assert state["method_family"] == "deterministic_signal_transform"
    assert state["output_features"]["feature_mode"] == "signal_transform_features"
    assert state["out_of_sample_transform"] == "per_sample_deterministic"


def test_nmf_requires_nonnegative_input_and_records_audit(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)
    rows = _read_csv(package_dir / "X.csv")
    rows[1][0] = "-0.25"
    _write_rows(package_dir / "X.csv", rows)

    blocked = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "nmf_blocked",
        method="nmf",
        n_components=2,
        random_state=42,
    )
    assert blocked["ok"] is False
    assert blocked["result"]["status"] == "blocked"
    assert blocked["errors"][0]["code"] == "NMF_NONNEGATIVE_REQUIRED"
    assert blocked["errors"][0]["details"]["nonnegative_check"] == "failed"
    assert not (tmp_path / "nmf_blocked").exists()

    rows[1][0] = "0.25"
    _write_rows(package_dir / "X.csv", rows)
    output_dir = tmp_path / "nmf_ready"
    ready = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        method="nmf",
        n_components=2,
        random_state=42,
    )
    assert ready["ok"] is True
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert state["requires_nonnegative_X"] is True
    assert state["nonnegative_check"] == "passed"
    assert state["input_min_value"] >= 0
    assert contract["nonnegative_check"] == "passed"
    assert (output_dir / "feature_manifest.csv").exists()


def test_lda_projection_clips_to_class_limit_and_records_supervision(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=24, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=24)
    labels = [["class"], *[["A", "B", "C", "D"][idx % 4] for idx in range(24)]]
    _write_rows(package_dir / "y.csv", labels)
    output_dir = tmp_path / "lda"

    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        method="lda_projection",
        n_components=10,
    )
    assert response["ok"] is True
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    method_state = state["method_states"][0]
    assert state["output_features"]["feature_mode"] == "supervised_modeling_embedding"
    assert method_state["parameters"]["requested_n_components"] == 10
    assert method_state["parameters"]["effective_n_components"] == 3
    assert method_state["parameters"]["component_limit"] == "n_classes - 1"
    assert method_state["fitted"]["supervised_y_used"] is True
    assert state["fit_scope"] == "train_only"
    assert state["val_test_y_used_for_fit"] is False


def test_sparse_pca_convergence_is_in_state_contract_and_manifest(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)
    output_dir = tmp_path / "sparse_pca"

    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        method="sparse_pca",
        n_components=2,
        random_state=42,
    )
    assert response["ok"] is True
    state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert set(state["convergence"]) >= {"converged", "n_iter", "max_iter", "random_seed", "warning"}
    assert state["convergence"]["random_seed"] == 42
    assert contract["convergence"] == state["convergence"]
    manifest = list(csv.DictReader((output_dir / "feature_manifest.csv").open(encoding="utf-8")))
    assert manifest[0]["method"] == "sparse_pca"
    assert manifest[0]["random_seed"] == "42"


def test_visual_and_experimental_embeddings_are_gated(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)

    tsne = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "tsne",
        method="tsne_embedding",
        n_components=2,
    )
    assert tsne["ok"] is False
    assert tsne["result"]["status"] == "needs_confirmation"
    assert tsne["errors"][0]["code"] == "TSNE_VISUALIZATION_ONLY_REQUIRES_UNSPLIT"
    assert not (tmp_path / "tsne").exists()

    gated = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "contrastive",
        method="attention_pooling",
    )
    assert gated["ok"] is False
    assert gated["result"]["status"] == "needs_confirmation"
    assert gated["errors"][0]["code"] == "FEATURE_METHOD_EXPERIMENTAL_GATED"
    assert "dedicated training protocol" in gated["errors"][0]["message"]

    discovery_dir = tmp_path / "tsne_discovery"
    discovery = feature_spectral_package(
        package_dir=package_dir,
        output_dir=discovery_dir,
        method="tsne_embedding",
        n_components=2,
        random_state=42,
    )
    assert discovery["ok"] is True
    contract = json.loads((discovery_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["intended_use"] == "visualization"
    assert contract["out_of_sample_transform"] == "unsupported"
    assert contract["allowed_for_optimizer_default"] is False
    assert contract["handoff"]["spectral_modeling"]["blocked"] is True
    assert contract["handoff"]["spectral_report"]["ready"] is True


def test_deep_embeddings_require_confirmation(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)
    blocked_dir = tmp_path / "blocked_deep"
    blocked = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=blocked_dir,
        method="autoencoder_embedding",
        n_components=2,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        random_state=42,
    )
    assert blocked["ok"] is False
    assert blocked["result"]["status"] == "needs_confirmation"
    assert blocked["errors"][0]["code"] == "DEEP_EMBEDDING_TRAINING_CONFIRMATION_REQUIRED"
    assert not blocked_dir.exists()


def test_deep_embeddings_execute_with_audits(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="deep embedding execution requires optional PyTorch")
    package_dir = _write_numeric_package(tmp_path / "package", task="classification", n_samples=18, n_features=8)
    split_contract = _write_numeric_split(tmp_path / "split", n_samples=18)
    methods = [
        "autoencoder_embedding",
        "denoising_autoencoder_embedding",
        "cnn_1d_embedding",
        "spectral_transformer_embedding",
        "resnet1d_embedding",
        "masked_spectral_autoencoder_embedding",
        "contrastive_spectral_embedding",
    ]
    prefixes = {
        "autoencoder_embedding": "AE",
        "denoising_autoencoder_embedding": "DAE",
        "cnn_1d_embedding": "CNN1D",
        "cls_former_embedding": "CLS",
        "resnet1d_embedding": "RESNET1D",
        "masked_spectral_autoencoder_embedding": "MAE",
        "contrastive_spectral_embedding": "CONTRAST",
    }
    for requested_method in methods:
        output_dir = tmp_path / requested_method
        response = feature_spectral_package(
            package_dir=package_dir,
            split_contract=split_contract,
            output_dir=output_dir,
            method=requested_method,
            n_components=2,
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0,
            noise_std=0.02,
            mask_ratio=0.2,
            temperature=0.2,
            patch_size=4,
            random_state=42,
            device="cpu",
            confirm_deep_embedding_training=True,
        )
        assert response["ok"] is True, (requested_method, response)
        state = json.loads((output_dir / "feature_state.json").read_text(encoding="utf-8"))
        canonical = "cls_former_embedding" if requested_method == "spectral_transformer_embedding" else requested_method
        assert state["method"] == canonical
        assert state["output_features"]["feature_mode"] == "modeling_embedding"
        assert state["deep_training_confirmation"]["status"] == "confirmed"
        assert state["training_audit"]["epochs_completed"] == 1
        assert state["training_audit"]["loss_finite"] is True
        assert (output_dir / "training_trace.csv").exists()
        assert (output_dir / "feature_manifest.csv").exists()
        assert _read_csv(output_dir / "X.csv")[0] == [f"{prefixes[canonical]}_001", f"{prefixes[canonical]}_002"]
        with (output_dir / "feature_transformer.pkl").open("rb") as handle:
            transformer = pickle.load(handle)
        assert transformer.transform(_read_X(package_dir / "X.csv")).shape == (18, 2)


def test_band_range_and_indices_select_expected_features(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    range_dir = tmp_path / "feature_range"
    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=range_dir, method="select_by_band_range", band_min=1000, band_max=1200)
    assert response["ok"] is True
    assert _read_csv(range_dir / "X.csv")[0] == ["1000", "1100", "1200"]

    index_dir = tmp_path / "feature_index"
    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=index_dir, method="select_by_band_indices", band_indices="2:4", index_base=1)
    assert response["ok"] is True
    assert _read_csv(index_dir / "X.csv")[0] == ["1000", "1100", "1200"]


def test_train_fit_method_without_split_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    response = feature_spectral_package(package_dir=package_dir, output_dir=tmp_path / "feature", method="pca", n_components=2)

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["confirmation_required"][0]["field"] == "split_contract"
    assert response["errors"][0]["code"] == "SPLIT_CONTRACT_REQUIRED_FOR_FIT"
    assert not (tmp_path / "feature").exists()


def test_pca_without_retention_uses_safe_default(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "feature", method="pca")

    assert response["ok"] is True
    state = json.loads((tmp_path / "feature" / "feature_state.json").read_text(encoding="utf-8"))
    assert state["method_states"][0]["parameters"]["n_components"] == 2
    assert any(item["code"] == "N_COMPONENTS_CLIPPED" for item in state["method_states"][0]["warnings"])


def test_pls_latent_variables_and_vip_write_expected_artifacts(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="regression")
    split_contract = _write_numeric_split(tmp_path / "split")

    pls_dir = tmp_path / "pls"
    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=pls_dir,
        method="pls_lv",
        n_components=3,
    )
    assert response["ok"] is True
    assert _read_csv(pls_dir / "X.csv")[0] == ["PLS_LV_001", "PLS_LV_002", "PLS_LV_003"]
    assert (pls_dir / "components.csv").exists()
    state = json.loads((pls_dir / "feature_state.json").read_text(encoding="utf-8"))
    assert state["output_features"]["feature_mode"] == "projection"
    assert state["leakage_check"]["fit_on_train_only"] is True
    assert state["supervised"] is True
    assert state["y_used"] is True
    assert state["fit_scope"] == "train_only"
    assert state["transform_scope_roles"] == ["train", "val", "test"]
    assert state["val_test_y_used_for_fit"] is False
    assert state["leakage_guard"] == "passed"
    feature_contract = json.loads((pls_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert feature_contract["supervised"] is True
    assert feature_contract["leakage_guard"]["status"] == "passed"

    vip_dir = tmp_path / "vip"
    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=vip_dir,
        method="pls_vip",
        n_components=3,
        top_k=4,
    )
    assert response["ok"] is True
    assert len(_read_csv(vip_dir / "X.csv")[0]) == 4
    assert (vip_dir / "selected_features.csv").exists()
    assert (vip_dir / "feature_scores.csv").exists()


def test_select_k_best_aliases_and_task_guards(tmp_path: Path) -> None:
    class_package = _write_numeric_package(tmp_path / "class_package", task="classification")
    split_contract = _write_numeric_split(tmp_path / "split")

    response = feature_spectral_package(
        package_dir=class_package,
        split_contract=split_contract,
        output_dir=tmp_path / "anova",
        method="anova",
        top_k=5,
    )
    assert response["ok"] is True
    assert len(_read_csv(tmp_path / "anova" / "X.csv")[0]) == 5
    state = json.loads((tmp_path / "anova" / "feature_state.json").read_text(encoding="utf-8"))
    assert state["params"]["score_func"] == "f_classif"
    assert state["task_type"] == "classification"

    mismatch = feature_spectral_package(
        package_dir=class_package,
        split_contract=split_contract,
        output_dir=tmp_path / "bad",
        method="f_regression",
        top_k=5,
    )
    assert mismatch["ok"] is False
    assert mismatch["errors"][0]["code"] == "F_REGRESSION_TASK_MISMATCH"


def test_interval_spa_cars_and_uve_execute_with_compact_artifacts(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="regression")
    split_contract = _write_numeric_split(tmp_path / "split")

    interval = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "interval",
        method="ipls",
        n_intervals=4,
        n_components=2,
        cv=3,
    )
    assert interval["ok"] is True
    assert (tmp_path / "interval" / "selected_intervals.csv").exists()

    spa = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "spa",
        method="spa",
        top_k=4,
    )
    assert spa["ok"] is True
    assert len(_read_csv(tmp_path / "spa" / "X.csv")[0]) == 4

    cars = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "cars",
        method="cars_pls",
        n_components=2,
        n_runs=4,
        sample_ratio=0.8,
        cv=3,
        random_state=7,
    )
    assert cars["ok"] is True
    assert (tmp_path / "cars" / "selection_trace.csv").exists()

    uve = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "uve",
        method="mc_uve",
        n_components=2,
        n_runs=4,
        top_k=4,
        random_state=7,
    )
    assert uve["ok"] is True
    assert (tmp_path / "uve" / "stability_scores.csv").exists()


def test_supervised_method_requires_y_and_split(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    (package_dir / "y.csv").unlink()
    contract = json.loads((package_dir / "data_contract.json").read_text(encoding="utf-8"))
    contract["files"]["y"] = None
    (package_dir / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    missing_split = feature_spectral_package(package_dir=package_dir, output_dir=tmp_path / "no_split", method="vip")
    assert missing_split["ok"] is False
    assert missing_split["result"]["status"] == "needs_confirmation"

    split_contract = _write_split(tmp_path / "split")
    missing_y = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "no_y",
        method="vip",
        n_components=2,
        top_k=2,
    )
    assert missing_y["ok"] is False
    assert missing_y["errors"][0]["code"] == "Y_REQUIRED"


def test_feature_config_supplies_method_and_params(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification")
    split_contract = _write_numeric_split(tmp_path / "split")
    config = tmp_path / "feature.json"
    config.write_text(json.dumps({"method": "select_k_best", "params": {"top_k": 3}}), encoding="utf-8")

    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "feature",
        feature_config=config,
    )
    assert response["ok"] is True
    assert len(_read_csv(tmp_path / "feature" / "X.csv")[0]) == 3


def test_key_parameters_require_confirmation_without_auto_defaults(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification")
    split_contract = _write_numeric_split(tmp_path / "split")

    for method, expected_fields in [
        ("pls_latent_variables", {"n_components"}),
        ("vip", {"selection_rule", "n_components"}),
        ("select_k_best", {"top_k"}),
        ("interval_pls", {"n_intervals", "n_components", "cv"}),
        ("spa", {"top_k"}),
        ("cars", {"n_components", "n_runs", "sample_ratio", "cv", "random_state"}),
        ("uve", {"n_components", "n_runs", "selection_rule", "random_state"}),
        ("mcuve", {"n_components", "n_runs", "selection_rule", "random_state"}),
    ]:
        response = feature_spectral_package(
            package_dir=package_dir,
            split_contract=split_contract,
            output_dir=tmp_path / method,
            method=method,
        )
        assert response["ok"] is False
        assert response["result"]["status"] == "needs_confirmation"
        assert response["errors"][0]["code"] == "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED"
        assert {item["field"] for item in response["result"]["confirmation_required"]} == expected_fields
        assert not (tmp_path / method).exists()


def test_auto_confirm_defaults_records_parameter_provenance(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification")
    split_contract = _write_numeric_split(tmp_path / "split")

    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "feature",
        method="vip",
        auto_confirm_feature_defaults=True,
    )
    assert response["ok"] is True
    state = json.loads((tmp_path / "feature" / "feature_state.json").read_text(encoding="utf-8"))
    assert state["params"]["top_k"] == 12
    assert state["params"]["n_components"] == 10
    assert state["defaulted_params"] == ["n_components", "top_k"]
    assert state["parameter_sources"]["top_k"] == "defaulted_auto_confirmed"
    assert state["defaults_confirmed"] is True


def test_uve_partial_parameters_report_only_remaining_bundle_items(tmp_path: Path) -> None:
    package_dir = _write_numeric_package(tmp_path / "package", task="classification")
    split_contract = _write_numeric_split(tmp_path / "split")

    response = feature_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=tmp_path / "uve",
        method="uve",
        n_runs=50,
        top_k=50,
        random_state=42,
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED"
    assert [item["field"] for item in response["result"]["confirmation_required"]] == ["n_components"]
    assert not (tmp_path / "uve").exists()


def test_incomplete_split_blocks_output(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    rows = _read_csv(split_contract.parent / "split_indices.csv")
    _write_rows(split_contract.parent / "split_indices.csv", rows[:-1])

    response = feature_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "feature", method="variance_threshold")

    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["errors"][0]["code"] == "SPLIT_INCOMPLETE"
    assert not (tmp_path / "feature").exists()


def test_feature_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-feature" / "scripts" / "feature_spectral_package.py",
            REPO_ROOT / "scripts" / "feature" / "feature_spectral_package.py",
        ]
    ):
        output_dir = tmp_path / f"feature_{idx}"
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
                "--method",
                "select_by_band_range",
                "--band-min",
                "900",
                "--band-max",
                "1100",
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
        assert payload["ok"] is True
        assert payload["result"]["data_contract"] == "data_contract.json"
        assert (output_dir / "feature_state.json").exists()

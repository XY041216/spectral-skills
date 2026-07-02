from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from spectral_core.splitter.workflow import split_spectral_package


REPO_ROOT = Path(__file__).resolve().parents[3]
LONG_SPLIT_HEADER = ["split_type", "method", "fold_id", "repeat_id", "role", "sample_index", "sample_id", "label", "group_id"]


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_package(
    root: Path,
    *,
    labels: list[str] | None = None,
    task_hint: str = "classification",
    metadata: list[dict[str, str]] | None = None,
) -> Path:
    n = len(labels) if labels is not None else 10
    _write_rows(root / "X.csv", [["900", "1000", "1100"], *[[idx, idx + 0.1, idx + 0.2] for idx in range(n)]])
    _write_rows(root / "sample_ids.csv", [["sample_id"], *[[f"S{idx + 1:03d}"] for idx in range(n)]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"]])
    files = {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": None, "metadata": None}
    if labels is not None:
        _write_rows(root / "y.csv", [["class"], *[[label] for label in labels]])
        files["y"] = "y.csv"
    if metadata is not None:
        header = sorted({key for row in metadata for key in row})
        _write_rows(root / "metadata.csv", [header, *[[row.get(column, "") for column in header] for row in metadata]])
        files["metadata"] = "metadata.csv"
    contract = {
        "contract_id": "data-test",
        "status": "ready",
        "files": files,
        "shape": {"n_samples": n, "n_features": 3},
        "task_hint": task_hint,
        "label_status": "present" if labels is not None else "absent",
        "metadata_status": "absent",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_regression_package(root: Path, *, n: int = 12, metadata: list[dict[str, str]] | None = None) -> Path:
    labels = [str(float(idx)) for idx in range(n)]
    return _write_package(root, labels=labels, task_hint="regression", metadata=metadata)


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def test_random_split_writes_contract_without_copying_data(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="random", ratio="8:2", random_seed=42)
    assert response["ok"] is True
    assert set(path.name for path in output_dir.iterdir()) == {"split_indices.csv", "split_contract.json", "split_summary.json"}
    assert not (output_dir / "X.csv").exists()

    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["contract_type"] == "split_contract"
    assert contract["method"] == "random"
    assert contract["random_seed"] == 42
    assert contract["n_samples"] == {"total": 10, "train": 8, "val": 0, "test": 2}
    assert contract["split_files"]["split_indices"] == "split_indices.csv"
    assert contract["handoff_ready"] is True
    rows = _read_csv(output_dir / "split_indices.csv")
    assert rows[0] == LONG_SPLIT_HEADER
    assert len(rows) == 11
    assert len({row[5] for row in rows[1:]}) == 10


def test_splitter_accepts_qc_cleaned_standard_package(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "cleaned_package", labels=["A"] * 5 + ["B"] * 5, task_hint="classification")
    contract_path = package_dir / "data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "processing_stage": "qc_cleaned",
            "parent_contract": "reader_package/data_contract.json",
            "qc_summary": {
                "source_mode": "clean",
                "removed_sample_count": 2,
                "cleaning_log": "qc_cleaning_log.json",
            },
        }
    )
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="stratified", ratio="8:2", random_seed=42)

    assert response["ok"] is True
    split_contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert split_contract["input_contract"].endswith("cleaned_package\\data_contract.json") or split_contract["input_contract"].endswith("cleaned_package/data_contract.json")
    assert split_contract["method"] == "stratified"


def test_random_split_is_reproducible(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    first = tmp_path / "split1"
    second = tmp_path / "split2"
    split_spectral_package(package_dir=package_dir, output_dir=first, method="random", ratio="7:3", random_seed=7)
    split_spectral_package(package_dir=package_dir, output_dir=second, method="random", ratio="7:3", random_seed=7)
    assert (first / "split_indices.csv").read_text(encoding="utf-8") == (second / "split_indices.csv").read_text(encoding="utf-8")


def test_three_way_ratio_and_custom_float_ratios_are_supported(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    ratio_output = tmp_path / "split_ratio"
    custom_output = tmp_path / "split_custom"

    ratio_response = split_spectral_package(package_dir=package_dir, output_dir=ratio_output, method="random", ratio="7:2:1", random_seed=11)
    custom_response = split_spectral_package(
        package_dir=package_dir,
        output_dir=custom_output,
        method="random",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=11,
    )

    assert ratio_response["ok"] is True
    assert ratio_response["result"]["shape"] == {"total": 10, "train": 7, "val": 2, "test": 1}
    assert custom_response["ok"] is True
    assert custom_response["result"]["shape"] == {"total": 10, "train": 6, "val": 2, "test": 2}


def test_incomplete_three_way_ratio_requires_confirmation_before_writing(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    output_dir = tmp_path / "split"

    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="random", ratio="6:2:", random_seed=42)

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "SPLIT_RATIO_CONFIRMATION_REQUIRED"
    assert response["errors"][0]["details"]["recommended_interpretation"] == "train:val:test = 6:2:2"
    assert not output_dir.exists()


def test_confirmed_incomplete_three_way_ratio_is_completed_as_622(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")

    response = split_spectral_package(
        package_dir=package_dir,
        output_dir=tmp_path / "split",
        method="random",
        ratio="6:2:",
        confirm_incomplete_ratio=True,
        random_seed=42,
    )

    assert response["ok"] is True
    assert response["result"]["shape"] == {"total": 10, "train": 6, "val": 2, "test": 2}


def test_chinese_colon_ratio_is_supported(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="random", ratio="8：2")
    assert response["ok"] is True
    assert response["result"]["shape"] == {"total": 10, "train": 8, "val": 0, "test": 2}


def test_stratified_split_preserves_class_presence(tmp_path: Path) -> None:
    labels = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    package_dir = _write_package(tmp_path / "package", labels=labels, task_hint="classification")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="stratified", ratio="6:2:2", random_seed=3)
    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    distribution = contract["label_distribution"]
    assert distribution["enabled"] is True
    assert distribution["before"] == {"A": 6, "B": 6, "C": 6}
    for split_name in ["train", "val", "test"]:
        assert set(distribution["after"][split_name]) == {"A", "B", "C"}
    assert contract["n_samples"]["total"] == 18
    assert contract["n_samples"]["train"] + contract["n_samples"]["val"] + contract["n_samples"]["test"] == 18
    assert contract["n_samples"]["val"] >= 3
    assert contract["n_samples"]["test"] >= 3


def test_predefined_split_from_metadata_column(tmp_path: Path) -> None:
    metadata = [{"split": value} for value in ["train"] * 6 + ["val"] * 2 + ["test"] * 2]
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 5 + ["B"] * 5, metadata=metadata)
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="predefined_split")

    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "holdout"
    assert contract["method"] == "predefined_split"
    assert contract["n_samples"] == {"total": 10, "train": 6, "val": 2, "test": 2}
    rows = _read_csv(output_dir / "split_indices.csv")
    assert rows[0] == LONG_SPLIT_HEADER


def test_predefined_split_from_external_split_indices(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    external = tmp_path / "external_split.csv"
    _write_rows(external, [["sample_id", "split"], *[[f"S{idx + 1:03d}", "train" if idx < 8 else "test"] for idx in range(10)]])
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="predefined_split", split_indices_file=external)

    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["sample_ids"]["train"] == [f"S{idx + 1:03d}" for idx in range(8)]
    assert contract["sample_ids"]["test"] == ["S009", "S010"]


def test_predefined_split_duplicate_external_assignment_blocks(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    external = tmp_path / "external_split.csv"
    _write_rows(
        external,
        [
            ["sample_id", "split"],
            ["S001", "train"],
            ["S001", "test"],
            *[[f"S{idx + 1:03d}", "train" if idx < 8 else "test"] for idx in range(1, 10)],
        ],
    )
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="predefined_split", split_indices_file=external)

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "PREDEFINED_SPLIT_DUPLICATE_ASSIGNMENT"


def test_kfold_and_stratified_kfold_write_folds(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 6 + ["B"] * 6, task_hint="classification")
    kfold_output = tmp_path / "kfold"
    strat_output = tmp_path / "stratified_kfold"

    kfold_response = split_spectral_package(package_dir=package_dir, output_dir=kfold_output, method="kfold", n_splits=4, random_seed=5)
    strat_response = split_spectral_package(package_dir=package_dir, output_dir=strat_output, method="stratified_kfold", n_splits=3, random_seed=5)

    assert kfold_response["ok"] is True
    assert strat_response["ok"] is True
    kfold_contract = json.loads((kfold_output / "split_contract.json").read_text(encoding="utf-8"))
    strat_contract = json.loads((strat_output / "split_contract.json").read_text(encoding="utf-8"))
    assert kfold_contract["split_type"] == "cross_validation"
    assert len(kfold_contract["folds"]) == 4
    assert len(strat_contract["folds"]) == 3
    assert "fold_size_summary" in strat_contract["diagnostics"]
    assert _read_csv(kfold_output / "split_indices.csv")[0] == LONG_SPLIT_HEADER


def test_leave_one_out_writes_one_fold_per_sample(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="leave_one_out")

    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "cross_validation"
    assert len(contract["folds"]) == 10
    assert all(len(fold["val_indices"]) == 1 for fold in contract["folds"])


def test_monte_carlo_cv_writes_repeats_with_default_ratio(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 6 + ["B"] * 6, task_hint="classification")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="stratified_monte_carlo_cv", n_repeats=5, random_seed=42)

    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["split_type"] == "repeated_holdout"
    assert contract["ratios"] == {"train": 0.7, "val": 0.0, "test": 0.3}
    assert len(contract["repeats"]) == 5
    assert _read_csv(output_dir / "split_indices.csv")[0] == LONG_SPLIT_HEADER


def test_kennard_stone_and_spxy_are_supported_for_representative_splits(tmp_path: Path) -> None:
    ks_package = _write_regression_package(tmp_path / "ks_package", n=12)
    spxy_package = _write_regression_package(tmp_path / "spxy_package", n=12)
    ks_output = tmp_path / "ks_split"
    spxy_output = tmp_path / "spxy_split"

    ks_response = split_spectral_package(package_dir=ks_package, output_dir=ks_output, method="kennard_stone", ratio="8:2")
    spxy_response = split_spectral_package(package_dir=spxy_package, output_dir=spxy_output, method="spxy", ratio="8:2")

    assert ks_response["ok"] is True
    assert spxy_response["ok"] is True
    ks_contract = json.loads((ks_output / "split_contract.json").read_text(encoding="utf-8"))
    spxy_contract = json.loads((spxy_output / "split_contract.json").read_text(encoding="utf-8"))
    assert ks_contract["method"] == "kennard_stone"
    assert spxy_contract["method"] == "spxy"
    assert "x_space_coverage" in ks_contract["diagnostics"]
    assert ks_contract["diagnostics"]["distance"]["x_metric"] == "euclidean"
    assert ks_contract["diagnostics"]["distance"]["x_scaling"] == "standardize"
    assert spxy_contract["diagnostics"]["distance"]["y_scaling"] == "minmax"
    assert spxy_contract["diagnostics"]["distance"]["combine_rule"] == "normalized_sum"


def test_spxy_blocks_classification_labels(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 5 + ["B"] * 5, task_hint="classification")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="spxy", ratio="8:2")

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "SPXY_REGRESSION_ONLY"


def test_regression_stratified_and_group_aware_splits(tmp_path: Path) -> None:
    metadata = [{"batch": f"B{idx // 2}"} for idx in range(12)]
    reg_package = _write_regression_package(tmp_path / "reg_package", n=12, metadata=metadata)
    group_package = _write_package(tmp_path / "group_package", labels=["A"] * 6 + ["B"] * 6, metadata=metadata)
    reg_output = tmp_path / "reg_split"
    group_output = tmp_path / "group_split"

    reg_response = split_spectral_package(package_dir=reg_package, output_dir=reg_output, method="regression_stratified", ratio="6:2:2", n_bins=3)
    group_response = split_spectral_package(package_dir=group_package, output_dir=group_output, method="group_aware", ratio="6:2:2", group_column="batch")

    assert reg_response["ok"] is True
    assert group_response["ok"] is True
    reg_contract = json.loads((reg_output / "split_contract.json").read_text(encoding="utf-8"))
    group_contract = json.loads((group_output / "split_contract.json").read_text(encoding="utf-8"))
    assert "regression_target_summary" in reg_contract["diagnostics"]
    assert reg_contract["diagnostics"]["binning"]["effective_n_bins"] <= 3
    assert group_contract["diagnostics"]["group_leakage_check"]["leakage_group_count"] == 0
    assert group_contract["diagnostics"]["group_column"] == "batch"
    assert any(row[-1] for row in _read_csv(group_output / "split_indices.csv")[1:])


def test_explicit_random_split_on_classification_warns_but_writes(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 5 + ["B"] * 5, task_hint="classification")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="random", ratio="8:2")

    assert response["ok"] is True
    assert response["warnings"][0]["code"] == "CLASSIFICATION_RANDOM_SPLIT"
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["method"] == "random"
    assert contract["label_distribution"]["enabled"] is False


def test_auto_classification_confirmed_uses_stratified(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 5 + ["B"] * 5, task_hint="classification")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="auto", ratio="8:2", confirm_stratified=True)

    assert response["ok"] is True
    contract = json.loads((output_dir / "split_contract.json").read_text(encoding="utf-8"))
    assert contract["method"] == "stratified"
    assert contract["label_distribution"]["enabled"] is True


def test_auto_classification_requires_stratified_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A"] * 4 + ["B"] * 4, task_hint="classification")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="auto", ratio="8:2")
    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "STRATIFIED_CONFIRMATION_REQUIRED"


def test_stratified_too_small_class_does_not_write(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=["A", "A", "B", "B", "C"], task_hint="classification")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="stratified", ratio="6:2:2")
    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "STRATIFIED_CLASS_TOO_SMALL"
    assert not output_dir.exists()


def test_missing_ratio_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="random")
    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "SPLIT_RATIO_REQUIRED"


def test_invalid_ratio_sum_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="random", train_ratio=0.8, test_ratio=0.3)
    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "SPLIT_RATIO_SUM_INVALID"
    assert not (tmp_path / "split").exists()


def test_duplicate_sample_ids_block_split(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    _write_rows(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S001"], *[[f"S{idx + 1:03d}"] for idx in range(2, 10)]])
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="random", ratio="8:2")
    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["errors"][0]["code"] == "SAMPLE_ID_DUPLICATE"


def test_split_indices_never_duplicate_or_omit_samples(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    output_dir = tmp_path / "split"
    response = split_spectral_package(package_dir=package_dir, output_dir=output_dir, method="random", ratio="6:2:2", random_seed=23)
    rows = _read_csv(output_dir / "split_indices.csv")[1:]
    observed = sorted(int(row[5]) for row in rows)

    assert response["ok"] is True
    assert observed == list(range(10))
    assert len({row[6] for row in rows}) == 10
    assert {row[4] for row in rows} == {"train", "val", "test"}


def test_non_finite_x_blocks_split(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    _write_rows(package_dir / "X.csv", [["900", "1000", "1100"], [1.0, 2.0, "nan"], [2.0, 3.0, 4.0]])
    _write_rows(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"]])
    contract = json.loads((package_dir / "data_contract.json").read_text(encoding="utf-8"))
    contract["shape"] = {"n_samples": 2, "n_features": 3}
    (package_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    response = split_spectral_package(package_dir=package_dir, output_dir=tmp_path / "split", method="random", ratio="1:1")

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "X_NON_FINITE"


def test_splitter_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package", labels=None, task_hint="unsupervised")
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-splitter" / "scripts" / "split_spectral_package.py",
            REPO_ROOT / "scripts" / "splitter" / "split_spectral_package.py",
        ]
    ):
        output_dir = tmp_path / f"split_{idx}"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(output_dir),
                "--method",
                "random",
                "--ratio",
                "8:2",
                "--random-seed",
                "42",
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
        assert payload["result"]["split_contract"] == "split_contract.json"
        assert (output_dir / "split_contract.json").exists()

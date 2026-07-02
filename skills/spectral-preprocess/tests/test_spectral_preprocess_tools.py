from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from spectral_core.preprocess.workflow import preprocess_spectral_package


REPO_ROOT = Path(__file__).resolve().parents[3]
LONG_SPLIT_HEADER = ["split_type", "method", "fold_id", "repeat_id", "role", "sample_index", "sample_id", "label", "group_id"]


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
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14],
            [11, 12, 13, 14, 15],
            [12, 13, 14, 15, 16],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"], [3, 1200, "cm-1"], [4, 1300, "cm-1"]])
    _write_rows(root / "y.csv", [["class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    _write_rows(root / "metadata.csv", [["batch"], ["B1"], ["B1"], ["B1"], ["B2"], ["B2"], ["B2"]])
    contract = {
        "contract_id": "data-preprocess-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": "metadata.csv"},
        "shape": {"n_samples": 6, "n_features": 5},
        "task_hint": "classification",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _write_reflectance_package(root: Path) -> Path:
    _write_package(root)
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100", "1200", "1300"],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8, 0.9],
            [0.6, 0.7, 0.8, 0.9, 1.0],
        ],
    )
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
        "ratios": {"train": 0.5, "val": 0.1667, "test": 0.3333},
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
        "folds": [
            {"fold_id": 1, "train_indices": [0, 1, 2, 3], "val_indices": [4, 5]},
            {"fold_id": 2, "train_indices": [0, 1, 4, 5], "val_indices": [2, 3]},
            {"fold_id": 3, "train_indices": [2, 3, 4, 5], "val_indices": [0, 1]},
        ],
    }
    path = root / "split_contract.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def _write_repeated_split(root: Path) -> Path:
    contract = {
        "contract_type": "split_contract",
        "contract_id": "split-repeated-test",
        "split_type": "repeated_holdout",
        "method": "monte_carlo_cv",
        "n_repeats": 2,
        "repeats": [
            {"repeat_id": 1, "train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
            {"repeat_id": 2, "train_indices": [2, 3, 4, 5], "test_indices": [0, 1]},
        ],
    }
    path = root / "split_contract.json"
    root.mkdir(parents=True, exist_ok=True)
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
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="none")

    assert response["ok"] is True
    assert set(path.name for path in output_dir.iterdir()) == {"X.csv", "sample_ids.csv", "band_axis.csv", "y.csv", "metadata.csv", "data_contract.json", "preprocess_state.json", "preprocess_contract.json"}
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["processing_stage"] == "preprocess"
    assert contract["preprocess_summary"]["methods"] == ["none"]
    assert contract["preprocess_summary"]["fit_scope"] == "not_applicable_stateless_or_per_sample"
    preprocess_contract = json.loads((output_dir / "preprocess_contract.json").read_text(encoding="utf-8"))
    assert preprocess_contract["input_package"] == str((package_dir / "data_contract.json").resolve())
    assert preprocess_contract["split_contract"] == str(split_contract.resolve())
    assert preprocess_contract["output_package"] == str((output_dir / "data_contract.json").resolve())
    assert _read_csv(output_dir / "sample_ids.csv")[1][0] == "S001"


def test_new_holdout_split_contract_indices_are_supported(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_new_holdout_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="standardization")

    assert response["ok"] is True
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["preprocess_summary"]["fit_scope"] == "train_only"
    assert json.loads((output_dir / "preprocess_state.json").read_text(encoding="utf-8"))["split"]["split_type"] == "holdout"


def test_cv_and_repeated_split_contracts_run_partition_wise(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")

    for name, split_contract in {
        "cv": _write_cv_split(tmp_path / "cv_split"),
        "repeated": _write_repeated_split(tmp_path / "repeated_split"),
    }.items():
        response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / f"preprocess_{name}", methods="snv")

        assert response["ok"] is True
        contract = json.loads((tmp_path / f"preprocess_{name}" / "preprocess_contract.json").read_text(encoding="utf-8"))
        assert contract["split_type"] == ("cross_validation" if name == "cv" else "repeated_holdout")
        assert contract["leakage_guard"]["fit_on"] == "not_applicable_stateless_or_per_sample"
        assert len(contract["iterations"]) == (3 if name == "cv" else 2)


def test_snv_is_per_sample_and_does_not_need_train_statistics(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="snv")
    X = _read_X(output_dir / "X.csv")
    state = json.loads((output_dir / "preprocess_state.json").read_text(encoding="utf-8"))

    assert response["ok"] is True
    assert state["fit_scope"] == "not_applicable_stateless_or_per_sample"
    assert state["method_states"][0]["fitted"]["requires_train_fit"] is False
    for row in X:
        assert abs(sum(row) / len(row)) < 1e-9
        std = math.sqrt(sum(value * value for value in row) / len(row))
        assert abs(std - 1.0) < 1e-9


def test_standardization_fits_train_only(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="standardization")
    state = json.loads((output_dir / "preprocess_state.json").read_text(encoding="utf-8"))
    X = _read_X(output_dir / "X.csv")
    train_rows = X[:3]

    assert response["ok"] is True
    fitted = state["method_states"][0]["fitted"]
    assert fitted["mean_vector"] == [2, 3, 4, 5, 6]
    for col_idx in range(5):
        mean = sum(row[col_idx] for row in train_rows) / len(train_rows)
        assert abs(mean) < 1e-9
    assert json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))["preprocess_summary"]["fit_scope"] == "train_only"


def test_minmax_scaling_fits_train_only(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="minmax_scaling")
    X = _read_X(output_dir / "X.csv")
    state = json.loads((output_dir / "preprocess_state.json").read_text(encoding="utf-8"))

    assert response["ok"] is True
    assert X[0] == [0, 0, 0, 0, 0]
    assert X[2] == [1, 1, 1, 1, 1]
    assert state["method_states"][0]["fitted"]["fit_sample_count"] == 3


def test_msc_reference_uses_train_only(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="msc")
    state = json.loads((output_dir / "preprocess_state.json").read_text(encoding="utf-8"))

    assert response["ok"] is True
    assert state["method_states"][0]["fitted"]["reference_spectrum"] == [2, 3, 4, 5, 6]
    assert state["method_states"][0]["fitted"]["fit_sample_count"] == 3


def test_sg_and_derivative_require_parameters_then_run(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    missing = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "missing", methods="sg_smoothing")
    assert missing["ok"] is False
    assert missing["result"]["status"] == "needs_confirmation"
    assert missing["errors"][0]["code"] == "SG_PARAMETERS_REQUIRED"

    output_dir = tmp_path / "preprocess"
    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="sg_smoothing,first_derivative,second_derivative", window_length=5, polyorder=2)
    assert response["ok"] is True
    assert json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))["shape"] == {"n_samples": 6, "n_features": 5}


def test_baseline_methods_require_confirmation_then_run(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    missing = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "missing", methods="polynomial_baseline")
    assert missing["ok"] is False
    assert missing["result"]["status"] == "needs_confirmation"
    assert missing["errors"][0]["code"] == "BASELINE_CONFIRMATION_REQUIRED"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="polynomial_baseline,detrend", poly_degree=1, confirm_baseline=True)
    assert response["ok"] is True
    state = json.loads((tmp_path / "preprocess" / "preprocess_state.json").read_text(encoding="utf-8"))
    assert state["method_states"][0]["method"] == "polynomial_baseline"
    assert state["method_states"][0]["parameters"]["degree"] == 1


def test_absorbance_conversion_requires_confirmation_and_positive_values(tmp_path: Path) -> None:
    package_dir = _write_reflectance_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    missing = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "missing", methods="reflectance_to_absorbance")
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "ABSORBANCE_CONFIRMATION_REQUIRED"

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="reflectance_to_absorbance", confirm_absorbance=True)
    X = _read_X(tmp_path / "preprocess" / "X.csv")
    assert response["ok"] is True
    assert abs(X[0][0] - 1.0) < 1e-9

    rows = _read_csv(package_dir / "X.csv")
    rows[1][1] = "0"
    _write_rows(package_dir / "X.csv", rows)
    blocked = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "blocked", methods="reflectance_to_absorbance", confirm_absorbance=True)
    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "ABSORBANCE_NON_POSITIVE_INPUT"


def test_band_range_select_updates_band_axis_and_contract(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")

    missing = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "missing", methods="band_range_select", band_range="1000:1200")
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "BAND_CHANGE_CONFIRMATION_REQUIRED"

    output_dir = tmp_path / "preprocess"
    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=output_dir, methods="band_range_select", band_range="1000:1200", confirm_band_change=True)
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    band_axis = _read_csv(output_dir / "band_axis.csv")
    X_header = _read_csv(output_dir / "X.csv")[0]

    assert response["ok"] is True
    assert contract["shape"] == {"n_samples": 6, "n_features": 3}
    assert X_header == ["1000", "1100", "1200"]
    assert [row[1] for row in band_axis[1:]] == ["1000", "1100", "1200"]


def test_band_range_is_executed_before_snv_and_recorded(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    output_dir = tmp_path / "preprocess"

    response = preprocess_spectral_package(
        package_dir=package_dir,
        split_contract=split_contract,
        output_dir=output_dir,
        methods="snv,band_range_select",
        band_range="1000:1200",
        confirm_band_change=True,
    )

    assert response["ok"] is True
    assert response["result"]["methods"] == ["band_range_select", "snv"]
    assert any(item["code"] == "PREPROCESS_ORDER_NORMALIZED" for item in response["warnings"])
    X = _read_X(output_dir / "X.csv")
    expected = [-1.224744871391589, 0.0, 1.224744871391589]
    assert all(math.isclose(observed, target, rel_tol=1e-9, abs_tol=1e-9) for observed, target in zip(X[0], expected))

    preprocess_contract = json.loads((output_dir / "preprocess_contract.json").read_text(encoding="utf-8"))
    assert preprocess_contract["requested_methods"] == ["snv", "band_range_select"]
    assert preprocess_contract["executed_methods"] == ["band_range_select", "snv"]
    assert preprocess_contract["order_normalized"] is True
    assert "excluded bands" in preprocess_contract["order_normalization_reason"]
    data_contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert data_contract["preprocess_summary"]["requested_methods"] == ["snv", "band_range_select"]
    assert data_contract["preprocess_summary"]["executed_methods"] == ["band_range_select", "snv"]


def test_train_fit_method_without_split_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    response = preprocess_spectral_package(package_dir=package_dir, output_dir=tmp_path / "preprocess", methods="standardization")

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "SPLIT_CONTRACT_REQUIRED_FOR_FIT"
    assert not (tmp_path / "preprocess").exists()


def test_confirmed_unsplit_fit_warns_and_writes(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "preprocess"
    response = preprocess_spectral_package(package_dir=package_dir, output_dir=output_dir, methods="standardization", confirm_unsplit_fit=True)

    assert response["ok"] is True
    assert response["warnings"][0]["code"] == "UNSPLIT_PREPROCESS_CONFIRMED"
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["preprocess_summary"]["fit_scope"] == "all_samples_confirmed"


def test_incomplete_split_blocks_output(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    rows = _read_csv(split_contract.parent / "split_indices.csv")
    _write_rows(split_contract.parent / "split_indices.csv", rows[:-1])

    response = preprocess_spectral_package(package_dir=package_dir, split_contract=split_contract, output_dir=tmp_path / "preprocess", methods="snv")
    assert response["ok"] is False
    assert response["result"]["status"] == "blocked"
    assert response["errors"][0]["code"] == "SPLIT_INCOMPLETE"
    assert not (tmp_path / "preprocess").exists()


def test_preprocess_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    split_contract = _write_split(tmp_path / "split")
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-preprocess" / "scripts" / "preprocess_spectral_package.py",
            REPO_ROOT / "scripts" / "preprocess" / "preprocess_spectral_package.py",
        ]
    ):
        output_dir = tmp_path / f"preprocess_{idx}"
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
                "--methods",
                "snv",
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
        assert (output_dir / "preprocess_state.json").exists()

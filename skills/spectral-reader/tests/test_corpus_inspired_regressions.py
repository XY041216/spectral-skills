from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


FORBIDDEN_OUTPUTS = {
    "_internal",
    "package_manifest.json",
    "summary.json",
    "preview_report.json",
    "validation_report.json",
    "profile_summary.json",
    "logs",
    "decision_trace.json",
    "read_plan.json",
    "folder_inventory.json",
    "variable_inventory.json",
    "dataset_inventory.json",
}


def _assert_minimal_ready(output: Path) -> dict:
    observed = {path.name for path in output.iterdir()}
    assert not (observed & FORBIDDEN_OUTPUTS)
    assert {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"}.issubset(observed)
    contract = json.loads((output / "data_contract.json").read_text(encoding="utf-8"))
    assert "confidence_scores" not in contract
    assert "read_plan" not in contract
    return contract


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_empty_header_sample_id_column_index_aligns_external_labels(tmp_path: Path) -> None:
    source = tmp_path / "matrix_no_header.csv"
    labels = tmp_path / "labels.csv"
    _write_csv(
        source,
        [
            ["", 900, 1000, 1100],
            ["S001", 0.1, 0.2, 0.3],
            ["S002", 0.4, 0.5, 0.6],
        ],
    )
    _write_csv(labels, [["sample_id", "class"], ["S001", "A"], ["S002", "B"]])

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=source,
        output_dir=output,
        sample_id_column="0",
        spectral_start_column="900",
        spectral_end_column="1100",
        label_file=labels,
        join_key="sample_id",
        label_column="class",
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    contract = _assert_minimal_ready(output)
    assert contract["shape"] == {"n_samples": 2, "n_features": 3}
    assert (output / "sample_ids.csv").read_text(encoding="utf-8").splitlines()[1:] == ["S001", "S002"]


def test_non_numeric_spectral_cell_blocks_without_skipping_rows(tmp_path: Path) -> None:
    source = tmp_path / "bad_value.csv"
    _write_csv(
        source,
        [
            ["sample_id", 900, 1000, 1100],
            ["S001", 0.1, 0.2, 0.3],
            ["S002", 0.4, "ERR", 0.6],
            ["S003", 0.7, 0.8, 0.9],
        ],
    )

    output = tmp_path / "out"
    response = read_spectral_dataset(input_path=source, output_dir=output)

    assert response["ok"] is False
    assert (response["errors"][0]["code"] in {"X_NON_NUMERIC", "SPECTRAL_BLOCK_NON_NUMERIC"})
    assert not (output / "X.csv").exists()
    assert not (output / "data_contract.json").exists()


def test_semicolon_decimal_comma_values_are_numeric(tmp_path: Path) -> None:
    source = tmp_path / "decimal_comma.csv"
    source.write_text(
        "sample_id;class;900;1000\nS001;A;0,12;0,13\nS002;B;0,22;0,23\n",
        encoding="utf-8",
    )

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=source,
        output_dir=output,
        delimiter=";",
        sample_id_column="sample_id",
        label_column="class",
    )

    assert response["ok"] is True
    contract = _assert_minimal_ready(output)
    assert contract["shape"] == {"n_samples": 2, "n_features": 2}
    assert (output / "y.csv").exists()


def test_npz_object_vectors_and_metadata_read_with_explicit_variables(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    source = tmp_path / "dataset.npz"
    np.savez(
        source,
        X=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        y=np.array(["A", "B"], dtype=object),
        sample_ids=np.array(["S001", "S002"], dtype=object),
        band_axis=np.array([900, 1000], dtype=float),
        metadata_batch=np.array(["B1", "B2"], dtype=object),
    )

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=source,
        output_dir=output,
        x_var="X",
        y_var="y",
        sample_ids_var="sample_ids",
        band_axis_var="band_axis",
        metadata_var="metadata_batch",
    )

    assert response["ok"] is True
    contract = _assert_minimal_ready(output)
    assert contract["metadata_status"] == "present"
    assert (output / "metadata.csv").exists()
    assert (output / "y.csv").exists()


def test_mat_row_vector_metadata_expands_to_sample_rows(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    scipy_io = pytest.importorskip("scipy.io")
    source = tmp_path / "dataset.mat"
    scipy_io.savemat(
        source,
        {
            "X": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            "labels": np.array(["A", "B"], dtype=object),
            "sample_ids": np.array(["S001", "S002"], dtype=object),
            "band_axis": np.array([900, 1000], dtype=float),
            "batch_codes": np.array([["B1", "B2"]], dtype=object),
        },
    )

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=source,
        output_dir=output,
        x_var="X",
        y_var="labels",
        sample_ids_var="sample_ids",
        band_axis_var="band_axis",
        metadata_var="batch_codes",
    )

    assert response["ok"] is True
    contract = _assert_minimal_ready(output)
    assert contract["metadata_status"] == "present"
    assert len((output / "metadata.csv").read_text(encoding="utf-8").splitlines()) == 3


def test_sample_folder_band_axis_mismatch_blocks(tmp_path: Path) -> None:
    folder = tmp_path / "sample_files"
    folder.mkdir()
    _write_csv(folder / "S001.csv", [["band", "value"], [900, 0.1], [1000, 0.2]])
    _write_csv(folder / "S002.csv", [["band", "value"], [900, 0.3], [1100, 0.4]])

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=folder,
        output_dir=output,
        sample_file_pattern="*.csv",
        sample_file_band_column="band",
        sample_file_value_column="value",
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "PER_FILE_BAND_AXIS_MISMATCH"
    assert not (output / "X.csv").exists()

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "skills" / "spectral-reader" / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _contract(output: Path) -> dict:
    return json.loads((output / "data_contract.json").read_text(encoding="utf-8"))


def _assert_minimal(output: Path) -> None:
    forbidden = {
        "_internal",
        "package_manifest.json",
        "summary.json",
        "preview_report.json",
        "validation_report.json",
        "profile_summary.json",
        "logs",
        "decision_trace.json",
        "read_plan.json",
    }
    observed = {path.name for path in output.iterdir()}
    assert not (observed & forbidden)
    assert {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"}.issubset(observed)
    contract = _contract(output)
    assert not _contains_key(contract, "confidence_scores")
    assert not _contains_key(contract, "read_plan")


def test_samples_as_columns_outputs_samples_by_features(tmp_path: Path) -> None:
    output = tmp_path / "columns"
    response = read_spectral_dataset(
        input_path=FIXTURES / "samples_as_columns_basic" / "spectra_columns.csv",
        output_dir=output,
        sample_orientation="columns",
        band_axis_column="band",
        spectral_columns=["sample_001"],
    )
    assert response["ok"] is True
    assert response["result"]["n_samples"] == 3
    assert response["result"]["n_features"] == 3
    assert _contract(output)["sample_orientation"] == "columns"
    assert _rows(output / "X.csv") == [
        ["900", "1000", "1100"],
        ["0.12", "0.13", "0.14"],
        ["0.15", "0.16", "0.17"],
        ["0.18", "0.19", "0.2"],
    ]
    assert _rows(output / "sample_ids.csv")[1:] == [["sample_001"], ["sample_002"], ["sample_003"]]
    assert [row[1] for row in _rows(output / "band_axis.csv")[1:]] == ["900", "1000", "1100"]
    _assert_minimal(output)


def test_samples_as_columns_without_label_is_unsupervised_ready(tmp_path: Path) -> None:
    output = tmp_path / "columns_unsupervised"
    response = read_spectral_dataset(
        input_path=FIXTURES / "samples_as_columns_no_label" / "spectra_columns.csv",
        output_dir=output,
        sample_orientation="columns",
        band_axis_column="band",
        spectral_columns=["S001"],
    )
    assert response["ok"] is True
    contract = _contract(output)
    assert contract["status"] == "ready"
    assert contract["label_status"] == "absent"
    assert not (output / "y.csv").exists()
    _assert_minimal(output)


def test_external_label_rows_aligns_y_and_keeps_spectrum_order(tmp_path: Path) -> None:
    output = tmp_path / "external"
    response = read_spectral_dataset(
        input_path=FIXTURES / "external_label_rows_basic" / "spectra.csv",
        output_dir=output,
        sample_id_column="sample_id",
        label_file=FIXTURES / "external_label_rows_basic" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    assert _contract(output)["label_status"] == "present"
    assert _rows(output / "sample_ids.csv")[1:] == [["S001"], ["S002"], ["S003"]]
    assert _rows(output / "y.csv")[1:] == [["A"], ["B"], ["A"]]
    assert _dicts(output / "metadata.csv") == [{"batch": "B1"}, {"batch": "B1"}, {"batch": "B2"}]
    _assert_minimal(output)


def test_external_label_metadata_merge_preserves_main_order(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    response = read_spectral_dataset(
        input_path=FIXTURES / "external_label_metadata_merge" / "spectra.csv",
        output_dir=output,
        sample_id_column="sample_id",
        metadata_columns=["instrument"],
        label_file=FIXTURES / "external_label_metadata_merge" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
        spectral_columns=["900 nm", "1000 nm"],
    )
    assert response["ok"] is True
    assert _rows(output / "sample_ids.csv")[1:] == [["S003"], ["S001"], ["S002"]]
    assert _rows(output / "y.csv")[1:] == [["A"], ["A"], ["B"]]
    metadata = _dicts(output / "metadata.csv")
    assert metadata[0] == {"instrument": "NIR-A", "batch": "B2", "operator": "op-c"}
    assert metadata[2] == {"instrument": "NIR-B", "batch": "B1", "operator": "op-b"}
    _assert_minimal(output)


def test_samples_as_columns_with_external_label(tmp_path: Path) -> None:
    output = tmp_path / "columns_external"
    response = read_spectral_dataset(
        input_path=FIXTURES / "samples_as_columns_with_external_label" / "spectra_columns.csv",
        output_dir=output,
        sample_orientation="columns",
        band_axis_column="band",
        spectral_columns=["S001"],
        label_file=FIXTURES / "samples_as_columns_with_external_label" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    assert _contract(output)["sample_orientation"] == "columns"
    assert _rows(output / "y.csv")[1:] == [["A"], ["B"], ["A"]]
    assert _dicts(output / "metadata.csv") == [{"batch": "B1"}, {"batch": "B1"}, {"batch": "B2"}]
    _assert_minimal(output)


def test_external_label_missing_and_duplicate_block(tmp_path: Path) -> None:
    missing = read_spectral_dataset(
        input_path=FIXTURES / "external_label_missing_sample" / "spectra.csv",
        output_dir=tmp_path / "missing",
        sample_id_column="sample_id",
        label_file=FIXTURES / "external_label_missing_sample" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "MISSING_REQUIRED_LABELS"
    duplicate = read_spectral_dataset(
        input_path=FIXTURES / "external_label_duplicate_key" / "spectra.csv",
        output_dir=tmp_path / "duplicate",
        sample_id_column="sample_id",
        label_file=FIXTURES / "external_label_duplicate_key" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert duplicate["ok"] is False
    assert duplicate["errors"][0]["code"] == "DUPLICATE_LABEL_KEYS"


def test_folder_one_file_per_sample_reads_unsupervised_matrix(tmp_path: Path) -> None:
    fixture = FIXTURES / "folder_one_file_per_sample"
    output = tmp_path / "folder_apply"
    response = read_spectral_dataset(input_path=fixture, output_dir=output)
    assert response["ok"] is True
    assert _contract(output)["sample_orientation"] == "one_file_per_sample"
    assert _contract(output)["label_status"] == "absent"
    assert response["result"]["n_samples"] == 3
    assert response["result"]["n_features"] == 3
    assert _rows(output / "X.csv") == [
        ["900", "1000", "1100"],
        ["0.1", "0.11", "0.12"],
        ["0.2", "0.21", "0.22"],
        ["0.3", "0.31", "0.32"],
    ]
    assert _rows(output / "sample_ids.csv")[1:] == [["S001"], ["S002"], ["S003"]]
    assert not (output / "y.csv").exists()
    _assert_minimal(output)


def test_folder_name_as_label_generates_minimal_contract(tmp_path: Path) -> None:
    fixture = FIXTURES / "folder_name_label_samples"
    output = tmp_path / "folder_label_apply"
    response = read_spectral_dataset(
        input_path=fixture,
        output_dir=output,
        folder_name_as_label=True,
        sample_file_pattern="*/*.csv",
        sample_file_band_column="wavelength",
        sample_file_value_column="intensity",
    )
    assert response["ok"] is True
    assert _rows(output / "sample_ids.csv")[1:] == [["S001"], ["S002"], ["S003"], ["S004"]]
    assert _rows(output / "y.csv")[1:] == [["tablet_A"], ["tablet_A"], ["tablet_B"], ["tablet_B"]]
    contract = _contract(output)
    assert contract["status"] == "ready"
    assert contract["label_status"] == "present"
    assert contract["sample_orientation"] == "one_file_per_sample"
    _assert_minimal(output)


def test_folder_name_label_workflow_writes_standard_output(tmp_path: Path) -> None:
    fixture = FIXTURES / "folder_name_label_samples"
    output_dir = tmp_path / "folder_workflow"
    response = read_spectral_dataset(
        input_path=fixture,
        output_dir=output_dir,
        folder_name_as_label=True,
        sample_file_pattern="*/*.csv",
        sample_file_band_column="wavelength",
        sample_file_value_column="intensity",
    )
    assert response["ok"] is True
    result = response["result"]
    assert "stage_results" not in result
    assert result["label_status"] == "present"
    assert result["n_samples"] == 4
    assert (output_dir / "X.csv").exists()
    assert (output_dir / "y.csv").exists()
    assert (output_dir / "data_contract.json").exists()
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "_internal").exists()


def test_complex_workflow_keeps_only_standard_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow"
    response = read_spectral_dataset(
        input_path=FIXTURES / "external_label_rows_basic" / "spectra.csv",
        output_dir=output_dir,
        sample_id_column="sample_id",
        label_file=FIXTURES / "external_label_rows_basic" / "labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    result = response["result"]
    assert "stage_results" not in result
    assert result["X"] == "X.csv"
    assert result["y"] == "y.csv"
    assert result["data_contract"] == "data_contract.json"
    contract = json.loads((output_dir / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["sample_orientation"] == "rows"
    assert contract["label_status"] == "present"
    assert not _contains_key(contract, "confidence_scores")
    assert not (output_dir / "package_manifest.json").exists()
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "_internal").exists()
    _assert_minimal(output_dir)


def test_debug_view_is_not_part_of_reader_api() -> None:
    import inspect

    assert "view" not in inspect.signature(read_spectral_dataset).parameters
    assert "mode" not in inspect.signature(read_spectral_dataset).parameters
    assert "read_plan" not in inspect.signature(read_spectral_dataset).parameters

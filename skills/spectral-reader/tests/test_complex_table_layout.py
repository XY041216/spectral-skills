from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spectral_core.reader.workflow import read_spectral_dataset


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FORBIDDEN_OUTPUTS = {
    "_internal",
    "package_manifest.json",
    "summary.json",
    "preview_report.json",
    "validation_report.json",
    "profile_summary.json",
    "logs",
    "read_plan.json",
    "dataset_inventory.json",
    "variable_inventory.json",
}


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _contract(path: Path) -> dict:
    return json.loads((path / "data_contract.json").read_text(encoding="utf-8"))


def _assert_minimal(out: Path, *, y: bool = False, metadata: bool = False) -> dict:
    expected = {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"}
    if y:
        expected.add("y.csv")
    if metadata:
        expected.add("metadata.csv")
    observed = {path.name for path in out.iterdir()}
    assert expected <= observed
    assert not (observed & FORBIDDEN_OUTPUTS)
    contract = _contract(out)
    encoded = json.dumps(contract)
    assert "confidence_scores" not in encoded
    assert "read_plan" not in encoded
    return contract


def test_csv_multirow_header_reads_minimal_output(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = read_spectral_dataset(
        input_path=FIXTURES / "multirow_header_rows" / "spectra_multirow_header.csv",
        output_dir=out,
        header_rows=[0, 1],
        sample_id_column="sample_id",
        metadata_columns=["batch"],
        label_column="Class",
        spectral_start_column="900",
        spectral_end_column="1100",
    )
    assert result["ok"], result
    assert (result["result"] or {})["status"] == "ready"
    contract = _assert_minimal(out, y=True, metadata=True)
    assert contract["table_layout"]["header_type"] == "multirow"
    assert _rows(out / "X.csv") == [["900", "1000", "1100"], ["0.12", "0.13", "0.14"], ["0.15", "0.16", "0.17"]]
    assert [row[1] for row in _rows(out / "band_axis.csv")[1:]] == ["900", "1000", "1100"]


def test_excel_multirow_header_reads_minimal_output(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    out = tmp_path / "out"
    result = read_spectral_dataset(
        input_path=FIXTURES / "excel_multirow_header" / "spectra_multirow_header.xlsx",
        output_dir=out,
        header_rows=[0, 1],
        sample_id_column="sample_id",
        metadata_columns=["batch"],
        label_column="Class",
        spectral_start_column="900",
        spectral_end_column="1100",
    )
    assert result["ok"], result
    _assert_minimal(out, y=True, metadata=True)
    assert [row[1] for row in _rows(out / "band_axis.csv")[1:]] == ["900", "1000", "1100"]


def test_external_band_axis_file_success_and_contract(tmp_path: Path) -> None:
    out = tmp_path / "out"
    fixture = FIXTURES / "external_band_axis_file"
    result = read_spectral_dataset(
        input_path=fixture / "spectra_matrix.csv",
        output_dir=out,
        source_base_dir=str(fixture),
        sample_id_column="sample_id",
        spectral_start_column="f1",
        spectral_end_column="f3",
        band_axis_file="bands.csv",
        band_axis_column="band",
    )
    assert result["ok"], result
    contract = _assert_minimal(out)
    assert contract["band_axis"]["type"] == "external_file"
    assert [row[1] for row in _rows(out / "band_axis.csv")[1:]] == ["900", "1000", "1100"]


def test_external_band_axis_length_mismatch_blocked(tmp_path: Path) -> None:
    fixture = FIXTURES / "external_band_axis_file"
    result = read_spectral_dataset(
        input_path=fixture / "spectra_matrix.csv",
        output_dir=tmp_path / "out",
        source_base_dir=str(fixture),
        sample_id_column="sample_id",
        spectral_start_column="f1",
        spectral_end_column="f3",
        band_axis_file="bands_bad.csv",
        band_axis_column="band",
    )
    assert not result["ok"]
    assert (result.get("errors") or [{}])[0].get("code") == "BAND_AXIS_LENGTH_MISMATCH"


def test_metadata_spectra_label_partition_excludes_non_spectral(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = read_spectral_dataset(
        input_path=FIXTURES / "metadata_spectra_partition" / "spectra_partition.csv",
        output_dir=out,
        sample_id_column="sample_id",
        metadata_columns=["origin", "batch", "moisture"],
        label_column="Class",
        spectral_start_column="900",
        spectral_end_column="1100",
    )
    assert result["ok"], result
    _assert_minimal(out, y=True, metadata=True)
    assert _rows(out / "X.csv")[0] == ["900", "1000", "1100"]
    assert _rows(out / "metadata.csv")[0] == ["origin", "batch", "moisture"]


def test_spectral_block_non_numeric_blocked(tmp_path: Path) -> None:
    result = read_spectral_dataset(
        input_path=FIXTURES / "spectral_block_non_numeric" / "spectra_non_numeric_block.csv",
        output_dir=tmp_path / "out",
        sample_id_column="sample_id",
        spectral_start_column="900",
        spectral_end_column="1100",
    )
    assert not result["ok"]
    assert (result.get("errors") or [{}])[0].get("code") == "SPECTRAL_BLOCK_NON_NUMERIC"


def test_multi_target_y_outputs_multiple_columns(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = read_spectral_dataset(
        input_path=FIXTURES / "multi_target_regression" / "spectra_multi_target.csv",
        output_dir=out,
        sample_id_column="sample_id",
        target_columns=["total_sugar", "nicotine", "potassium"],
        spectral_start_column="900",
        spectral_end_column="1000",
    )
    assert result["ok"], result
    contract = _assert_minimal(out, y=True)
    assert contract["task_hint"] == "multi_target_regression"
    assert contract["target_columns"] == ["total_sugar", "nicotine", "potassium"]
    assert _rows(out / "y.csv")[0] == ["total_sugar", "nicotine", "potassium"]
    assert _rows(out / "X.csv")[0] == ["900", "1000"]


def test_label_target_conflict_blocked(tmp_path: Path) -> None:
    result = read_spectral_dataset(
        input_path=FIXTURES / "label_target_conflict" / "spectra_label_target_conflict.csv",
        output_dir=tmp_path / "out",
        sample_id_column="sample_id",
        label_column="Class",
        target_columns=["nicotine"],
        spectral_start_column="900",
        spectral_end_column="1000",
    )
    assert not result["ok"]
    assert (result.get("errors") or [{}])[0].get("code") == "LABEL_TARGET_CONFLICT"

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "skills" / "spectral-reader" / "fixtures"

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


FORBIDDEN_OUTPUTS = [
    "_internal",
    "package_manifest.json",
    "summary.json",
    "preview_report.json",
    "validation_report.json",
    "profile_summary.json",
    "logs",
    "decision_trace.json",
    "read_plan.json",
]


def _contract(output: Path) -> dict:
    return json.loads((output / "data_contract.json").read_text(encoding="utf-8"))


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _assert_minimal_output(output: Path, expected: set[str]) -> dict:
    observed = {path.name for path in output.iterdir()}
    assert observed == expected
    for name in FORBIDDEN_OUTPUTS:
        assert not (output / name).exists()
    contract = _contract(output)
    assert "confidence_scores" not in contract
    assert "read_plan" not in contract
    for ref in (contract.get("files") or {}).values():
        if ref:
            assert (output / ref).exists()
    return contract


def test_excel_single_sheet_rows_one_shot_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "xlsx_rows"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_single_sheet_rows" / "spectra_rows.xlsx",
        output_dir=output,
    )
    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "ready"
    assert result["n_samples"] == 3
    assert result["n_features"] == 3
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "xlsx"
    assert contract["sample_orientation"] == "rows"
    assert contract["label_status"] == "present"
    assert contract["source"]["spectral_sheet"] == "Sheet1"


def test_excel_single_sheet_columns_one_shot_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "xlsx_columns"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_single_sheet_columns" / "spectra_columns.xlsx",
        output_dir=output,
        sample_orientation="columns",
        band_axis_column="band",
        spectral_columns=["S001"],
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "xlsx"
    assert contract["sample_orientation"] == "columns"
    assert contract["label_status"] == "absent"
    assert _rows(output / "sample_ids.csv")[1:] == [["S001"], ["S002"], ["S003"]]


def test_excel_multi_sheet_specified_spectral_sheet_reads(tmp_path: Path) -> None:
    output = tmp_path / "xlsx_spectral_sheet"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_multi_sheet_spectra_label" / "workbook_spectra_label.xlsx",
        output_dir=output,
        spectral_sheet="Spectra",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source"]["spectral_sheet"] == "Spectra"
    assert contract["source"]["label_sheet"] is None
    assert contract["label_status"] == "absent"


def test_excel_label_sheet_aligns_by_sample_id_and_merges_metadata(tmp_path: Path) -> None:
    output = tmp_path / "xlsx_label_sheet"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_multi_sheet_spectra_label" / "workbook_spectra_label.xlsx",
        output_dir=output,
        spectral_sheet="Spectra",
        label_sheet="Labels",
        sample_id_column="sample_id",
        label_column="Class",
        metadata_columns=["batch"],
        join_key="sample_id",
    )
    assert response["ok"] is True
    assert response.get("warnings") == []
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["source"]["spectral_sheet"] == "Spectra"
    assert contract["source"]["label_sheet"] == "Labels"
    assert contract["label_status"] == "present"
    assert contract["metadata_status"] == "present"
    assert _rows(output / "y.csv")[1:] == [["A"], ["B"], ["A"]]
    assert _rows(output / "metadata.csv")[1:] == [["B1"], ["B1"], ["B2"]]


def test_excel_multi_sheet_without_spectral_sheet_needs_confirmation(tmp_path: Path) -> None:
    output = tmp_path / "needs_confirmation"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_multi_sheet_need_confirmation" / "workbook_multiple_candidates.xlsx",
        output_dir=output,
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["spectral_sheet"]
    assert not output.exists()


def test_excel_missing_sheet_blocks_without_outputs(tmp_path: Path) -> None:
    output = tmp_path / "missing_sheet"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_multi_sheet_spectra_label" / "workbook_spectra_label.xlsx",
        output_dir=output,
        spectral_sheet="Missing",
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "SHEET_NOT_FOUND"
    assert not (output / "X.csv").exists()


def test_excel_label_sheet_duplicate_key_blocks(tmp_path: Path) -> None:
    output = tmp_path / "duplicate_label"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_label_duplicate_key" / "workbook_duplicate_key.xlsx",
        output_dir=output,
        spectral_sheet="Spectra",
        label_sheet="Labels",
        sample_id_column="sample_id",
        label_column="Class",
        join_key="sample_id",
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "DUPLICATE_LABEL_KEYS"
    assert not (output / "data_contract.json").exists()


def test_excel_label_sheet_missing_label_blocks(tmp_path: Path) -> None:
    output = tmp_path / "missing_label"
    response = read_spectral_dataset(
        input_path=FIXTURES / "excel_label_missing_sample" / "workbook_missing_label.xlsx",
        output_dir=output,
        spectral_sheet="Spectra",
        label_sheet="Labels",
        sample_id_column="sample_id",
        label_column="Class",
        join_key="sample_id",
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "MISSING_REQUIRED_LABELS"
    assert not (output / "data_contract.json").exists()


def test_ods_single_sheet_rows_or_missing_engine_block(tmp_path: Path) -> None:
    output = tmp_path / "ods_rows"
    source = tmp_path / "spectra_rows.ods"
    if importlib.util.find_spec("odf") is None:
        source.write_bytes(b"not an ods workbook")
        response = read_spectral_dataset(input_path=source, output_dir=output)
        assert response["ok"] is False
        assert response["errors"][0]["code"] in {"PREVIEW_FAILED", "ODS_ENGINE_MISSING"}
        assert not (output / "data_contract.json").exists()
        return

    pd.DataFrame(
        {
            "sample_id": ["S001", "S002"],
            "900": [0.12, 0.15],
            "1000": [0.13, 0.16],
            "Class": ["A", "B"],
        }
    ).to_excel(source, index=False, engine="odf")
    response = read_spectral_dataset(input_path=source, output_dir=output)
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "ods"
    assert contract["label_status"] == "present"

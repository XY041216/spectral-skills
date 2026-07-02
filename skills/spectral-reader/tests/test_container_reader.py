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


FORBIDDEN_OUTPUTS = [
    "_internal",
    "package_manifest.json",
    "summary.json",
    "preview_report.json",
    "validation_report.json",
    "profile_summary.json",
    "variable_inventory.json",
    "logs",
    "decision_trace.json",
    "read_plan.json",
]


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _assert_minimal(output: Path, expected: set[str]) -> dict:
    assert {path.name for path in output.iterdir()} == expected
    for name in FORBIDDEN_OUTPUTS:
        assert not (output / name).exists()
    contract = json.loads((output / "data_contract.json").read_text(encoding="utf-8"))
    assert "confidence_scores" not in contract
    assert "read_plan" not in contract
    assert "variable_inventory" not in contract
    for ref in (contract.get("files") or {}).values():
        if ref:
            assert (output / ref).exists()
    return contract


def test_npy_single_matrix_generates_ids_and_band_axis(tmp_path: Path) -> None:
    output = tmp_path / "npy"
    response = read_spectral_dataset(
        input_path=FIXTURES / "npy_single_matrix" / "X.npy",
        output_dir=output,
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "npy"
    assert contract["source"]["X_variable"] == "__npy_array__"
    assert contract["label_status"] == "absent"
    assert _rows(output / "sample_ids.csv")[1:] == [["sample_001"], ["sample_002"], ["sample_003"]]
    assert _rows(output / "band_axis.csv")[1:] == [["0", "feature_001", ""], ["1", "feature_002", ""], ["2", "feature_003", ""]]


def test_npz_explicit_variable_mapping_reads_xy_axis(tmp_path: Path) -> None:
    output = tmp_path / "npz"
    response = read_spectral_dataset(
        input_path=FIXTURES / "npz_basic_xy" / "dataset.npz",
        output_dir=output,
        x_var="X",
        y_var="y",
        sample_ids_var="sample_ids",
        band_axis_var="band_axis",
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "npz"
    assert contract["source"]["X_variable"] == "X"
    assert contract["source"]["y_variable"] == "y"
    assert contract["label_status"] == "present"
    assert _rows(output / "y.csv")[1:] == [["A"], ["B"], ["A"]]


def test_npz_single_x_candidate_can_auto_select(tmp_path: Path) -> None:
    output = tmp_path / "npz_auto"
    response = read_spectral_dataset(
        input_path=FIXTURES / "npz_single_x_candidate" / "dataset_one_x.npz",
        output_dir=output,
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source"]["X_variable"] == "X"


def test_npz_multiple_x_candidates_needs_confirmation(tmp_path: Path) -> None:
    output = tmp_path / "npz_multi"
    response = read_spectral_dataset(
        input_path=FIXTURES / "npz_multiple_x_candidates" / "dataset_multi_x.npz",
        output_dir=output,
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["x_var"]
    assert not output.exists()


def test_npz_missing_or_bad_variables_block(tmp_path: Path) -> None:
    missing = read_spectral_dataset(
        input_path=FIXTURES / "npz_basic_xy" / "dataset.npz",
        output_dir=tmp_path / "missing",
        x_var="Missing",
    )
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "X_VAR_NOT_FOUND"

    bad_y = read_spectral_dataset(
        input_path=FIXTURES / "npz_bad_lengths" / "dataset_bad_y.npz",
        output_dir=tmp_path / "bad_y",
        x_var="X",
        y_var="y",
    )
    assert bad_y["ok"] is False
    assert bad_y["errors"][0]["code"] == "Y_LENGTH_MISMATCH"

    bad_band = read_spectral_dataset(
        input_path=FIXTURES / "npz_bad_lengths" / "dataset_bad_band.npz",
        output_dir=tmp_path / "bad_band",
        x_var="X",
        band_axis_var="band_axis",
    )
    assert bad_band["ok"] is False
    assert bad_band["errors"][0]["code"] == "BAND_AXIS_LENGTH_MISMATCH"


def test_npz_external_label_alignment_requires_real_sample_ids_and_succeeds(tmp_path: Path) -> None:
    fixture = FIXTURES / "npz_external_label"
    output = tmp_path / "npz_external"
    response = read_spectral_dataset(
        input_path=fixture / "spectra.npz",
        output_dir=output,
        x_var="X",
        sample_ids_var="sample_ids",
        band_axis_var="band_axis",
        label_file=fixture / "labels.csv",
        label_column="Class",
        join_key="sample_id",
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["label_status"] == "present"

    blocked = read_spectral_dataset(
        input_path=fixture / "spectra.npz",
        output_dir=tmp_path / "npz_external_blocked",
        x_var="X",
        band_axis_var="band_axis",
        label_file=fixture / "labels.csv",
        label_column="Class",
        join_key="sample_id",
    )
    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "EXTERNAL_LABEL_REQUIRES_SAMPLE_IDS"


def test_mat_basic_variables_read_success(tmp_path: Path) -> None:
    output = tmp_path / "mat"
    response = read_spectral_dataset(
        input_path=FIXTURES / "mat_basic_xy" / "dataset.mat",
        output_dir=output,
        x_var="X",
        y_var="y",
        sample_ids_var="sample_ids",
        band_axis_var="band_axis",
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "mat"
    assert contract["source"]["X_variable"] == "X"
    assert contract["label_status"] == "present"


def test_mat_multiple_x_candidates_and_missing_x(tmp_path: Path) -> None:
    multi = read_spectral_dataset(
        input_path=FIXTURES / "mat_multiple_x_candidates" / "dataset_multi_x.mat",
        output_dir=tmp_path / "mat_multi",
    )
    assert multi["ok"] is True
    assert multi["result"]["status"] == "needs_confirmation"
    assert multi["result"]["required_fields"] == ["x_var"]

    missing = read_spectral_dataset(
        input_path=FIXTURES / "mat_missing_x_var" / "dataset_missing_x.mat",
        output_dir=tmp_path / "mat_missing",
    )
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "X_VAR_NOT_FOUND"


def test_mat_v73_blocks_clearly(tmp_path: Path) -> None:
    response = read_spectral_dataset(
        input_path=FIXTURES / "mat_v73_header" / "dataset_v73.mat",
        output_dir=tmp_path / "mat_v73",
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] in {"PREVIEW_FAILED", "MAT_V73_NOT_SUPPORTED"}
    assert not (tmp_path / "mat_v73" / "data_contract.json").exists()

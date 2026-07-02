from __future__ import annotations

import csv
import importlib.util
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
    "dataset_inventory.json",
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
    assert "dataset_inventory" not in contract
    for ref in (contract.get("files") or {}).values():
        if ref:
            assert (output / ref).exists()
    return contract


def test_hdf5_single_x_dataset_generates_ids_and_band_axis(tmp_path: Path) -> None:
    output = tmp_path / "hdf5_single"
    response = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_single_x" / "dataset.h5",
        output_dir=output,
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "hdf5"
    assert contract["source"]["X_path"] == "/X"
    assert contract["label_status"] == "absent"
    assert _rows(output / "sample_ids.csv")[1:] == [["sample_001"], ["sample_002"], ["sample_003"]]


def test_hdf5_explicit_path_mapping_reads_xy_axis(tmp_path: Path) -> None:
    output = tmp_path / "hdf5_basic"
    response = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_basic_xy" / "dataset.h5",
        output_dir=output,
        x_path="/spectra/X",
        y_path="/labels/y",
        sample_ids_path="/meta/sample_ids",
        band_axis_path="/axis/band_axis",
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "hdf5"
    assert contract["source"]["X_path"] == "/spectra/X"
    assert contract["source"]["y_path"] == "/labels/y"
    assert contract["label_status"] == "present"
    assert _rows(output / "sample_ids.csv")[1:] == [["S001"], ["S002"], ["S003"]]


def test_hdf5_multiple_x_candidates_needs_confirmation(tmp_path: Path) -> None:
    output = tmp_path / "hdf5_multi"
    response = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_multiple_x_candidates" / "dataset_multi_x.h5",
        output_dir=output,
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["x_path"]
    assert not output.exists()


def test_hdf5_missing_or_bad_paths_block(tmp_path: Path) -> None:
    missing = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_basic_xy" / "dataset.h5",
        output_dir=tmp_path / "missing",
        x_path="/missing/X",
    )
    assert missing["ok"] is False
    assert missing["errors"][0]["code"] == "DATASET_PATH_NOT_FOUND"

    bad_y = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_bad_lengths" / "dataset_bad_y.h5",
        output_dir=tmp_path / "bad_y",
        x_path="/X",
        y_path="/y",
    )
    assert bad_y["ok"] is False
    assert bad_y["errors"][0]["code"] == "Y_LENGTH_MISMATCH"

    bad_band = read_spectral_dataset(
        input_path=FIXTURES / "hdf5_bad_lengths" / "dataset_bad_band.h5",
        output_dir=tmp_path / "bad_band",
        x_path="/X",
        band_axis_path="/band_axis",
    )
    assert bad_band["ok"] is False
    assert bad_band["errors"][0]["code"] == "BAND_AXIS_LENGTH_MISMATCH"


def test_hdf5_external_label_alignment_requires_real_sample_ids_and_succeeds(tmp_path: Path) -> None:
    fixture = FIXTURES / "hdf5_external_label"
    output = tmp_path / "hdf5_external"
    response = read_spectral_dataset(
        input_path=fixture / "spectra.h5",
        output_dir=output,
        x_path="/spectra/X",
        sample_ids_path="/meta/sample_ids",
        band_axis_path="/axis/band_axis",
        label_file=fixture / "labels.csv",
        label_column="Class",
        join_key="sample_id",
        metadata_columns=["Batch"],
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["label_status"] == "present"

    blocked = read_spectral_dataset(
        input_path=fixture / "spectra.h5",
        output_dir=tmp_path / "hdf5_external_blocked",
        x_path="/spectra/X",
        band_axis_path="/axis/band_axis",
        label_file=fixture / "labels.csv",
        label_column="Class",
        join_key="sample_id",
    )
    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "EXTERNAL_LABEL_REQUIRES_SAMPLE_IDS"


def test_netcdf_dependency_or_reading_path(tmp_path: Path) -> None:
    if importlib.util.find_spec("netCDF4") is None:
        response = read_spectral_dataset(
            input_path=FIXTURES / "netcdf_basic_xy" / "dataset.nc",
            output_dir=tmp_path / "netcdf_missing_dep",
            x_path="X",
        )
        assert response["ok"] is False
        assert response["errors"][0]["code"] == "NETCDF4_MISSING"
        return

    output = tmp_path / "netcdf_basic"
    response = read_spectral_dataset(
        input_path=FIXTURES / "netcdf_basic_xy" / "dataset.nc",
        output_dir=output,
        x_path="X",
        y_path="y",
        sample_ids_path="sample_ids",
        band_axis_path="band_axis",
    )
    assert response["ok"] is True
    contract = _assert_minimal(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "netcdf"
    assert contract["source"]["X_path"] == "X"


def test_netcdf_multiple_x_candidates_when_available(tmp_path: Path) -> None:
    if importlib.util.find_spec("netCDF4") is None:
        return
    response = read_spectral_dataset(
        input_path=FIXTURES / "netcdf_multiple_x_candidates" / "dataset_multi_x.nc",
        output_dir=tmp_path / "netcdf_multi",
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["x_path"]

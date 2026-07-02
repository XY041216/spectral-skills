from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "skills" / "spectral-reader" / "fixtures"
CLI = REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "read_spectral_dataset.py"

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

FORBIDDEN_CONTRACT_KEYS = {
    "confidence_scores",
    "read_plan",
    "preview_evidence",
    "validation_details",
    "logs",
    "inventory",
    "package_manifest",
}


@dataclass(frozen=True)
class ReadyCase:
    name: str
    input_path: Path | str
    args: list[str] = field(default_factory=list)
    expect_y: bool = False
    expect_metadata: bool = False


@dataclass(frozen=True)
class NonReadyCase:
    name: str
    input_path: Path | str
    args: list[str] = field(default_factory=list)
    status: str = "needs_confirmation"
    code: str | None = None


READY_CASES = [
    ReadyCase("csv_rows", FIXTURES / "workflow_basic" / "data.csv", expect_y=True),
    ReadyCase(
        "csv_columns",
        FIXTURES / "samples_as_columns_basic" / "spectra_columns.csv",
        ["--sample-orientation", "columns", "--band-axis-column", "band", "--spectral-columns", "sample_001"],
    ),
    ReadyCase(
        "external_label_sample_id",
        FIXTURES / "external_label_rows_basic" / "spectra.csv",
        [
            "--sample-id-column",
            "sample_id",
            "--label-file",
            str(FIXTURES / "external_label_rows_basic" / "labels.csv"),
            "--join-key",
            "sample_id",
            "--label-column",
            "Class",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase("one_file_per_sample_folder", FIXTURES / "folder_one_file_per_sample", expect_metadata=True),
    ReadyCase("mixed_folder_single_spectra", FIXTURES / "mixed_folder_single_spectra"),
    ReadyCase(
        "mixed_folder_spectra_label",
        FIXTURES / "mixed_folder_spectra_label",
        ["--join-key", "sample_id", "--label-column", "Class"],
        expect_y=True,
    ),
    ReadyCase(
        "mixed_folder_spectra_label_metadata",
        FIXTURES / "mixed_folder_spectra_label_metadata",
        ["--join-key", "sample_id", "--label-column", "Class"],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase(
        "mixed_folder_band_axis",
        FIXTURES / "mixed_folder_spectra_label_band_axis",
        ["--join-key", "sample_id", "--label-column", "Class", "--band-axis-column", "band"],
        expect_y=True,
    ),
    ReadyCase(
        "folder_name_as_label",
        FIXTURES / "folder_name_as_label_basic",
        [
            "--folder-name-as-label",
            "--sample-file-pattern",
            "*/*.csv",
            "--sample-file-band-column",
            "band",
            "--sample-file-value-column",
            "absorbance",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase(
        "file_name_as_label",
        FIXTURES / "file_name_as_label_basic",
        [
            "--file-name-as-label",
            "--sample-file-pattern",
            "*.csv",
            "--sample-file-band-column",
            "band",
            "--sample-file-value-column",
            "reflectance",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase(
        "row_order_label_alignment",
        FIXTURES / "row_order_label_alignment_basic" / "spectra.csv",
        [
            "--label-file",
            str(FIXTURES / "row_order_label_alignment_basic" / "labels.csv"),
            "--label-column",
            "Class",
            "--label-alignment",
            "row_order",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase("excel_rows", FIXTURES / "excel_single_sheet_rows" / "spectra_rows.xlsx", expect_y=True),
    ReadyCase(
        "excel_columns",
        FIXTURES / "excel_single_sheet_columns" / "spectra_columns.xlsx",
        ["--sample-orientation", "columns", "--band-axis-column", "band", "--spectral-columns", "S001"],
    ),
    ReadyCase(
        "excel_multi_sheet_labels",
        FIXTURES / "excel_multi_sheet_spectra_label" / "workbook_spectra_label.xlsx",
        [
            "--spectral-sheet",
            "Spectra",
            "--label-sheet",
            "Labels",
            "--sample-id-column",
            "sample_id",
            "--join-key",
            "sample_id",
            "--label-column",
            "Class",
            "--metadata-columns",
            "batch",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase("npy_single_matrix", FIXTURES / "npy_single_matrix" / "X.npy"),
    ReadyCase(
        "npz_variable_map",
        FIXTURES / "npz_basic_xy" / "dataset.npz",
        ["--x-var", "X", "--y-var", "y", "--sample-ids-var", "sample_ids", "--band-axis-var", "band_axis"],
        expect_y=True,
    ),
    ReadyCase(
        "mat_variable_map",
        FIXTURES / "mat_basic_xy" / "dataset.mat",
        ["--x-var", "X", "--y-var", "y", "--sample-ids-var", "sample_ids", "--band-axis-var", "band_axis"],
        expect_y=True,
    ),
    ReadyCase("hdf5_single_dataset", FIXTURES / "hdf5_single_x" / "dataset.h5"),
    ReadyCase(
        "hdf5_path_map",
        FIXTURES / "hdf5_basic_xy" / "dataset.h5",
        ["--x-path", "/spectra/X", "--y-path", "/labels/y", "--sample-ids-path", "/meta/sample_ids", "--band-axis-path", "/axis/band_axis"],
        expect_y=True,
    ),
    ReadyCase(
        "external_band_axis",
        FIXTURES / "external_band_axis_file" / "spectra_matrix.csv",
        [
            "--source-base-dir",
            str(FIXTURES / "external_band_axis_file"),
            "--sample-id-column",
            "sample_id",
            "--spectral-start-column",
            "f1",
            "--spectral-end-column",
            "f3",
            "--band-axis-file",
            "bands.csv",
            "--band-axis-column",
            "band",
        ],
    ),
    ReadyCase(
        "multirow_header",
        FIXTURES / "multirow_header_rows" / "spectra_multirow_header.csv",
        [
            "--header-rows",
            "0,1",
            "--sample-id-column",
            "sample_id",
            "--metadata-columns",
            "batch",
            "--label-column",
            "Class",
            "--spectral-start-column",
            "900",
            "--spectral-end-column",
            "1100",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase(
        "metadata_spectra_label_partition",
        FIXTURES / "metadata_spectra_partition" / "spectra_partition.csv",
        [
            "--sample-id-column",
            "sample_id",
            "--metadata-columns",
            "origin,batch,moisture",
            "--label-column",
            "Class",
            "--spectral-start-column",
            "900",
            "--spectral-end-column",
            "1100",
        ],
        expect_y=True,
        expect_metadata=True,
    ),
    ReadyCase(
        "multi_target_regression",
        FIXTURES / "multi_target_regression" / "spectra_multi_target.csv",
        [
            "--sample-id-column",
            "sample_id",
            "--target-columns",
            "total_sugar,nicotine,potassium",
            "--spectral-start-column",
            "900",
            "--spectral-end-column",
            "1000",
        ],
        expect_y=True,
    ),
]


NEEDS_CONFIRMATION_CASES = [
    NonReadyCase("mixed_multiple_spectra", FIXTURES / "mixed_folder_multiple_spectra_candidates", ["--join-key", "sample_id", "--label-column", "Class"]),
    NonReadyCase("mixed_multiple_labels", FIXTURES / "mixed_folder_multiple_label_candidates", ["--join-key", "sample_id", "--label-column", "Class"]),
    NonReadyCase("npz_multiple_x", FIXTURES / "npz_multiple_x_candidates" / "dataset_multi_x.npz"),
    NonReadyCase("hdf5_multiple_x", FIXTURES / "hdf5_multiple_x_candidates" / "dataset_multi_x.h5"),
    NonReadyCase("excel_multiple_sheets", FIXTURES / "excel_multi_sheet_need_confirmation" / "workbook_multiple_candidates.xlsx"),
]


BLOCKED_CASES = [
    NonReadyCase(
        "row_order_without_explicit_flag",
        FIXTURES / "row_order_alignment_without_explicit_flag" / "spectra.csv",
        ["--label-file", str(FIXTURES / "row_order_alignment_without_explicit_flag" / "labels.csv"), "--label-column", "Class"],
        status="blocked",
        code="ROW_ORDER_ALIGNMENT_NOT_ALLOWED",
    ),
    NonReadyCase(
        "row_order_length_mismatch",
        FIXTURES / "row_order_label_alignment_length_mismatch" / "spectra.csv",
        [
            "--label-file",
            str(FIXTURES / "row_order_label_alignment_length_mismatch" / "labels.csv"),
            "--label-column",
            "Class",
            "--allow-row-order-labels",
        ],
        status="blocked",
        code="ROW_ORDER_LENGTH_MISMATCH",
    ),
    NonReadyCase(
        "external_label_duplicate_key",
        FIXTURES / "external_label_duplicate_key" / "spectra.csv",
        [
            "--sample-id-column",
            "sample_id",
            "--label-file",
            str(FIXTURES / "external_label_duplicate_key" / "labels.csv"),
            "--join-key",
            "sample_id",
            "--label-column",
            "Class",
        ],
        status="blocked",
        code="DUPLICATE_LABEL_KEYS",
    ),
    NonReadyCase(
        "band_axis_length_mismatch",
        FIXTURES / "external_band_axis_file" / "spectra_matrix.csv",
        [
            "--source-base-dir",
            str(FIXTURES / "external_band_axis_file"),
            "--sample-id-column",
            "sample_id",
            "--spectral-start-column",
            "f1",
            "--spectral-end-column",
            "f3",
            "--band-axis-file",
            "bands_bad.csv",
            "--band-axis-column",
            "band",
        ],
        status="blocked",
        code="BAND_AXIS_LENGTH_MISMATCH",
    ),
    NonReadyCase(
        "spectral_block_non_numeric",
        FIXTURES / "spectral_block_non_numeric" / "spectra_non_numeric_block.csv",
        ["--sample-id-column", "sample_id", "--spectral-start-column", "900", "--spectral-end-column", "1100"],
        status="blocked",
        code="SPECTRAL_BLOCK_NON_NUMERIC",
    ),
]


def test_freeze_acceptance_ready_outputs_are_downstream_loadable(tmp_path: Path) -> None:
    cases = list(READY_CASES)
    if importlib.util.find_spec("netCDF4") is not None:
        cases.append(
            ReadyCase(
                "netcdf_variable_path",
                FIXTURES / "netcdf_basic_xy" / "dataset.nc",
                ["--x-path", "X", "--y-path", "y", "--sample-ids-path", "sample_ids", "--band-axis-path", "band_axis"],
                expect_y=True,
            )
        )

    assert len(cases) >= 20
    for case in cases:
        output = tmp_path / case.name
        payload = _run_reader(case.input_path, output, case.args, expect_success=True)
        assert payload["result"]["status"] == "ready", case.name
        _assert_downstream_loadable(output, expect_y=case.expect_y, expect_metadata=case.expect_metadata)


def test_freeze_acceptance_needs_confirmation_does_not_write_ready_outputs(tmp_path: Path) -> None:
    for case in NEEDS_CONFIRMATION_CASES:
        output = tmp_path / case.name
        payload = _run_reader(case.input_path, output, case.args, expect_success=True)
        result = payload["result"]
        assert result["status"] == "needs_confirmation", case.name
        assert result.get("reason"), case.name
        assert result.get("required_fields"), case.name
        assert result.get("suggested_arguments"), case.name
        assert not (output / "X.csv").exists()
        assert not (output / "data_contract.json").exists()


def test_freeze_acceptance_blocked_does_not_write_ready_outputs(tmp_path: Path) -> None:
    for case in BLOCKED_CASES:
        output = tmp_path / case.name
        payload = _run_reader(case.input_path, output, case.args, expect_success=False)
        assert payload["ok"] is False, case.name
        errors = payload.get("errors") or []
        assert errors and errors[0].get("message"), case.name
        if case.code:
            assert errors[0].get("code") == case.code
        assert not (output / "X.csv").exists()
        assert not (output / "data_contract.json").exists()


def _run_reader(input_path: Path | str, output: Path, args: list[str], *, expect_success: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(CLI),
        "--input",
        str(input_path),
        "--output-dir",
        str(output),
        "--json",
        *args,
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if expect_success:
        assert completed.returncode == 0, completed.stderr or completed.stdout
    else:
        assert completed.returncode != 0, completed.stdout
    assert completed.stdout.strip(), completed.stderr
    return json.loads(completed.stdout)


def _assert_downstream_loadable(output: Path, *, expect_y: bool, expect_metadata: bool) -> None:
    observed = {path.name for path in output.iterdir()}
    expected = {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"}
    if expect_y:
        expected.add("y.csv")
    if expect_metadata:
        expected.add("metadata.csv")
    assert observed == expected
    assert not (observed & FORBIDDEN_OUTPUTS)

    contract = json.loads((output / "data_contract.json").read_text(encoding="utf-8"))
    encoded_contract = json.dumps(contract, ensure_ascii=False)
    for key in FORBIDDEN_CONTRACT_KEYS:
        assert key not in encoded_contract
    for key in ["status", "reader_version", "source_type", "source", "files", "shape", "sample_orientation", "label_status", "metadata_status", "task_hint", "band_axis", "warnings"]:
        assert key in contract

    files = contract["files"]
    for ref in files.values():
        if ref:
            assert (output / ref).exists()

    x_rows = _read_csv(output / files["X"])
    sample_rows = _read_csv(output / files["sample_ids"])
    band_rows = _read_csv(output / files["band_axis"])
    assert len(x_rows) >= 2
    assert len(sample_rows) - 1 == len(x_rows) - 1 == contract["shape"]["n_samples"]
    assert len(band_rows) - 1 == len(x_rows[0]) == contract["shape"]["n_features"]

    for row in x_rows[1:]:
        assert len(row) == contract["shape"]["n_features"]
        for value in row:
            float(value)

    y_ref = files.get("y")
    assert bool(y_ref) is expect_y
    if y_ref:
        assert len(_read_csv(output / y_ref)) - 1 == contract["shape"]["n_samples"]

    metadata_ref = files.get("metadata")
    assert bool(metadata_ref) is expect_metadata
    if metadata_ref:
        assert len(_read_csv(output / metadata_ref)) - 1 == contract["shape"]["n_samples"]


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))

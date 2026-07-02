from __future__ import annotations

import json
import subprocess
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
    "logs",
    "decision_trace.json",
    "read_plan.json",
]


def _assert_minimal_output(output: Path, expected: set[str]) -> dict:
    observed = {path.name for path in output.iterdir()}
    assert observed == expected
    for forbidden in FORBIDDEN_OUTPUTS:
        assert not (output / forbidden).exists()
    contract = json.loads((output / "data_contract.json").read_text(encoding="utf-8"))
    assert "confidence_scores" not in contract
    assert "read_plan" not in contract
    assert contract["files"]["X"] == "X.csv"
    assert (output / contract["files"]["X"]).exists()
    assert (output / contract["files"]["sample_ids"]).exists()
    assert (output / contract["files"]["band_axis"]).exists()
    return contract


def test_one_shot_reads_csv_rows_into_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "one_shot_rows"
    response = read_spectral_dataset(
        input_path=FIXTURES / "apply_csv_basic" / "data.csv",
        output_dir=output,
    )
    assert response["ok"] is True
    result = response["result"]
    assert result["status"] == "ready"
    assert result["X"] == "X.csv"
    assert result["data_contract"] == "data_contract.json"
    assert "stage_results" not in result
    assert "read_plan" not in result
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["shape"] == {"n_samples": 3, "n_features": 3}
    assert contract["label_status"] == "present"


def test_explicit_band_and_task_parameters_are_minimal_contract_fields(tmp_path: Path) -> None:
    output = tmp_path / "explicit_semantics"
    response = read_spectral_dataset(
        input_path=FIXTURES / "workflow_basic" / "data.csv",
        output_dir=output,
        band_unit="nm",
        band_type="wavelength",
        spectral_type="nir",
        task_type="classification",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["band_unit"] == "nm"
    assert contract["band_axis"]["unit"] == "nm"
    assert contract["band_axis"]["type"] == "wavelength"
    assert contract["spectral_type"] == "nir"
    assert contract["task_hint"] == "classification"


def test_explicit_layout_parameters_confirm_read_plan_in_contract(tmp_path: Path) -> None:
    output = tmp_path / "explicit_confirmed_layout"
    response = read_spectral_dataset(
        input_path=FIXTURES / "workflow_basic" / "data.csv",
        output_dir=output,
        sample_orientation="rows",
        sample_id_column_index=0,
        label_column="class",
        spectral_start_column="900 nm",
        spectral_end_column="1100 nm",
        band_unit="nm",
        band_type="wavelength",
        task_type="classification",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    confirmation = contract["reader_confirmation"]
    assert confirmation["status"] == "confirmed"
    assert confirmation["source"] == "explicit_cli_arguments"
    assert confirmation["sample_id_column"] == 0
    assert confirmation["label_column"] == "class"
    assert confirmation["spectral_columns"] == "900 nm..1100 nm"
    assert confirmation["band_unit"] == "nm"
    assert confirmation["band_type"] == "wavelength"


def test_cli_numeric_band_headers_are_treated_as_headers_before_indices(tmp_path: Path) -> None:
    source = tmp_path / "tablet_like.csv"
    source.write_text(
        "\n".join(
            [
                ",3600,3599,200,class",
                "10,0.11,0.12,0.13,0",
                "11,0.21,0.22,0.23,1",
                "12,0.31,0.32,0.33,0",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reader_output"
    script = REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "read_spectral_dataset.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--sample-orientation",
            "rows",
            "--sample-id-column-index",
            "0",
            "--label-column",
            "class",
            "--spectral-start-column",
            "3600",
            "--spectral-end-column",
            "200",
            "--band-type",
            "wavenumber",
            "--band-unit",
            "cm-1",
            "--task-type",
            "classification",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["shape"] == {"n_samples": 3, "n_features": 3}
    assert contract["sample_id_source"] == "source_first_column_empty_header"
    assert contract["reader_confirmation"]["source"] == "explicit_cli_arguments"
    assert (output / "X.csv").read_text(encoding="utf-8").splitlines()[0] == "3600,3599,200"


def test_wide_table_numeric_header_block_is_auto_detected(tmp_path: Path) -> None:
    source = tmp_path / "wide_tablet_like.csv"
    bands = [str(value) for value in range(3600, 3539, -1)]
    rows = [[f"S{i:03d}", *[round(i + j / 1000, 6) for j in range(len(bands))], str(i % 2)] for i in range(1, 5)]
    source.write_text(
        "\n".join([",".join(["", *bands, "class"]), *[",".join(map(str, row)) for row in rows]]),
        encoding="utf-8",
    )
    output = tmp_path / "auto_wide_reader"
    response = read_spectral_dataset(
        input_path=source,
        output_dir=output,
        task_type="classification",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["shape"] == {"n_samples": 4, "n_features": len(bands)}
    assert contract["sample_id_source"] == "source_first_column_empty_header"
    assert contract["band_axis"]["type"] == "wavenumber"
    assert contract["band_axis"]["unit"] == "cm-1"
    assert contract["label_status"] == "present"
    assert (output / "X.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(bands)


def test_invalid_wavenumber_unit_returns_schema_hint(tmp_path: Path) -> None:
    output = tmp_path / "invalid_unit"
    response = read_spectral_dataset(
        input_path=FIXTURES / "workflow_basic" / "data.csv",
        output_dir=output,
        sample_orientation="rows",
        sample_id_column_index=0,
        label_column="class",
        spectral_start_column="900 nm",
        spectral_end_column="1100 nm",
        band_unit="wavenumber",
        band_type="wavenumber",
        task_type="classification",
        confirm_read_plan=True,
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "READ_PLAN_SCHEMA_INVALID"
    details = response["errors"][0]["details"]
    assert details["field"] == "band_unit"
    assert details["received"] == "wavenumber"
    assert "cm-1" in details["allowed"]
    assert "use --band-type wavenumber --band-unit cm-1" in details["hint"]


def test_one_shot_reads_csv_columns_into_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "one_shot_columns"
    response = read_spectral_dataset(
        input_path=FIXTURES / "samples_as_columns_basic" / "spectra_columns.csv",
        output_dir=output,
        sample_orientation="columns",
        band_axis_column="band",
        spectral_columns=["sample_001"],
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["sample_orientation"] == "columns"
    assert contract["label_status"] == "absent"


def test_one_shot_reads_external_label_file_into_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "one_shot_external"
    response = read_spectral_dataset(
        input_path=FIXTURES / "external_label_rows_basic" / "spectra.csv",
        output_dir=output,
        sample_id_column="sample_id",
        label_file=FIXTURES / "external_label_rows_basic" / "labels.csv",
        label_column="Class",
        join_key="sample_id",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["label_status"] == "present"
    assert contract["metadata_status"] == "present"


def test_one_shot_reads_sample_folder_into_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "one_shot_folder"
    response = read_spectral_dataset(
        input_path=FIXTURES / "folder_one_file_per_sample",
        output_dir=output,
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["source_type"] == "folder"
    assert contract["label_status"] == "absent"


def test_folder_name_as_label_one_shot_reads_y_and_pattern_filtered_files(tmp_path: Path) -> None:
    output = tmp_path / "folder_label"
    response = read_spectral_dataset(
        input_path=FIXTURES / "folder_name_as_label_basic",
        output_dir=output,
        folder_name_as_label=True,
        sample_file_pattern="*/*.csv",
        sample_file_band_column="band",
        sample_file_value_column="absorbance",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["source_type"] == "folder"
    assert contract["label_status"] == "present"
    assert (output / "y.csv").read_text(encoding="utf-8").splitlines() == ["label", "class_A", "class_A", "class_B"]
    assert (output / "sample_ids.csv").read_text(encoding="utf-8").splitlines() == ["sample_id", "s001", "s002", "s003"]
    assert [line.split(",")[1] for line in (output / "band_axis.csv").read_text(encoding="utf-8").splitlines()[1:]] == ["900", "1000", "1100"]
    assert len((output / "X.csv").read_text(encoding="utf-8").splitlines()) == 4


def test_file_name_as_label_one_shot_uses_prefix_before_underscore(tmp_path: Path) -> None:
    output = tmp_path / "file_label"
    response = read_spectral_dataset(
        input_path=FIXTURES / "file_name_as_label_basic",
        output_dir=output,
        file_name_as_label=True,
        sample_file_pattern="*.csv",
        sample_file_band_column="band",
        sample_file_value_column="reflectance",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["label_status"] == "present"
    assert (output / "y.csv").read_text(encoding="utf-8").splitlines() == ["label", "A", "A", "B"]


def test_row_order_label_alignment_explicit_flag_reads_y_and_metadata(tmp_path: Path) -> None:
    fixture = FIXTURES / "row_order_label_alignment_basic"
    output = tmp_path / "row_order"
    response = read_spectral_dataset(
        input_path=fixture / "spectra.csv",
        output_dir=output,
        label_file=fixture / "labels.csv",
        label_column="Class",
        label_alignment="row_order",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["label_alignment"] == "row_order"
    assert contract["label_status"] == "present"
    assert (output / "y.csv").read_text(encoding="utf-8").splitlines() == ["Class", "A", "A", "B"]
    assert (output / "metadata.csv").read_text(encoding="utf-8").splitlines() == ["batch", "B1", "B1", "B2"]


def test_row_order_label_alignment_requires_explicit_flag(tmp_path: Path) -> None:
    fixture = FIXTURES / "row_order_alignment_without_explicit_flag"
    response = read_spectral_dataset(
        input_path=fixture / "spectra.csv",
        output_dir=tmp_path / "row_order_not_allowed",
        label_file=fixture / "labels.csv",
        label_column="Class",
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "ROW_ORDER_ALIGNMENT_NOT_ALLOWED"
    assert not (tmp_path / "row_order_not_allowed" / "X.csv").exists()


def test_row_order_label_alignment_length_mismatch_blocked(tmp_path: Path) -> None:
    fixture = FIXTURES / "row_order_label_alignment_length_mismatch"
    response = read_spectral_dataset(
        input_path=fixture / "spectra.csv",
        output_dir=tmp_path / "row_order_mismatch",
        label_file=fixture / "labels.csv",
        label_column="Class",
        allow_row_order_labels=True,
    )
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "ROW_ORDER_LENGTH_MISMATCH"
    assert not (tmp_path / "row_order_mismatch" / "X.csv").exists()


def test_mixed_folder_single_spectra_reads_minimal_output(tmp_path: Path) -> None:
    output = tmp_path / "mixed_single"
    response = read_spectral_dataset(
        input_path=FIXTURES / "mixed_folder_single_spectra",
        output_dir=output,
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source_type"] == "folder"
    assert contract["source"]["spectra_file"].endswith("spectra.csv")
    assert contract["label_status"] == "absent"


def test_mixed_folder_spectra_label_reads_y(tmp_path: Path) -> None:
    output = tmp_path / "mixed_label"
    response = read_spectral_dataset(
        input_path=FIXTURES / "mixed_folder_spectra_label",
        output_dir=output,
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source"]["label_file"].endswith("labels.csv")
    assert contract["label_status"] == "present"


def test_mixed_folder_spectra_label_metadata_merges_metadata(tmp_path: Path) -> None:
    output = tmp_path / "mixed_metadata"
    response = read_spectral_dataset(
        input_path=FIXTURES / "mixed_folder_spectra_label_metadata",
        output_dir=output,
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"})
    assert contract["source"]["metadata_file"].endswith("metadata.csv")
    assert contract["metadata_status"] == "present"
    rows = (output / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "batch,origin"
    assert rows[1:] == ["B1,north", "B1,south", "B2,east"]


def test_mixed_folder_band_axis_file_reads_axis(tmp_path: Path) -> None:
    output = tmp_path / "mixed_band_axis"
    response = read_spectral_dataset(
        input_path=FIXTURES / "mixed_folder_spectra_label_band_axis",
        output_dir=output,
        join_key="sample_id",
        label_column="Class",
        band_axis_column="band",
    )
    assert response["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source"]["band_axis_file"].endswith("bands.csv")
    assert contract["band_axis"]["type"] == "external_file"
    assert [line.split(",")[1] for line in (output / "band_axis.csv").read_text(encoding="utf-8").splitlines()[1:]] == ["900", "1000", "1100"]


def test_mixed_folder_multiple_spectra_candidates_needs_confirmation(tmp_path: Path) -> None:
    response = read_spectral_dataset(
        input_path=FIXTURES / "mixed_folder_multiple_spectra_candidates",
        output_dir=tmp_path / "mixed_multi_spectra",
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["spectra_file"]
    assert "--spectra-file" in response["result"]["suggested_arguments"]


def test_mixed_folder_multiple_label_candidates_needs_confirmation_and_explicit_override(tmp_path: Path) -> None:
    fixture = FIXTURES / "mixed_folder_multiple_label_candidates"
    response = read_spectral_dataset(
        input_path=fixture,
        output_dir=tmp_path / "mixed_multi_label",
        join_key="sample_id",
        label_column="Class",
    )
    assert response["ok"] is True
    assert response["result"]["status"] == "needs_confirmation"
    assert response["result"]["required_fields"] == ["label_file"]

    output = tmp_path / "mixed_multi_label_explicit"
    explicit = read_spectral_dataset(
        input_path=fixture,
        output_dir=output,
        spectra_file="spectra.csv",
        label_file="labels.csv",
        join_key="sample_id",
        label_column="Class",
    )
    assert explicit["ok"] is True
    contract = _assert_minimal_output(output, {"X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "data_contract.json"})
    assert contract["source"]["spectra_file"].endswith("spectra.csv")
    assert contract["source"]["label_file"].endswith("labels.csv")


def test_no_view_parameter_is_supported() -> None:
    import inspect

    assert "view" not in inspect.signature(read_spectral_dataset).parameters
    assert "mode" not in inspect.signature(read_spectral_dataset).parameters
    assert "read_plan" not in inspect.signature(read_spectral_dataset).parameters


def test_output_dir_existing_without_overwrite_fails_and_overwrite_succeeds(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    blocked = read_spectral_dataset(
        input_path=FIXTURES / "workflow_basic" / "data.csv",
        output_dir=output_dir,
    )
    assert blocked["ok"] is False
    assert blocked["errors"][0]["code"] == "OUTPUT_DIR_EXISTS"

    rebuilt = read_spectral_dataset(
        input_path=FIXTURES / "workflow_basic" / "data.csv",
        output_dir=output_dir,
        overwrite=True,
    )
    assert rebuilt["ok"] is True
    assert not (output_dir / "keep.txt").exists()
    assert (output_dir / "X.csv").exists()


def test_reader_cli_and_fallback_outputs_json(tmp_path: Path) -> None:
    script = REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "read_spectral_dataset.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(FIXTURES / "workflow_basic" / "data.csv"),
            "--output-dir",
            str(tmp_path / "cli_read"),
            "--band-unit",
            "cm-1",
            "--band-type",
            "wavenumber",
            "--spectral-type",
            "nir",
            "--task-type",
            "classification",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["tool"] == "read_spectral_dataset"
    assert payload["result"]["status"] == "ready"
    assert (tmp_path / "cli_read" / "data_contract.json").exists()
    cli_contract = json.loads((tmp_path / "cli_read" / "data_contract.json").read_text(encoding="utf-8"))
    assert cli_contract["band_unit"] == "cm-1"
    assert cli_contract["band_axis"]["type"] == "wavenumber"
    assert cli_contract["spectral_type"] == "nir"

    fallback = REPO_ROOT / "scripts" / "reader" / "read_spectral_dataset.py"
    fallback_completed = subprocess.run(
        [
            sys.executable,
            str(fallback),
            "--input",
            str(FIXTURES / "workflow_basic" / "data.csv"),
            "--output-dir",
            str(tmp_path / "fallback_full"),
            "--band-unit",
            "nm",
            "--band-type",
            "wavelength",
            "--spectral-type",
            "vis-nir",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    fallback_payload = json.loads(fallback_completed.stdout)
    assert fallback_payload["ok"] is True
    assert fallback_payload["result"]["data_contract"] == "data_contract.json"
    assert (tmp_path / "fallback_full" / "data_contract.json").exists()
    fallback_contract = json.loads((tmp_path / "fallback_full" / "data_contract.json").read_text(encoding="utf-8"))
    assert fallback_contract["band_unit"] == "nm"
    assert fallback_contract["band_axis"]["type"] == "wavelength"
    assert fallback_contract["spectral_type"] == "vis-nir"

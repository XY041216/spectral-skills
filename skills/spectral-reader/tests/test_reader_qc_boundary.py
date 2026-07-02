from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _contract(output: Path) -> dict:
    return json.loads((output / "data_contract.json").read_text(encoding="utf-8"))


def test_missing_x_values_are_preserved_for_qc(tmp_path: Path) -> None:
    source = tmp_path / "missing_x.csv"
    _write_csv(
        source,
        [
            ["sample_id", "900", "1000", "Class"],
            ["S001", "0.10", "", "A"],
            ["S002", "NA", "0.23", "B"],
        ],
    )

    output = tmp_path / "out"
    response = read_spectral_dataset(input_path=source, output_dir=output)

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    assert _contract(output)["shape"] == {"n_samples": 2, "n_features": 2}
    lines = (output / "X.csv").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "0.1,nan"
    assert lines[2] == "nan,0.23"
    assert _contract(output)["missing_value_status"] == "present"


def test_invalid_non_missing_x_value_still_blocks(tmp_path: Path) -> None:
    source = tmp_path / "bad_x.csv"
    _write_csv(source, [["sample_id", "900", "1000"], ["S001", "0.10", "ERR"]])

    output = tmp_path / "out"
    response = read_spectral_dataset(input_path=source, output_dir=output)

    assert response["ok"] is False
    assert response["errors"][0]["code"] in {"X_NON_NUMERIC", "SPECTRAL_BLOCK_NON_NUMERIC"}
    assert not (output / "X.csv").exists()


def test_missing_sample_id_requires_confirmation_then_generates(tmp_path: Path) -> None:
    source = tmp_path / "missing_sample_id.csv"
    _write_csv(source, [["sample_id", "900", "1000"], ["S001", "0.10", "0.11"], ["", "0.20", "0.21"]])

    first_output = tmp_path / "first"
    first = read_spectral_dataset(input_path=source, output_dir=first_output)

    assert first["ok"] is True
    assert first["result"]["status"] == "needs_confirmation"
    assert "missing_sample_id_policy" in first["result"]["required_fields"]
    assert not (first_output / "X.csv").exists()

    second_output = tmp_path / "second"
    second = read_spectral_dataset(
        input_path=source,
        output_dir=second_output,
        missing_sample_id_policy="generate",
    )

    assert second["ok"] is True
    assert second["result"]["status"] == "ready"
    sample_ids = (second_output / "sample_ids.csv").read_text(encoding="utf-8").splitlines()
    assert sample_ids == ["sample_id", "S001", "generated_sample_001"]
    assert _contract(second_output)["sample_id_status"] == "partially_generated_after_confirmation"


def test_external_label_missing_sample_still_blocks(tmp_path: Path) -> None:
    spectra = tmp_path / "spectra.csv"
    labels = tmp_path / "labels.csv"
    _write_csv(spectra, [["sample_id", "900", "1000"], ["S001", "0.10", "0.11"], ["S002", "0.20", "0.21"]])
    _write_csv(labels, [["sample_id", "Class"], ["S001", "A"]])

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=spectra,
        output_dir=output,
        label_file=labels,
        join_key="sample_id",
        label_column="Class",
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "MISSING_REQUIRED_LABELS"
    assert not (output / "X.csv").exists()


def test_external_band_axis_length_mismatch_still_blocks(tmp_path: Path) -> None:
    spectra = tmp_path / "spectra.csv"
    bands = tmp_path / "bands.csv"
    _write_csv(spectra, [["sample_id", "f1", "f2"], ["S001", "0.10", "0.11"]])
    _write_csv(bands, [["band"], [900]])

    output = tmp_path / "out"
    response = read_spectral_dataset(
        input_path=spectra,
        output_dir=output,
        sample_id_column="sample_id",
        spectral_start_column="f1",
        spectral_end_column="f2",
        band_axis_file=bands,
        band_axis_column="band",
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "BAND_AXIS_LENGTH_MISMATCH"
    assert not (output / "X.csv").exists()


def test_duplicate_sample_ids_are_reader_blocking(tmp_path: Path) -> None:
    source = tmp_path / "duplicate_sample_ids.csv"
    _write_csv(source, [["sample_id", "900", "1000"], ["S001", "0.10", "0.11"], ["S001", "0.20", "0.21"]])

    output = tmp_path / "out"
    response = read_spectral_dataset(input_path=source, output_dir=output)

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "DUPLICATE_SAMPLE_IDS"
    assert not (output / "X.csv").exists()

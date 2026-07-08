from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "skills" / "spectral-reader" / "fixtures"
WORKSPACE_ROOT = REPO_ROOT.parent
CORPUS = WORKSPACE_ROOT / "spectral_test_corpus" / "valid_cases" / "07_csv_external_band_axis_bundle"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


def _contract(output: Path) -> dict:
    return json.loads((output / "data_contract.json").read_text(encoding="utf-8"))


def _write_wide_wavenumber_table(path: Path) -> None:
    bands = list(range(3600, 199, -1))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *bands, "class"])
        for sample_idx in range(6):
            writer.writerow([sample_idx, *[sample_idx + band / 10000 for band in bands], sample_idx % 2])


def test_one_shot_reads_utf8_bom_csv(tmp_path: Path) -> None:
    data_path = tmp_path / "bom.csv"
    data_path.write_text((FIXTURES / "workflow_basic" / "data.csv").read_text(encoding="utf-8"), encoding="utf-8-sig")

    output = tmp_path / "bom_output"
    response = read_spectral_dataset(input_path=data_path, output_dir=output)

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    assert _contract(output)["shape"] == {"n_samples": 3, "n_features": 3}
    assert (output / "X.csv").exists()
    assert not (output / "read_plan.json").exists()


def test_one_shot_reads_wide_wavenumber_table_without_truncating_auto_detection(tmp_path: Path) -> None:
    data_path = tmp_path / "wide.csv"
    _write_wide_wavenumber_table(data_path)

    automatic_output = tmp_path / "automatic_output"
    automatic = read_spectral_dataset(
        input_path=data_path,
        output_dir=automatic_output,
        sample_orientation="rows",
        label_column="class",
        task_type="classification",
    )

    assert automatic["ok"] is True
    assert automatic["result"]["status"] == "ready"
    assert automatic["result"]["n_features"] == 3401
    assert _contract(automatic_output)["shape"] == {"n_samples": 6, "n_features": 3401}

    output = tmp_path / "confirmed_output"
    confirmed = read_spectral_dataset(
        input_path=data_path,
        output_dir=output,
        sample_orientation="rows",
        label_column="class",
        spectral_start_column="3600",
        spectral_end_column="200",
        task_type="classification",
    )

    assert confirmed["ok"] is True
    assert confirmed["result"]["status"] == "ready"
    assert confirmed["result"]["n_features"] == 3401
    assert _contract(output)["shape"] == {"n_samples": 6, "n_features": 3401}


def test_one_shot_uses_spectral_column_indexes_before_numeric_header_names(tmp_path: Path) -> None:
    data_path = tmp_path / "wide.csv"
    _write_wide_wavenumber_table(data_path)

    output = tmp_path / "indexed_output"
    response = read_spectral_dataset(
        input_path=data_path,
        output_dir=output,
        sample_orientation="rows",
        sample_id_column_index=0,
        label_column="class",
        spectral_start_column=1,
        spectral_end_column=3401,
        task_type="classification",
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    assert response["result"]["n_features"] == 3401
    assert _contract(output)["shape"] == {"n_samples": 6, "n_features": 3401}


def test_cli_accepts_spectral_column_indexes_without_numeric_header_collision(tmp_path: Path) -> None:
    data_path = tmp_path / "wide.csv"
    _write_wide_wavenumber_table(data_path)
    output = tmp_path / "cli_indexed_output"
    script = REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "read_spectral_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(data_path),
            "--output-dir",
            str(output),
            "--sample-orientation",
            "rows",
            "--sample-id-column-index",
            "0",
            "--label-column",
            "class",
            "--spectral-start-column-index",
            "1",
            "--spectral-end-column-index",
            "3401",
            "--task-type",
            "classification",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["result"]["n_features"] == 3401
    assert _contract(output)["shape"] == {"n_samples": 6, "n_features": 3401}


def test_cli_accepts_empty_header_sample_id_by_column_index(tmp_path: Path) -> None:
    data_path = tmp_path / "wide.csv"
    _write_wide_wavenumber_table(data_path)
    output = tmp_path / "cli_output"
    script = REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "read_spectral_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(data_path),
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
            "--task-type",
            "classification",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert (output / "sample_ids.csv").read_text(encoding="utf-8").splitlines()[:4] == ["sample_id", "0", "1", "2"]
    contract = _contract(output)
    assert contract["sample_id_status"] == "original"
    assert contract["sample_id_source"] == "source_first_column_empty_header"


def test_one_shot_accepts_chinese_absolute_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "中文路径"
    data_dir.mkdir()
    data_path = data_dir / "数据.csv"
    data_path.write_text((FIXTURES / "workflow_basic" / "data.csv").read_text(encoding="utf-8"), encoding="utf-8")

    output = tmp_path / "中文输出"
    response = read_spectral_dataset(input_path=data_path, output_dir=output)

    assert response["ok"] is True
    assert response["result"]["n_samples"] == 3
    assert _contract(output)["source"]["input"].endswith("数据.csv")
    assert (output / "data_contract.json").exists()


def test_one_shot_resolves_relative_input_with_source_base_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "数据目录"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text((FIXTURES / "workflow_basic" / "data.csv").read_text(encoding="utf-8"), encoding="utf-8")

    output = tmp_path / "relative_output"
    response = read_spectral_dataset(
        input_path="data.csv",
        source_base_dir=str(data_dir),
        output_dir=output,
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    assert _contract(output)["shape"]["n_samples"] == 3
    assert (output / "X.csv").exists()
    assert not (output / "package_manifest.json").exists()


def test_one_shot_reads_external_band_axis_bundle_without_manual_spectral_columns(tmp_path: Path) -> None:
    if not CORPUS.exists():
        import pytest

        pytest.skip("real corpus fixture not available")

    output = tmp_path / "bundle_output"
    response = read_spectral_dataset(
        input_path=CORPUS,
        source_base_dir=str(CORPUS),
        output_dir=output,
        spectra_file="spectra_only.csv",
        sample_orientation="rows",
        sample_ids_file="sample_ids.csv",
        label_file="labels.csv",
        metadata_file="metadata.csv",
        join_key="sample_id",
        label_column="class",
        band_axis_file="band_axis.csv",
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "ready"
    contract = _contract(output)
    assert contract["shape"] == {"n_samples": 12, "n_features": 17}
    assert contract["sample_id_status"] == "original"
    assert contract["label_status"] == "present"
    assert contract["metadata_status"] == "present"
    assert contract["band_axis"]["type"] == "external_file"
    assert (output / "sample_ids.csv").read_text(encoding="utf-8").splitlines()[:4] == ["sample_id", "S001", "S002", "S003"]

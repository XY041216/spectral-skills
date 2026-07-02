"""Write QC outputs in the reader standard package shape."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import write_json_file

from .contract import build_qc_data_contract
from .io import SpectralQCPackage


class QCPackageWriteError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def write_standard_package(
    package: SpectralQCPackage,
    *,
    output_dir: str | Path,
    parent_contract: str,
    qc_summary: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise QCPackageWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    _write_rows(root / "X.csv", [package.feature_names, *package.X])
    _write_rows(root / "sample_ids.csv", [["sample_id"], *[[sample_id] for sample_id in package.sample_ids]])
    _write_rows(root / "band_axis.csv", _band_rows(package))
    written = ["X.csv", "sample_ids.csv", "band_axis.csv"]
    if package.y is not None:
        _write_rows(root / "y.csv", [package.y_header or ["y"], *package.y])
        written.append("y.csv")
    if package.metadata is not None:
        _write_dict_rows(root / "metadata.csv", package.metadata)
        written.append("metadata.csv")
    contract = build_qc_data_contract(package, parent_contract=parent_contract, qc_summary=qc_summary)
    write_json_file(root / "data_contract.json", contract, ensure_ascii=False)
    written.append("data_contract.json")
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": written,
        "data_contract": "data_contract.json",
        "shape": {"n_samples": package.n_samples, "n_features": package.n_features},
        "qc_summary": contract["qc_summary"],
        "next_step_hint": "Use this output directory as the standard package for downstream skills.",
    }


def _band_rows(package: SpectralQCPackage) -> list[list[Any]]:
    header = package.band_axis_header or ["index", "value", "unit"]
    if len(header) >= 2 and header[1].lower() == "value":
        rows = []
        for idx, value in enumerate(package.band_axis):
            row = [idx, value]
            while len(row) < len(header):
                row.append("")
            rows.append(row)
        return [header, *rows]
    return [header, *[[value] for value in package.band_axis]]


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

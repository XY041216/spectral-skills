"""Spectral-reader MCP server skeleton."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - environment dependent
    FastMCP = None

from spectral_core.reader.response import ok_response
from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION
from spectral_core.reader.workflow import read_spectral_dataset as core_read_spectral_dataset


if FastMCP is not None:
    mcp = FastMCP("spectral-reader")
else:
    mcp = None


def server_health() -> dict[str, Any]:
    """Check spectral-reader installation and runtime data-reading availability."""

    return ok_response(
        "server_health",
        {
            "status": "skeleton",
            "core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "mcp_available": FastMCP is not None,
            "primary_entrypoint": "read_spectral_dataset",
            "tools": ["server_health", "read_spectral_dataset"],
        },
        backend="mcp",
    )


def read_spectral_dataset(
    input_path: str | None = None,
    output_dir: str | None = None,
    source_base_dir: str | None = None,
    sheet: str | None = None,
    sheet_index: int | None = None,
    spectral_sheet: str | None = None,
    label_sheet: str | None = None,
    sample_orientation: str | None = None,
    sample_id_column: str | None = None,
    label_column: str | None = None,
    target_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    spectral_columns: list[str] | None = None,
    band_axis_column: str | int | None = None,
    x_var: str | None = None,
    y_var: str | None = None,
    sample_ids_var: str | None = None,
    band_axis_var: str | None = None,
    metadata_var: str | None = None,
    x_path: str | None = None,
    y_path: str | None = None,
    sample_ids_path: str | None = None,
    band_axis_path: str | None = None,
    metadata_path: str | None = None,
    label_file: str | None = None,
    join_key: str | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    skiprows: int | None = None,
    header_row: int | None = None,
    overwrite: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Read an input spectral data file or folder into X.csv, optional y.csv, sample_ids.csv, band_axis.csv, optional metadata.csv, and data_contract.json."""

    return core_read_spectral_dataset(
        input_path=input_path,
        output_dir=output_dir,
        source_base_dir=source_base_dir,
        sheet=sheet,
        sheet_index=sheet_index,
        spectral_sheet=spectral_sheet,
        label_sheet=label_sheet,
        sample_orientation=sample_orientation,
        sample_id_column=sample_id_column,
        label_column=label_column,
        target_columns=target_columns,
        metadata_columns=metadata_columns,
        spectral_columns=spectral_columns,
        band_axis_column=band_axis_column,
        x_var=x_var,
        y_var=y_var,
        sample_ids_var=sample_ids_var,
        band_axis_var=band_axis_var,
        metadata_var=metadata_var,
        x_path=x_path,
        y_path=y_path,
        sample_ids_path=sample_ids_path,
        band_axis_path=band_axis_path,
        metadata_path=metadata_path,
        label_file=label_file,
        join_key=join_key,
        delimiter=delimiter,
        encoding=encoding,
        skiprows=skiprows,
        header_row=header_row,
        overwrite=overwrite,
        strict=strict,
        backend="mcp",
    )


if mcp is not None:
    mcp.tool()(server_health)
    mcp.tool()(read_spectral_dataset)


def main() -> None:
    if mcp is None:
        raise RuntimeError("MCP SDK is not installed; use scripts fallback for the skeleton stage.")
    mcp.run()


if __name__ == "__main__":
    main()

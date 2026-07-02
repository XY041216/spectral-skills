"""Global fallback CLI for the spectral-reader one-shot reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.reader.workflow import read_spectral_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read spectral data into standard downstream-ready CSV files.")
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-base-dir")
    parser.add_argument("--spectra-file")
    parser.add_argument("--sheet")
    parser.add_argument("--sheet-index", type=int)
    parser.add_argument("--spectral-sheet")
    parser.add_argument("--label-sheet")
    parser.add_argument("--sample-orientation", choices=["rows", "columns"])
    parser.add_argument("--sample-id-column")
    parser.add_argument("--sample-id-column-index", type=int)
    parser.add_argument("--sample-id-column-position", type=int)
    parser.add_argument("--label-column")
    parser.add_argument("--target-columns")
    parser.add_argument("--metadata-columns")
    parser.add_argument("--spectral-columns")
    parser.add_argument("--band-axis-column")
    parser.add_argument("--x-var")
    parser.add_argument("--y-var")
    parser.add_argument("--sample-ids-var")
    parser.add_argument("--sample-ids-file")
    parser.add_argument("--band-axis-var")
    parser.add_argument("--metadata-var")
    parser.add_argument("--x-path")
    parser.add_argument("--y-path")
    parser.add_argument("--sample-ids-path")
    parser.add_argument("--band-axis-path")
    parser.add_argument("--metadata-path")
    parser.add_argument("--label-file")
    parser.add_argument("--metadata-file")
    parser.add_argument("--join-key")
    parser.add_argument("--folder-name-as-label", action="store_true")
    parser.add_argument("--file-name-as-label", action="store_true")
    parser.add_argument("--sample-file-pattern")
    parser.add_argument("--sample-file-value-column")
    parser.add_argument("--sample-file-band-column")
    parser.add_argument("--label-alignment", choices=["sample_id", "filename", "folder_name", "row_order"])
    parser.add_argument("--allow-row-order-labels", action="store_true")
    parser.add_argument("--allow-generated-sample-ids-for-missing", action="store_true")
    parser.add_argument("--confirm-generate-missing-sample-ids", action="store_true")
    parser.add_argument("--missing-sample-id-policy", choices=["blocked", "generate"])
    parser.add_argument("--missing-value-tokens")
    parser.add_argument("--delimiter")
    parser.add_argument("--encoding")
    parser.add_argument("--skiprows", type=int)
    parser.add_argument("--header-row", type=int)
    parser.add_argument("--header-rows")
    parser.add_argument("--data-start-row", type=int)
    parser.add_argument("--data-end-row", type=int)
    parser.add_argument("--spectral-start-column")
    parser.add_argument("--spectral-end-column")
    parser.add_argument("--band-axis-file")
    parser.add_argument("--band-unit")
    parser.add_argument("--band-type")
    parser.add_argument("--spectral-type")
    parser.add_argument("--task-type")
    parser.add_argument("--confirm-read-plan", action="store_true")
    parser.add_argument("--auto-folder", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    response = read_spectral_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        source_base_dir=args.source_base_dir,
        spectra_file=args.spectra_file,
        sheet=args.sheet,
        sheet_index=args.sheet_index,
        spectral_sheet=args.spectral_sheet,
        label_sheet=args.label_sheet,
        sample_orientation=args.sample_orientation,
        sample_id_column=args.sample_id_column,
        sample_id_column_index=args.sample_id_column_index if args.sample_id_column_index is not None else args.sample_id_column_position,
        label_column=args.label_column,
        target_columns=_split_csv_arg(args.target_columns),
        metadata_columns=_split_csv_arg(args.metadata_columns),
        spectral_columns=_split_csv_arg(args.spectral_columns),
        band_axis_column=_parse_column_arg(args.band_axis_column),
        x_var=args.x_var,
        y_var=args.y_var,
        sample_ids_var=args.sample_ids_var,
        sample_ids_file=args.sample_ids_file,
        band_axis_var=args.band_axis_var,
        metadata_var=args.metadata_var,
        x_path=args.x_path,
        y_path=args.y_path,
        sample_ids_path=args.sample_ids_path,
        band_axis_path=args.band_axis_path,
        metadata_path=args.metadata_path,
        label_file=args.label_file,
        metadata_file=args.metadata_file,
        join_key=args.join_key,
        folder_name_as_label=args.folder_name_as_label,
        file_name_as_label=args.file_name_as_label,
        sample_file_pattern=args.sample_file_pattern,
        sample_file_value_column=_parse_column_arg(args.sample_file_value_column),
        sample_file_band_column=_parse_column_arg(args.sample_file_band_column),
        label_alignment=args.label_alignment,
        allow_row_order_labels=args.allow_row_order_labels,
        allow_generated_sample_ids_for_missing=args.allow_generated_sample_ids_for_missing or args.confirm_generate_missing_sample_ids,
        missing_sample_id_policy=args.missing_sample_id_policy,
        missing_value_tokens=_split_csv_arg(args.missing_value_tokens),
        delimiter=args.delimiter,
        encoding=args.encoding,
        skiprows=args.skiprows,
        header_row=args.header_row,
        header_rows=_split_int_arg(args.header_rows),
        data_start_row=args.data_start_row,
        data_end_row=args.data_end_row,
        spectral_start_column=args.spectral_start_column,
        spectral_end_column=args.spectral_end_column,
        band_axis_file=args.band_axis_file,
        band_unit=args.band_unit,
        band_type=args.band_type,
        spectral_type=args.spectral_type,
        task_type=args.task_type,
        confirm_read_plan=args.confirm_read_plan,
        auto_folder=args.auto_folder,
        overwrite=args.overwrite,
        strict=args.strict,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_column_arg(value: str | None) -> str | int | None:
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    return text


def _split_int_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

# Reading Semantics Patterns

These patterns describe the semantics that `read_spectral_dataset` needs before
it can read safely. Internal settings may represent them, but users and
downstream skills consume only the standard CSV files and `data_contract.json`.

## Ready To Read

The reader is ready to execute when:

- source path is readable;
- input kind and sample orientation are known;
- spectral columns are explicit;
- label/target semantics are confirmed when requested;
- required confirmations are resolved.

After this point, reader output should either be reliable or blocked by a hard
assertion. Do not emit low-confidence ready output.

## Needs Confirmation

Return `needs_confirmation` when the input is plausible but the user still
needs to confirm preamble skip, spectral boundaries, label column, sample IDs,
orientation, or external alignment.

## Blocked

Return `blocked` for missing source, unsupported layout, unknown matrix, missing
required label/target, absent columns, or unresolvable alignment.

## One-Confirmation Flow

If the only unresolved item is a spectral boundary confirmation, ask the user
once. After the user confirms, run `read_spectral_dataset` and write only the
standard output files.

## Samples As Columns Pattern

Use this pattern for CSV/TSV/TXT tables whose first column is the band axis and
whose remaining columns are samples:

```json
{
  "read_mode": "matrix_file",
  "sample_orientation": "columns",
  "samples_as_columns": {
    "enabled": true,
    "band_axis_column": "band",
    "sample_id_source": "column_headers",
    "data_start_row": 1,
    "sample_start_column": 1,
    "transpose_required": true
  },
  "task_hint": "unsupervised"
}
```

The output writes `X.csv` as samples by features.

## External Label File Pattern

Use this pattern for spectra rows plus a separate label file:

```json
{
  "sample_orientation": "rows",
  "sample_id": {"column": "sample_id"},
  "label": {"source": "external_file", "column": "Class"},
  "label_file": {
    "path": "labels.csv",
    "sample_id_column": "sample_id",
    "label_column": "Class",
    "metadata_columns": ["batch"]
  },
  "alignment_plan": {
    "join_key": "sample_id",
    "join_type": "left",
    "preserve_spectrum_order": true,
    "duplicate_policy": "blocked",
    "missing_label_policy": "blocked"
  }
}
```

Block duplicate label keys or missing required labels for supervised reads.

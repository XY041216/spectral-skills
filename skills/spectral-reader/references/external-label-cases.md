# External Label Cases

These examples describe external label alignment during spectral data reading.
They are not hard rules.

## External Label Alignment Rules

- Exact ID join: `alignment_plan.method=exact_id`.
- Normalized ID join: record the normalization rule internally.
- Row-order join: require explicit confirmation.
- Missing or duplicate labels: keep the plan `blocked` for supervised tasks.
- Label metadata in the external file should be recorded separately from
  `label.column` or `target.column`.

## Exact ID Alignment

Both spectra and label file contain unique sample IDs. Match exactly and
preserve spectral sample order.

## Cleaned ID Alignment

IDs differ by case, prefixes, hyphens, underscores, or leading zeros. Mark
alignment provisional and ask for confirmation.

## Row Order Alignment

Row counts match but no reliable IDs exist. Always confirm before using labels.

## Label Metadata

Label files may contain both target columns and metadata columns. Record each
role separately.

## Missing Or Duplicate Labels

Missing labels, unmatched IDs, or duplicate IDs should produce provisional or
blocked status depending on severity and task.

## Current Supported Execution

CSV/TSV/TXT external label files are supported when exact sample-id alignment
is confirmed. The reader preserves the spectra file order, writes aligned
`y.csv`, and merges `label_file.metadata_columns` into
`metadata.csv`.

Example alignment:

```json
{
  "label_file": {
    "path": "labels.csv",
    "sample_id_column": "sample_id",
    "label_column": "Class",
    "metadata_columns": ["batch"]
  },
  "alignment_plan": {
    "join_key": "sample_id",
    "allow_unmatched_spectra": false,
    "allow_unmatched_labels": true,
    "duplicate_policy": "blocked",
    "missing_label_policy": "blocked"
  }
}
```

If `labels.csv` has duplicate `sample_id`, return blocked with
`DUPLICATE_LABEL_KEYS`. If a spectra sample lacks a required label, return
blocked with `MISSING_REQUIRED_LABELS`; do not emit partially supervised
`y.csv`.
blocked with `MISSING_REQUIRED_LABELS`.

# External Label File

Use this fragment when spectra and labels/targets/metadata are in separate
CSV/TSV/TXT files.

## Evidence

Folder or multi-file preview may show:

- a spectra-like file with spectral columns;
- a label-like file name;
- shared `sample_id` column candidates;
- label-like columns such as `Class`;
- metadata-like columns such as `batch`.

Preview does not join labels. It only records evidence for the Agent.

## Confirmations

Require confirmation for:

- spectral file;
- label file;
- join key;
- label or target column;
- metadata columns from the label file;
- alignment policy.

## Read Plan Fields

Use exact ID alignment by default:

```json
{
  "label_file": {
    "path": "labels.csv",
    "file_type": ".csv",
    "delimiter": ",",
    "sample_id_column": "sample_id",
    "label_column": "Class",
    "metadata_columns": ["batch"],
    "required": true
  },
  "alignment_plan": {
    "join_key": "sample_id",
    "join_type": "left",
    "preserve_spectrum_order": true,
    "allow_unmatched_spectra": false,
    "allow_unmatched_labels": true,
    "duplicate_policy": "blocked",
    "missing_label_policy": "blocked"
  }
}
```

The executor preserves the spectra file sample order and writes aligned `y.csv`
plus merged `metadata.csv`.

## Blocked Conditions

Block supervised reads when the join key is missing, label keys are duplicate,
required spectra samples have no labels, the label column is absent, or the
spectra file lacks sample IDs.

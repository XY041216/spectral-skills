# Data Contract Examples

## Ready Contract

```json
{
  "status": "ready",
  "X": "X.csv",
  "y": "y.csv",
  "sample_ids": "sample_ids.csv",
  "band_axis": "band_axis.csv",
  "metadata": "metadata.csv",
  "n_samples": 12,
  "n_features": 17,
  "sample_orientation": "rows",
  "label_status": "available",
  "metadata_status": "available",
  "task_hint": "classification",
  "band_unit": "cm-1",
  "source_type": "csv"
}
```

## Unlabeled Contract

```json
{
  "status": "ready",
  "X": "X.csv",
  "y": null,
  "sample_ids": "sample_ids.csv",
  "band_axis": "band_axis.csv",
  "metadata": null,
  "n_samples": 3,
  "n_features": 256,
  "sample_orientation": "columns",
  "label_status": "none",
  "metadata_status": "none",
  "task_hint": "unknown",
  "band_unit": "nm",
  "source_type": "csv"
}
```

## Output Layout

```text
output/
  X.csv
  y.csv
  sample_ids.csv
  band_axis.csv
  metadata.csv
  data_contract.json
```

Do not include confidence scores, full read plans, preview evidence,
validation reports, stage results, logs, summaries, package manifests, or
internal refs in the final Data Contract.

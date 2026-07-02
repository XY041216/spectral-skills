# Reader Output Contract

The final reader output directory is flat and minimal:

```
output/
  X.csv
  y.csv
  sample_ids.csv
  band_axis.csv
  metadata.csv
  data_contract.json
```

`y.csv`, `sample_ids.csv`, and `metadata.csv` are written only when available
or meaningful. If labels are absent, omit `y.csv` and set
`label_status: none` in `data_contract.json`.

Do not write `package_manifest.json`, `summary.json`, `_internal/`,
`preview_report.json`, `read_plan.json`, `profile_summary.json`,
`validation_report.json`, logs, decision traces, or confidence drafts as final
outputs.

## Minimal Data Contract

`data_contract.json` contains only downstream-useful fields:

- `status`
- `X`
- `y`
- `sample_ids`
- `band_axis`
- `metadata`
- `n_samples`
- `n_features`
- `sample_orientation`
- `label_status`
- `task_hint`
- `band_unit`
- `source_type`

Do not include confidence scores, full read plans, preview evidence,
validation details, stage results, internal refs, debug refs, logs, or package
manifests.

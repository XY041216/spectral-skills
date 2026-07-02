# Output Contract

Write a new standard spectral package with transformed `X.csv` and unchanged
sample alignment.

Required output:

- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- `data_contract.json`
- `preprocess_state.json`

Write `y.csv` and `metadata.csv` only when present in the input package.

`data_contract.json` must include `processing_stage: preprocess`,
`parent_contract`, `split_contract`, and compact `preprocess_summary`.

`preprocess_state.json` records method sequence, parameters, and train-fitted
state needed for reproducibility. Do not write reports, logs, debug folders, or
copied train/test matrices by default.

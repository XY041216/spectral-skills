# Standard Package Input

Read only a `spectral-reader` or `spectral-qc` standard package.

Required files:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`

Optional files:

- `y.csv`
- `metadata.csv`

Resolve file paths from `data_contract.json` when present. Do not depend on
reader or QC intermediate files.

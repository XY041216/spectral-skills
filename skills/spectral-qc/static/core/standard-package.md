# Standard Package

QC input is the standard package written by `spectral-reader`.

Required files:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`

Optional files:

- `y.csv`
- `metadata.csv`

When QC writes a modified dataset, it writes the same filenames to a new output
directory under `cleaned_package/`. Do not create a second handoff format.

Observation-only QC writes only `qc_result.json` by default.

If no cleaning is applied, `qc_result.json` must set:

- `output_package: null`
- `next_package_for_downstream: <input_package>`

If cleaning is applied, `qc_result.json` must set:

- `output_package: <cleaned_package>`
- `next_package_for_downstream: <cleaned_package>`

## Integrity Checks

Before any QC computation, verify:

- `X.csv` is rectangular and sample by band.
- `sample_ids.csv` row count equals the row count of `X.csv`.
- `band_axis.csv` length equals the column count of `X.csv`.
- `y.csv`, when present, aligns to `X.csv` by row count.
- `metadata.csv`, when present, aligns to `X.csv` by row count.
- file references in `data_contract.json` resolve inside the package.

If package integrity fails, block QC and send the user back to reader or ask for
a corrected package. Do not repair reader-level structural errors silently.

## Contract Updates

Keep QC additions compact and downstream-useful:

- record that the package has passed QC checks in `qc_result.json` or was
  edited after confirmation;
- record high-level actions such as removed sample count or imputation method;
- after confirmed band or sample deletion, update `shape.n_samples`,
  `shape.n_features`, top-level `n_samples`, top-level `n_features`, and
  `band_axis.count` to match the written files;
- keep warnings concise.

Do not store full candidate matrices, full algorithm traces, logs, or fixed
reports in `data_contract.json`.

# Excel / ODS Multi Sheet Cases

## Single Sheet Rows

Workbook sheet:

`sample_id, 900, 1000, 1100, Class`

Read as rows. Output `X.csv`, `sample_ids.csv`, `band_axis.csv`, `y.csv`, and
`data_contract.json`.

## Single Sheet Columns

Workbook sheet:

`band, S001, S002, S003`

Read as columns with `--sample-orientation columns --band-axis-column band`.
Output samples by features after transpose.

## Spectra Sheet + Label Sheet

Use `--spectral-sheet Spectra --label-sheet Labels --join-key sample_id`.
Align labels to the spectral sheet order. Metadata columns from the label sheet
may be merged into `metadata.csv`.

## Multiple Candidate Sheets

If more than one sheet could be spectral data and no sheet is specified, return
`needs_confirmation` with `--spectral-sheet` suggestions. Do not write output
files.

## Duplicate Key Blocked

If the label sheet has duplicate join keys, return blocked with
`DUPLICATE_LABEL_KEYS`.

## Missing Label Blocked

If a required label is missing for any spectral sample, return blocked with
`MISSING_REQUIRED_LABELS`.

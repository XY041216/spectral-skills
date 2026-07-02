# Missing Value Cases

## Missing X Cells

Blank cells, `NA`, `N/A`, `NaN`, `NULL`, `None`, `missing`, `--`, `-`, `.`,
`Inf`, and `-Inf` inside confirmed spectral columns are read as missing values.
The reader keeps the sample by band matrix rectangular and writes minimal
standard output.

## Invalid X Text

Values such as `error`, `bad`, `abc`, unit strings, label text, or header
fragments inside X are not missing tokens. Return `blocked` and ask the user to
check table boundaries, spectral columns, header rows, or missing-token rules.

## Embedded Missing Label Or Target

If a label or target column in the main table has missing cells, keep the row
and write `y.csv` with missing values preserved. Regression targets must still
be numeric or missing.

## External Labels Missing Samples

If an external label file lacks a label for any spectra sample, return
`blocked`. Do not emit incomplete `y.csv`, delete samples, or align by row order
unless explicitly requested for a row-order label scenario.

## Band Axis Length Mismatch

An external band_axis file must have the same length as X feature count. If not,
return `blocked`; do not truncate or pad either side.

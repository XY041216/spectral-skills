# Missing Values

Use this fragment when spectral tables contain blank cells or missing markers.

## X Values

- Missing X tokens such as blank, `NA`, `N/A`, `NaN`, `NULL`, `None`,
  `missing`, `--`, `-`, and `.` should be read as missing values.
- Keep X rectangular as samples by bands. Do not fill, drop samples, or drop
  bands in the reader.
- Non-missing text such as `error`, `bad`, `abc`, labels, units, or leftover
  headers inside X should block the read.
- `Inf` and `-Inf` are missing-risk values for reader output and should be
  normalized to missing values when they appear in X.

## Labels, Targets, And Metadata

- Embedded labels or targets with missing cells can be written with missing
  values preserved.
- Regression targets must be numeric or missing. Non-missing text in confirmed
  target columns should block or require confirmation.
- Metadata missing values are preserved. Metadata imputation is not a reader
  task.

## Handoff

The reader preserves structure. Filling, deleting, retaining, or marking missing
values belongs to spectral-qc.

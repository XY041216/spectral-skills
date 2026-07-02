# Excel Layout Cases

Use this fragment when reading Excel or ODS spectral workbooks.

## Single Sheet

If there is one clear sheet, read it as the spectral table. Samples may be rows
or columns depending on the confirmed orientation.

## Multiple Sheets

If multiple sheets may contain spectra, return `needs_confirmation` and ask for
`spectral_sheet`. A sheet named like `Labels`, `Targets`, or `Metadata` may be
used as an auxiliary label or metadata source only when selected or uniquely
identified.

## Sheet Labels

When a label sheet is used, align by the confirmed join key and preserve the
spectral sample order. Duplicate label keys or missing required labels must
return `blocked`.

## Multirow Headers

For multirow headers, flatten the selected header rows for extraction while
preserving numeric band values in `band_axis.csv`.

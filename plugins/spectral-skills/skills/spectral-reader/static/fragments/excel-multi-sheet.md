# Excel / ODS Multi Sheet

Excel and ODS tables are read into the same minimal output as CSV/TSV/TXT:
`X.csv`, optional `y.csv`, `sample_ids.csv`, `band_axis.csv`, optional
`metadata.csv`, and `data_contract.json`.

For a workbook with one non-empty sheet, use that sheet when the spectral
columns are clear. For multiple candidate sheets, do not auto-confirm the
spectral sheet. Ask only for the missing sheet selection and suggest
`--spectral-sheet` or `--sheet-index`.

When a workbook contains a separate label sheet, require:

- `--spectral-sheet`
- `--label-sheet`
- `--sample-id-column`
- `--join-key`
- `--label-column` or `--target-columns`

Keep the spectral sheet sample order. Block duplicate label keys and missing
required labels. Do not merge multiple spectral sheets automatically.

Stable blocked reasons include `SHEET_NOT_FOUND`, `EMPTY_SHEET`,
`SPECTRAL_SHEET_NOT_CONFIRMED`, `DUPLICATE_LABEL_KEYS`,
`MISSING_REQUIRED_LABELS`, `EXCEL_ENGINE_MISSING`, and `ODS_ENGINE_MISSING`.

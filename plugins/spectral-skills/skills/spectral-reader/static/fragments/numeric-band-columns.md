# Numeric Band Columns

Use this fragment when table headers are numeric or numeric-like and may be
spectral bands.

## Recognition

- Contiguous numeric headers such as `900`, `1000`, `1100` are strong spectral
  column candidates.
- Numeric headers written as workbook values, such as `900.0`, may match user
  input `900` when selecting spectral boundaries.
- Numeric headers with units such as `900 nm` or `1000 cm-1` are band-axis
  candidates and should preserve their numeric values in `band_axis.csv`.
- Numeric metadata near the spectrum, such as `moisture`, `No.`, or sequence
  columns, must not enter X unless explicitly selected as spectral columns.
- A table header with a sample ID field followed by many band-like numeric
  headers should remain the header even if a later data row contains an invalid
  string. Bad spectral cells should block the read, not shift the header.

## Confirmation

Ask for the smallest confirmation when numeric metadata and numeric band
headers are adjacent. Prefer `spectral_start_column` and `spectral_end_column`
for contiguous regions.

If the sample ID header is blank but the first column contains sample IDs, a
column index such as `--sample-id-column-index 0` may identify it, including
empty first-header index columns.

## Output

When numeric headers are used as bands, write X as samples by features and
write the parsed header values to `band_axis.csv`. If no reliable band values
exist, generate `feature_001`, `feature_002`, and so on.

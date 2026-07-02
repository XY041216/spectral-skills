# Instrument Export Cases

These cases help the Agent interpret preview evidence from instrument-exported
tables. They are examples, not hard rules.

## Leading Export Notes

Some instruments write lines such as software version, operator, acquisition
date, or method before the real table header. Treat these as preamble
candidates. Set `skiprows` and `header_row` only after confirming those lines
are not samples.

## Metadata Columns Before Spectra

Instrument exports often include sequence number, vial position, remark,
operator, or batch columns before spectral bands. Treat these as metadata
candidates until confirmed; do not include them in `spectral_columns` by
position alone.

## Unit-Bearing Band Headers

Headers such as `900 cm-1`, `1000 cm-1`, or `1100 nm` can support
`band_axis.source=column_headers` and a matching `band_unit`, but the Agent
still confirms spectral boundaries when nearby columns are ambiguous.

# Wavenumber Columns

Wavenumber headers may look like `900 cm-1`, `900cm-1`, `900 cm鈦宦筦, or plain
numeric columns supported by context.

Set `band_unit = cm-1` when `band_like_column_evidence` or column preview
supports wavenumber. Consecutive wavenumber columns should be recorded as
`spectral_columns` and `band_axis`.

For samples-as-columns tables, wavenumber values may appear in
`samples_as_columns.band_axis_column` instead of column headers.

Do not drop the first wavenumber column because it is near metadata. Distinguish
spectral numeric headers from numeric sequence columns such as `No.`.

## Reading Semantics

This scene usually affects `spectral_columns`, `band_axis`,
`band_unit = cm-1`, `spectral_type`, and `internal evidence`.

Require confirmation when the first or last wavenumber column is ambiguous, or
when plain numeric headers could be either band values or metadata. Block the
plan if no usable band-like column set can be stated.


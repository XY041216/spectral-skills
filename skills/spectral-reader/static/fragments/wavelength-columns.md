# Wavelength Columns

Wavelength headers may look like `900 nm`, `900nm`, `wl_900`, `band_900`, or
plain numeric columns supported by context.

Set `band_unit = nm` when evidence supports wavelength. Record continuous
wavelength headers as `spectral_columns` and `band_axis`.

For samples-as-columns tables, wavelength values may appear in
`samples_as_columns.band_axis_column` instead of column headers.

## Reading Semantics

This scene usually affects `spectral_columns`, `band_axis`, `band_unit = nm`,
`spectral_type`, and `internal evidence`.

Require confirmation when wavelength columns are mixed with metadata-like
numeric columns or when the unit is inferred from naming context only. Block the
plan if the wavelength axis cannot be represented as columns, a row, a file, or
an explicit index.


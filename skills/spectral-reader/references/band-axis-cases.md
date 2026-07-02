# Band Axis Cases

These examples describe how band-axis evidence is represented during spectral
data reading.

## Band Axis Source Cases

- Nanometer headers: `band_unit=nm`, `band_axis.source=column_headers`.
- Wavenumber headers: `band_unit=cm-1`, `band_axis.source=column_headers`.
- First-row axis: record `band_axis.source=first_row` and table boundaries.
- First-column axis: record `sample_orientation=columns` and
  `band_axis.source=first_column`.
- External axis file: record the axis file path and alignment rule.
- Pure numeric headers: require boundary or unit confirmation when context is
  not enough.

## Nanometer Columns

Headers like `900 nm`, `900nm`, `wl_900`, or `band_900` may indicate
`band_unit=nm`.

## Wavenumber Columns

Headers like `900 cm-1`, `900cm-1`, or `波数900` may indicate `band_unit=cm-1`.

## First Row Axis

Some files place the band axis in the first row and samples in later rows.
Record header handling and confirm if ambiguous.

## First Column Axis

Samples-as-columns tables may place band axis in the first column.

## Independent Axis File

External wavelength or wavenumber files must be recorded as separate source
refs and aligned by count or explicit key.

## Pure Numeric Headers

Plain numeric columns are band axis candidates when they are contiguous,
monotonic or instrument-like, and not obvious IDs or sequence numbers.

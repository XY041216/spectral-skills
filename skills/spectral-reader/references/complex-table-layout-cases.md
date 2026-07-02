# Complex Table Layout Cases

## Multi-Row Header Rows

Confirm `header_rows=0,1`, `sample_id_column=sample_id`,
`metadata_columns=batch`, `label_column=Class`, and spectral columns or a
spectral range covering `900` to `1100`.

## External Band Axis File

Confirm the main X columns and set `band_axis_file=bands.csv` with
`band_axis_column=band`. Block if the band count differs from the feature count.

## Metadata, Spectra, Label Partition

Confirm metadata columns before the spectral region, a continuous spectral
region, and a label column after the spectral region. Metadata and label columns
must not enter X.

## Multi-Target Regression

Confirm `target_columns` such as `total_sugar,nicotine,potassium`. Write these
columns to `y.csv` and set `task_hint=multi_target_regression`.

## Label/Target Conflict

If both `label_column` and `target_columns` are supplied, block with
`LABEL_TARGET_CONFLICT`.

## Non-Numeric Spectral Block

If any selected spectral cell cannot be converted to a number, block with
`SPECTRAL_BLOCK_NON_NUMERIC`.

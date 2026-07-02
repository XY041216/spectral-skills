# Samples As Columns

Use this fragment when preview evidence suggests bands are rows and samples
are columns.

## Evidence

- first column is band-like and mostly numeric;
- first column is monotonic or plausibly ordered;
- remaining columns are mostly numeric;
- remaining column headers look like sample IDs;
- row count looks more like feature count than sample count.

Preview only records this evidence. The Agent must still produce or repair a
read_plan and get required confirmations.

## Confirmations

Require confirmation for:

- `sample_orientation=columns`;
- `samples_as_columns.band_axis_column`;
- `samples_as_columns.sample_id_source`;
- whether labels are absent or supplied by `label_file`.

## Read Plan Fields

Set:

```json
{
  "read_mode": "matrix_file",
  "sample_orientation": "columns",
  "samples_as_columns": {
    "enabled": true,
    "band_axis_column": "band",
    "sample_id_source": "column_headers",
    "sample_id_row": 0,
    "data_start_row": 1,
    "sample_start_column": 1,
    "transpose_required": true
  }
}
```

The executor writes `X` as samples by features after transposition.

## Blocked Conditions

Block when band-axis column is absent, sample columns are missing, X is
non-numeric, sample IDs are duplicate or missing, or band-axis length does not
match the transposed feature count.

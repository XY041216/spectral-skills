# HDF5 / NetCDF Dataset Path Selection

Use HDF5 / NetCDF paths only to identify the arrays needed for the standard reader output.
Do not persist dataset inventories, debug reports, or read plans.

Required mapping:

- `X`: a two-dimensional numeric dataset or variable.
- `y`: optional one-dimensional label or target vector.
- `sample_ids`: optional one-dimensional sample identifier vector.
- `band_axis`: optional one-dimensional feature axis vector.
- `metadata`: optional one-dimensional or two-dimensional metadata array.

If exactly one two-dimensional numeric dataset exists, it can be used as `X`.
If multiple two-dimensional numeric datasets exist, return `needs_confirmation` and ask for `--x-path`.

Generated values:

- If `sample_ids` is absent and no external label file is used, generate stable `sample_001` IDs.
- If `band_axis` is absent, generate stable `feature_001` values.
- If an external label file is used, real `sample_ids` must be provided by `--sample-ids-path`.

Blocked cases:

- HDF5 without `h5py`: `H5PY_MISSING`.
- NetCDF without `netCDF4`: `NETCDF4_MISSING`.
- Missing requested path: `DATASET_PATH_NOT_FOUND` or `VARIABLE_PATH_NOT_FOUND`.
- X is not a two-dimensional numeric array.
- y, sample_ids, metadata, or band_axis length does not match X.

Out of scope:

- automatic cube unfolding
- automatic multi-dataset merging
- instrument-specific HDF5 interpretation
- object references, compound dtype semantics, sparse arrays, and complex dimension inference

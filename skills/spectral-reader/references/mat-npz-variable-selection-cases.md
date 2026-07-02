# NPY / NPZ / MAT Variable Selection Cases

## NPY Single Matrix

`X.npy` contains one 2D numeric array. Read it as samples by features. Generate
`sample_001...` and `feature_001...` when IDs and band axis are absent.

## NPZ X + y + band Axis

`dataset.npz` contains `X`, `y`, `sample_ids`, and `band_axis`. Use explicit
variable arguments when present and write the minimal standard output.

`y`, `sample_ids`, and metadata may be object/string arrays when they flatten to
the expected length.

## NPZ Multiple X Candidates

If `X_raw` and `X_processed` are both 2D numeric arrays and `--x-var` is not
specified, return `needs_confirmation` with candidate names.

## MAT Variables

Non-HDF5 MAT files may provide `X`, `y`, `sample_ids`, and `band_axis`. Simple
string cells for labels and sample IDs are supported when they flatten cleanly.
Simple metadata row vectors or column vectors can be expanded to one metadata
row per sample.

## MAT v7.3

MAT v7.3 is HDF5-based. Return blocked with `MAT_V73_NOT_SUPPORTED`.

## Container + External Labels

When labels are supplied by CSV/TSV/TXT/Excel, align by `join_key` using real
container sample IDs. Block if sample IDs would otherwise be generated.

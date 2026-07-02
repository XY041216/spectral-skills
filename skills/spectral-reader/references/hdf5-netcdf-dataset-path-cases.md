# HDF5 / NetCDF Dataset Path Cases

## HDF5 Single X

Input contains only `/X` as a two-dimensional numeric dataset.
The reader may use `/X` as `X`, generate sample IDs and band axis, and write the minimal output files.

## HDF5 Mapped Paths

Input contains `/spectra/X`, `/labels/y`, `/meta/sample_ids`, and `/axis/band_axis`.
When these paths are provided, read them directly and preserve X sample order.

## HDF5 Multiple X Candidates

Input contains `/raw/X` and `/processed/X`.
Without `--x-path`, return `needs_confirmation` with suggested X paths.

## NetCDF Variables

Input contains `X`, `y`, `sample_ids`, and `band_axis` variables.
Use variable names as paths and output the same standard files as CSV or NPZ inputs.

## NetCDF Multiple X Candidates

Input contains `X_raw` and `X_processed`.
Without `--x-path`, return `needs_confirmation`.

## External Labels

If a label file is used, HDF5 / NetCDF input must provide sample IDs by `--sample-ids-path`.
Align labels by `join_key`, block duplicate keys, and block missing required labels.

# NPY / NPZ / MAT Variable Selection

NPY, NPZ, and non-HDF5 MAT files are container-style inputs. The reader uses
variable evidence only to select the arrays needed for standard output.

NPY contains one array. It must be a 2D numeric matrix. If no sample IDs or band
axis are present, generate stable `sample_001...` and `feature_001...` values.

NPZ and MAT may contain several variables. Use:

- `--x-var` for the 2D numeric X matrix
- `--y-var` for label or target values
- `--sample-ids-var` for sample IDs
- `--band-axis-var` for wavelength, wavenumber, or feature labels
- `--metadata-var` for simple 1D/2D metadata

If exactly one 2D numeric array exists, it may be used as X. If multiple 2D
numeric arrays exist and `--x-var` is missing, return `needs_confirmation`.

Object/string vectors for `y`, `sample_ids`, `band_axis`, and simple metadata
are valid when they flatten cleanly to the sample or feature length. A metadata
row vector or column vector may expand to one metadata value per sample.

MAT v7.3 is HDF5-based and is blocked with `MAT_V73_NOT_SUPPORTED`. Deep struct,
complex cell arrays, HDF5, NetCDF, hyperspectral cubes, sparse arrays, and
automatic multi-variable merging are not part of this reader step.

External label alignment for container inputs requires real sample IDs from a
sample ID variable. Do not align external labels to generated row numbers.

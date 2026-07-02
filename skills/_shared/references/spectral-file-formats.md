# Spectral File Formats

Spectral data can appear in many containers. File format support and semantic
interpretation are separate decisions: a parser may read a file while a skill
still needs evidence to decide sample orientation, labels, band axis, and task.

## Common Forms

- CSV / TSV / TXT: delimited text tables or instrument exports.
- Excel / ODS: spreadsheet workbooks with one or more sheets.
- MAT: MATLAB files containing arrays or structs.
- NPY / NPZ: numpy arrays or packed arrays.
- HDF5 / NetCDF: hierarchical scientific containers.
- JCAMP-DX: text-based spectroscopy exchange format.
- Instrument-exported text: may contain metadata headers and numeric blocks.
- One-file-per-sample folders: each file represents one sample spectrum.

Concrete reading rules, header repair, and format-specific read plans belong to
the reader skill references.

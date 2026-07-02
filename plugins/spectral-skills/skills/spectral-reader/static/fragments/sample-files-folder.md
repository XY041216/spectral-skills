# Sample Files Folder

One-file-per-sample folders contain many files, each representing one sample.

Sample IDs may come from filenames. Parent folder names are label candidates
and must be confirmed before becoming labels.

Current executable scope is conservative: individual CSV/TSV/TXT files contain
two columns (`band_axis` + intensity), all files share the same band axis, and
sample IDs come from file names. Parent folder names or file names may become
labels only after confirmation.

Do not automatically interpolate or resample; band-axis mismatches are blocked.
Single-row sample spectra remain a planned extension unless the read_plan has a
separate confirmed reader implementation.

## Reading Semantics

This scene usually affects `read_mode = sample_files_folder`,
`sample_orientation = one_file_per_sample`, `sample_file_pattern`,
`sample_file_columns.band_axis`, `sample_file_columns.value`,
`file_name_as_sample_id`, `folder_name_as_label`,
`per_file_band_axis_policy`, and `alignment_plan`.

Require confirmation when folder names or file names are used as labels. Block
the plan if the file pattern is undefined, files cannot be inventoried, or band
axis policy cannot be stated.


# Metadata Before Spectra

Spectral tables often place metadata before spectral columns.

Common metadata columns include `remark`, `No.`, `index`, `batch`, `date`,
`operator`, `sample_id`, `sample`, `编号`, `序号`, and `备注`.

Use `metadata_like_column_evidence`, `sample_id_like_column_evidence`, and
`band_like_column_evidence` from preview. Numeric sequence columns must not
automatically enter X. Record them in `metadata_columns` unless the user
confirms otherwise. Spectral columns should be a contiguous or explicitly listed
range in `spectral_columns`.

## Reading Semantics

This scene usually affects `metadata.columns`, `sample_id.column`,
`spectral_columns`, `data_start_column`, `data_end_column`, and
`internal evidence`.

Require confirmation when a numeric ID or sequence column is adjacent to the
first band-like column, or when the metadata/spectral boundary is inferred from
names only. Return `blocked` if no spectral column evidence remains after
excluding metadata candidates.

When metadata is supplied by an external label file, record those columns under
`label_file.metadata_columns` and merge them only after sample-id alignment.


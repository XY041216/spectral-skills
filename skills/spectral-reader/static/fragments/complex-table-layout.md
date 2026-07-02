# Complex Table Layout

Use these rules only to confirm reading semantics for irregular but structured
spectral tables. Do not persist preview evidence or intermediate read plans.

- Multi-row headers require explicit `header_rows`, for example `0,1`.
- Use the last non-empty header cell as the working column name so type rows
  such as `sample_info` or `spectra` do not pollute `band_axis`.
- Use `spectral_start_column` and `spectral_end_column` when spectra are a
  continuous region.
- Use `band_axis_file` plus `band_axis_column` when feature names are generic.
- Exclude `sample_id_column`, `metadata_columns`, `label_column`, and
  `target_columns` from X.
- Multiple target columns are written as multi-column `y.csv`.
- If `label_column` and `target_columns` are both provided, block and ask for a
  single supervised role.
- Do not merge multiple independent data blocks automatically.

# Sample File Folder Cases

These examples describe one-file-per-sample folder reading. They are not hard
rules.

## Folder Reading Rules

- Folder input uses `sample_orientation=one_file_per_sample`.
- File glob comes from `sample_file_pattern`.
- Two-column files need confirmed band-axis and value columns.
- File stems become sample IDs by default.
- Folder names become labels only when requested with
  `folder_name_as_label=true`.
- Band axes must be consistent across sample files.

Current executable scope is limited to CSV/TSV/TXT two-column sample files with
a shared band axis. Single-row sample spectra and automatic resampling are not
performed by reader.

## Multiple Sample Files

A folder may contain many TXT or CSV files, each representing one spectrum.
File stems can become sample IDs.

## Folder Labels

Parent folder names may be class candidates. Confirm before using them as
labels.

## Two-Column Spectrum

Each file may contain `band_axis,intensity`. Check that band axes match across
files. The reader can execute this case when the band and value columns are
confirmed.

## Single Row Spectrum

Each file may contain one row of intensities. Treat this as a planned extension
unless a future reader implementation explicitly supports it.

## Axis Mismatch

If band axes differ, do not interpolate automatically. Return
`needs_confirmation` or `blocked` with a concise reason.

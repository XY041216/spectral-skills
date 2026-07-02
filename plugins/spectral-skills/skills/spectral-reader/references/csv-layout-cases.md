# CSV Layout Cases

These examples describe common CSV/TXT/TSV spectral data layouts. They are not
hard rules and do not replace user confirmation.

## Reading Notes

`read_spectral_dataset` extracts exactly the confirmed spectral columns, sample
ID, label, target, and metadata fields. Missing columns, invalid non-missing X
values, or a band-axis length mismatch should return `blocked` with a short
reason.

Blank, `NA`, `N/A`, `nan`, and similar spectral cells are missing values. Keep
the sample by band matrix rectangular and pass those missing values downstream;
QC decides whether to fill, remove, or retain them.

The final user output remains the standard files plus `data_contract.json`.

## Layout Recognition Cases

- Standard rows table: `sample_orientation=rows`.
- Preamble table: set `skiprows`, `header_row`, and add
  a minimal confirmation when needed.
- Metadata before spectra: put sequence, remark, batch, and operator fields in
  metadata columns, not spectral columns.
- Wavenumber columns: set `band_unit=cm-1` and preserve first/last band
  boundary evidence.
- Label column: set `label_column` only when evidence and confirmation support
  classification.
- Uncertain boundary: return `needs_confirmation`; missing spectral columns
  should return `blocked`.

## Standard Samples As Rows

Columns: `sample_id`, spectral bands, optional `class` or `target`. Read plan
semantics set `sample_orientation=rows`.

## Preamble Plus Header

Leading lines begin with `# exported by`, `# instrument`, or similar notes. The
true table begins at a later row. Read plan sets `skiprows` and `header_row`,
then asks the user to confirm skipped lines.

## Metadata + Wavenumber + Label

Example header:

```text
remark,Unnamed: 0,No.,900 cm-1,1000 cm-1,...,2500 cm-1,Class
```

Agent pattern:

- `remark` is metadata.
- `Unnamed: 0` is sample ID candidate when values look like sample names.
- `No.` is metadata or sequence, not X by default.
- `900 cm-1` through `2500 cm-1` are spectral columns.
- `Class` is label.
- `band_unit=cm-1`.
- Confirm skipping preamble and confirm spectral column boundaries.

## First Column ID Last Column Label

First column may be `sample_id`; last column may be `label`. Middle numeric
band columns form X.

## Samples As Columns

Rows represent bands; columns represent samples. Confirm orientation.

## Pure Numeric Band Columns

Numeric column names may be band axis only when contiguous and surrounded by
non-spectral metadata or label columns.

## Decimal Comma Values

Semicolon-delimited exports may use decimal commas, such as `0,123`. When the
delimiter is confirmed as `;`, these values can be read as numeric spectral
values.

## Blank Sample ID Header

Some exports leave the first header cell empty while the first column contains
sample IDs. Use a column index, for example `sample_id_column=0`, and keep the
numeric band headers out of sample IDs, labels, and metadata.

## Invalid Spectral Cell

If a spectral region contains a value such as `ERR`, the reader should return
`blocked`. Do not skip earlier rows or reinterpret the bad row as a header.

## Missing Sample IDs

If a sample ID column is present but some entries are blank, ask for
confirmation before generating replacement IDs for those rows. Do not silently
invent sample IDs when labels or metadata may depend on them.

## Chinese Columns

Use Chinese role clues such as `样本编号`, `序号`, `类别`, `含量`, `波长`, `波数`,
and `备注`.

# Confirmation Dialogue Cases

Ask only for unresolved semantics. Do not narrate validation stages.

## Preamble

```text
Should I skip the first three comment lines and use the next row as the table header?
```

## Spectral Boundaries

```text
Should columns 900 nm through 1700 nm be treated as the spectral matrix X?
```

## Label

```text
Is `class` the sample label column for this read?
```

## Samples As Columns

```text
Are samples stored as columns, with `band` as the band axis and the remaining column headers as sample IDs?
```

## External Label File

```text
Should I align `labels.csv` to the spectra by `sample_id`, use `Class` as y, and merge `batch` into metadata?
```

## After One Confirmation

Once the user confirms the only blocking semantic item, proceed in the
background to read, assert alignment, build `data_contract.json`, write
the standard CSV files, and return a concise result. Do not write summaries,
package manifests, audit directories, or internal validation reports.

# Sample ID Cases

## No Sample ID Column

If no sample ID column exists, generate stable IDs in sample order.

## Some Sample IDs Missing

If a sample ID column exists and some rows are blank, return
`needs_confirmation` with a missing sample ID policy. After confirmation,
generate replacement IDs only for the missing rows and preserve existing IDs.

## Duplicate Sample IDs

Duplicate sample IDs are blocked because they break sample-level alignment for
labels, metadata, splitting, and QC.

## Repeated Spectra

Identical or near-identical X rows are not a reader concern when sample IDs are
valid. Leave repeated spectra for spectral-check.

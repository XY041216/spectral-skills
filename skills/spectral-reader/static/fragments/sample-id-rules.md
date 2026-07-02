# Sample ID Rules

Use this fragment when sample identifiers are absent, incomplete, or duplicated.

## No Sample ID Column

If the source has no sample ID column, the reader may generate stable IDs such
as `sample_001`, `sample_002`, and `sample_003`.

## Partially Missing Sample IDs

If a sample ID column exists but some rows are missing IDs, return
`needs_confirmation` before generating replacements.

When the user confirms generation, preserve existing IDs and generate only the
missing positions with non-conflicting IDs such as `generated_sample_001`.

## Duplicate Sample IDs

Duplicate sample IDs are a reader structural error because downstream label,
metadata, split, and QC operations depend on unique sample keys. Return
`blocked`.

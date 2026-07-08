# Reader And QC Boundary

The reader reads and standardizes structure. It does not perform QC decisions.

Reader should preserve and pass through:

- missing X values;
- embedded missing labels or targets;
- missing metadata values;
- duplicated or near-duplicated spectra when sample IDs are distinct;
- intensity outliers, noisy spectra, constant bands, low-variance bands,
  target outliers, and class imbalance;
- nonmonotonic or uneven band spacing when the band count matches X.

Reader should block or request confirmation for structural problems:

- external label samples missing for a supervised read;
- duplicate label join keys;
- duplicate sample IDs;
- band_axis length different from X feature count;
- row counts that cannot align;
- missing sample IDs in an existing sample ID column unless generation is
  confirmed;
- multiple plausible files, sheets, or variables without user selection.

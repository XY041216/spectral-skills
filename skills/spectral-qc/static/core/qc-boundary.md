# QC Boundary

`spectral-qc` starts from a ready standard spectral package. It does not read
raw instrument exports or infer source layouts.

## In Scope

- Check missing values in `X.csv`, `y.csv`, and `metadata.csv`.
- Detect constant or low-variance bands.
- Detect duplicate or near-duplicate spectra candidates.
- Detect intensity, noise, PCA score, residual, and robust-statistic outlier
  candidates.
- Detect target outlier candidates for regression tasks.
- Check class count and class imbalance risks for classification tasks.
- Explain risks and recommend conservative actions.
- Apply user-confirmed edits and re-export the standard package.

## Out of Scope

- Raw file reading and source layout parsing.
- External label alignment from raw files.
- Train, validation, or test splitting.
- Smoothing, baseline correction, normalization, derivatives, or other formal
  spectral preprocessing.
- Feature engineering, feature selection, modeling, or optimization.
- Fixed report generation, audit chains, debug modes, or log systems.

## Neighbor Skills

- `spectral-reader` creates the first standard package.
- `spectral-qc` checks and optionally cleans that package.
- `spectral-splitter` splits a standard package after reader or QC.
- `spectral-preprocess` transforms spectra after QC and split decisions.
- `spectral-modeling` trains models after data is ready.

# Reproducibility Rules

Any operation that changes data, indices, features, model state, or report
results must be reproducible from its Contract and artifacts.

## Required Records

Record:

- input sources and input Contract IDs;
- parameters and defaults actually used;
- random seeds and random number libraries;
- selected columns, sample indices, feature indices, and band ranges;
- user confirmations;
- software versions;
- schema versions;
- execution backend and fallback path;
- warnings, errors, and skipped steps.

## Fit/Transform Separation

Any learned transformation must record where it was fit and where it was
applied. Preprocessing, scaling, supervised feature selection, and model
selection must avoid using validation or test labels during fit.

## Search Records

Optimization and model selection must record trial definitions, trial results,
selection criteria, and the final chosen configuration. Reporting only the best
result is not reproducible.

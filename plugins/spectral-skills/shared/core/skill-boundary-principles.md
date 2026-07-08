# Skill Boundary Principles

Each Spectral Skill owns one layer of the workflow.

## Boundaries

- `spectral-reader`: read files, standardize data references, infer layout,
  align labels, and produce a Spectral Data Contract. It does not perform QC,
  splitting, preprocessing, feature engineering, modeling, or optimization.
- `spectral-check`: inspect quality, detect risks, and recommend quality actions.
  It does not mutate the modeling dataset unless a later skill applies changes.
- `spectral-splitter`: create leakage-aware split indices and validation
  protocols. It does not preprocess or train models.
- `spectral-preprocess`: transform spectral matrices according to confirmed fit
  scopes and parameters. It does not choose labels, split data, or train models.
- `spectral-feature`: perform dimensionality reduction, variable selection, and
  feature construction. It must respect split and leakage rules.
- `spectral-modeling`: train and evaluate models using confirmed upstream
  contracts.
- `spectral-optimizer`: search pipeline or hyperparameter combinations and
  record trials.
- `spectral-report`: summarize, explain, visualize, and export reports from
  upstream artifacts.

When a request crosses boundaries, complete and validate the upstream Contract
before handing off.

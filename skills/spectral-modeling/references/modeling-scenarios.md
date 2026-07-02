# Modeling Scenarios

- "Train a classifier from feature output" -> use feature package
  `data_contract.json`, the existing `split_contract.json`, task
  `classification`, and requested classification models.
- "Use PLSR for regression" -> task `regression`, model `plsr`; tune
  `n_components` on validation split or train-only CV.
- "Use PLS-DA or SIMCA" -> task `classification`, model `pls_da` or
  `simca`; record `model_family=chemometrics` and keep selection inside train
  or validation rules.
- "Use GPR for regression uncertainty" -> task `regression`, model `gpr`;
  write ordinary predictions plus `prediction_std.csv` and
  `uncertainty_summary.json` when available.
- "Use CLS-Former / DKL-GP / prototype model" -> treat as experimental
  small-sample modeling. Ask for critical training parameter confirmation or
  require `--auto-confirm-model-defaults`; block if torch/gpytorch is missing.
- "Compare SVM and random forest" -> fit each candidate on train, select by
  validation metric, then evaluate the chosen model on test.
- "Use splitter output directly" -> allowed if the package is standard and has
  `y.csv`; do not run preprocessing or feature extraction inside modeling.
- "Tune on test" -> refuse by default because test data would leak into model
  selection.

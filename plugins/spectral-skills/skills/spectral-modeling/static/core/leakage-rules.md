# Leakage Rules

Train candidate models only on train samples.

Validation samples are used only for hyperparameter selection, candidate model
comparison, and early model choice. If no validation split exists, use
cross-validation inside the train split.

Use test samples only once after model type and parameters are fixed. Never use
test metrics to choose preprocessing, features, model type, hyperparameters, or
thresholds.

Write `test_used_for_selection: false` in `modeling_contract.json`.

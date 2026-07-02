# Leakage Rules

Run splitter before any train-fitted preprocessing, feature engineering, or
modeling.

Let preprocess fit global parameters on train only. Let feature fit PCA or
selection rules on train only. Let modeling train on train, select on validation
or train-only CV, and evaluate the test split only after model selection.

Refuse default workflows that use all data for standardization/PCA or use test
metrics for method or model selection.

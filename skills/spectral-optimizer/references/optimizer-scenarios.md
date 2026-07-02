# Optimizer Scenarios

- "Recommend a pipeline for 120 x 3401 classification data" ->
  `recommend_from_profile`.
- "Find the best VIP top_k" -> `tune_method --target-step feature --method vip`.
- "Compare PCA, VIP, SPA" -> `compare_step --target-step feature`.
- "Find the best model among SVM, PLS-DA, logistic regression" ->
  `compare_step --target-step modeling`.
- "Fully optimize the pipeline" -> `optimize_pipeline` with a confirmed
  `--max-trials` budget.

If a trial result table is supplied, select the best row only by `val_*` or
`cv_*` metrics. Ignore `test_*` metrics for selection.

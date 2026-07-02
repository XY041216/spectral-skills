# Optimizer Leakage Rules

- Use train data for fitting inside child workflow trials.
- Use validation or inner CV metrics for optimizer selection.
- For classification, prefer `val_macro_f1` as the default primary selection
  metric; use accuracy, balanced accuracy, and AUC only as secondary metrics.
- Never use test metrics to choose method, parameters, or pipeline.
- Use `spectral-modeling --evaluation-mode validation_only` for holdout
  candidate trials that need official modeling behavior without reading test.
- Evaluate the final selected pipeline on test once after selection.
- Ask the user to confirm final/confirmatory test evaluation before the first
  final holdout test access. If `test_access_log.json` already exists, state
  that later test metrics are confirmatory and no longer fully blind.
- Record `test_used_for_selection: false` in every optimizer contract.
- If `test_access_log.json` indicates the test set was already viewed, mark
  later final test metrics as confirmatory rather than fully blind.
- If only test metrics are available, return `blocked` or ask the user to run
  validation/CV trials.

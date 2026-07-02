# Statistical Reporting

## Units and Repetition

- Distinguish sample, fold, repeat, dataset, and independent experiment.
- Folds from one CV run are not independent biological experiments.
- For repeated k-fold, prefer repeat-level aggregates or paired method differences.
- With fewer than five repeat units, show all points and a median/range rather than emphasizing a box.

## Uncertainty

- Define every SD, SE, CI, IQR, prediction interval, and shaded band in the caption.
- Do not convert fold spread into a confidence interval without an explicit method.
- Preserve pairing IDs when comparing models on the same folds/repeats.

## Comparisons

- Prefer effect sizes and paired difference CIs.
- When formal multi-model testing is justified, use an omnibus test followed by multiplicity-adjusted post hoc comparisons such as Holm.
- Do not run naive independent t-tests on dependent folds.
- Do not conduct candidate significance screening on one final test split.

## Metrics

- Classification: prefer Macro-F1; add accuracy, balanced accuracy, ROC-AUC/PR-AUC when supported.
- Regression: report R2, RMSE, and MAE with split and units.
- Keep test metrics isolated from validation/CV selection evidence.


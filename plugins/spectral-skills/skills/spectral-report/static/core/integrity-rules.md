# Integrity Rules

1. Establish the statistical unit before choosing a chart or test.
2. Preserve paired structure for models evaluated on the same folds/repeats.
3. Treat repeated k-fold folds as dependent; prefer repeat-level aggregates or paired effect differences.
4. Draw uncertainty only from real folds, repeats, bootstrap samples, prediction distributions, or independent experiments.
5. A single holdout value supports a point, not a box, violin, CI, or significance mark.
6. Optimizer figures may use validation/CV metrics only. Keep final-test evidence in a separate validation panel or figure.
7. A final test evaluates a locked pipeline; it does not rank candidates.
8. Show class support and macro/balanced metrics when imbalance matters.
9. Record every scale conversion, including fraction-to-percent, in source data and the report contract.
10. Keep every plotted value traceable to source data and upstream files.

Block when these rules cannot be satisfied without inventing evidence.


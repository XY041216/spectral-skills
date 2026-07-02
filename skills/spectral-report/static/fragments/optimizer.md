# Optimizer Results

Read `references/chart-decision-matrix.md`, `references/statistical-reporting.md`, `references/layout-recipes.md`, and `references/caption-rules.md`.

- Read `optimizer_contract.json`, `trial_results.csv`, and `best_pipeline.json` together.
- Use only validation/CV fields for candidate ranking and visual emphasis. Ignore candidate test fields even if present.
- Verify the highlighted trial matches `best_pipeline.json`; do not select a new winner in the report.
- Use a horizontal dot plot for nominal pipeline candidates, a line for ordered parameter response, or a heatmap/landscape for structured grids.
- Show failed/skipped trials separately from scored trials; do not coerce failures to zero.
- Record confirmed budget, completed count, selection metric, tie-breaker, and post-test exploratory context.
- Put final locked-model test evidence in a separate validation panel/figure with explicit test-isolation status.


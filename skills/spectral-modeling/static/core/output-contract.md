# Output Contract

Write:

- `modeling_contract.json`
- `modeling_summary.json`
- `metrics.json`
- `predictions.csv`
- optional `model_artifact.pkl`
- optional `confusion_matrix.csv`
- optional `test_access_log.json` when final holdout test metrics are read

Do not write train/test matrix copies or a new standard spectral package.

`metrics.json` must keep train, validation, and test metrics separate.
For `evaluation_mode=validation_only`, `test_metrics` must be empty,
predictions must include only train/validation rows, and the writer must not
create `confusion_matrix.csv`, final `model_artifact.pkl`, or
`test_access_log.json`.
`predictions.csv` must include `sample_id`, `split`, `y_true`, and `y_pred`.
The downstream handoff file is `modeling_contract.json`.
Both `modeling_contract.json` and `modeling_summary.json` must state the
preprocessing, feature engineering, split, seed, and candidate model conditions
under which the selected model and metrics were obtained. They must also record
`evaluation_mode` and `test_accessed`.

## Repeated Classifier Comparison Outputs

When the task is repeated classifier comparison rather than model selection,
write a separate per-classifier comparison contract and tables:

- `classifier_comparison_contract.json`
- `classifier_repeat_metrics.csv`
- `classifier_metric_summary.csv`
- optional `classifier_repeat_predictions.csv`

The comparison contract must state
`comparison_mode=per_classifier_repeated_evaluation`,
`model_selection_enabled=false`, `candidate_model_set_source`,
`candidate_models`, `model_params_source`, `same_repeats_across_models=true`,
`statistical_unit=repeat`, and the exact preprocessing/feature pipeline
lineage. Existing local scripts or result directories are not confirmation of a
candidate model set.

`classifier_repeat_metrics.csv` must contain one row for each
`repeat_id` x `model_method` pair. If the implementation can only produce
per-repeat winners, return `blocked` with
`PER_CLASSIFIER_REPEATED_OUTPUT_UNAVAILABLE` and do not present the output as a
fair classifier comparison.

`classifier_metric_summary.csv` must include mean, SD, n, and statistical unit
for every reported metric. Report outputs such as `spectral-report` depend on
these columns to produce boxplots, raw repeat points, and paper-ready
`mean +/- SD` three-line tables.

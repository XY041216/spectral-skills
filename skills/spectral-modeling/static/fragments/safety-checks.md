# Safety Checks

Return `needs_confirmation` when task type or model choice is missing, or when
there is no test split and the user has not confirmed a validation-only run.
When `--require-test-confirmation` is set, also return
`TEST_EVALUATION_CONFIRMATION_REQUIRED` until the user explicitly confirms
final/confirmatory test evaluation with `--confirm-test-evaluation`.

Return `blocked` when `y.csv` is missing, split assignments are invalid,
classification has fewer than two classes, regression targets are non-numeric,
the requested model is unsupported for the task, or an optional model
dependency is missing.

Return `needs_confirmation` when an experimental small-sample model is selected
without confirmed critical parameters. Confirm only the compact training
controls the user must understand, such as epochs, batch size, alpha/device for
CLS-Former, or preprojection/components/embedding/kernel for DKL-GP.

Refuse requests to use the test split for tuning or model selection by default.

# Regression

Read `references/chart-decision-matrix.md`, `references/axis-and-unit-rules.md`, `references/statistical-reporting.md`, and `references/caption-rules.md`.

- Use predicted versus measured as the main panel, with a 1:1 line and matched units/ranges.
- Add residual versus predicted and an error distribution only when they support a distinct diagnostic question.
- Label R2, RMSE, and MAE with the evaluated split. RMSE/MAE must include the target unit.
- Plot prediction intervals or standard deviations only when supplied by the model output; state coverage level.
- For repeated validation, preserve repeat IDs and report repeat-level mean/std or paired differences.
- Separate targets with incompatible units into panels or figures. Do not normalize them silently into one scale.
- For multi-target regression, require target names and units before plotting.


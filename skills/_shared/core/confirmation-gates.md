# Confirmation Gates

Confirmation gates protect decisions that change data meaning, sample
correspondence, leakage risk, or downstream validity.

## Must Confirm

Require explicit user confirmation when a workflow will:

- select or change a label, target, sample ID, group ID, batch column, or task
  type;
- remove samples, remove bands, remove variables, impute labels, or supplement
  missing labels;
- align separate files by row order, filename pattern, normalized IDs, folder
  names, or any non-exact key;
- choose a split strategy, group-aware rule, time-aware rule, external
  validation set, or nested cross-validation design;
- decide the fit scope of preprocessing, feature selection, scaling, or
  normalization;
- choose the primary model-selection metric, optimization objective, or
  reporting target;
- accept any action that changes X, y, sample IDs, band axis, split indices,
  selected features, fitted model parameters, or reported metrics.

## Should Confirm

Ask for confirmation when evidence is plausible but ambiguous:

- multiple plausible task types exist;
- multiple class or target columns are present;
- spectral units, band axis direction, or measurement type are uncertain;
- small sample size makes a validation design unstable;
- several preprocessing or feature pipelines have similar justification;
- warnings are not blocking but may materially affect interpretation.

## No Confirmation Needed

Do not ask for low-risk operational choices that do not change semantics:

- JSON indentation, output filename suffixes, temporary cache paths, delimiter
  detection already verified by data preview, or warning-only diagnostics;
- optional dependency warnings that do not affect the current file type;
- deterministic schema validation and read-only profiling.

## Status Relationship

- `confirmed`: all required gates for the current downstream use are resolved.
- `provisional`: at least one required gate remains unresolved, but limited
  inspection or selected downstream work may continue with the gate preserved.
- `blocked`: required evidence is missing, a user decision is unavailable, or a
  high-risk operation would be unsafe.

Every confirmation must be recorded in the relevant Contract, including the
question, proposed decision, evidence, user response, timestamp, and affected
fields.

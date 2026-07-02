# Spectral Reader Workflow

`spectral-reader` is a user-facing data reading skill. It is not a workflow
archive, debugging surface, log store, or data audit system.

## Main Flow

Use this flow for normal work:

1. inspect enough of the input to determine read semantics;
2. confirm ambiguous semantics with the user when needed;
3. read the data using the confirmed semantics;
4. run hard structural assertions;
5. write standard output files.

The final output directory contains only standard files and `data_contract.json`.
Preview evidence, read plans, validation details, profiles, logs, and decision
traces are not persisted by default.

## Status

- `provisional`: critical semantics are not confirmed; do not formally read.
- `confirmed`: critical semantics are confirmed and reading may execute.
- `ready`: standard output files were written and passed reader assertions.
- `blocked`: a hard assertion or missing input prevents usable output.

If the reader cannot guarantee reliable output, return `blocked` or
`needs_confirmation`; do not return low-confidence ready data.

## Boundaries

Reader does not run QC, split, preprocess, feature engineering, modeling,
optimization, or reporting. Development checks such as `check_consistency`,
`server_health`, pytest, and plugin checks are release checks, not normal
reading workflow steps.

# Spectral Reader Confirmation Gates

Users confirm semantics, not implementation stages.

Confirm when the reader cannot safely decide:

- leading preamble skip;
- sample orientation;
- label or target column;
- sample ID source;
- spectral column boundaries;
- folder or filename semantics;
- external label file, join key, label column, metadata columns, and alignment
  policy;
- spectral type when it affects meaning.

## Status Semantics

- `provisional`: unresolved semantic confirmation remains; do not formally
  read.
- `confirmed`: critical semantics are confirmed by the user or deterministic
  rules; output is reliable reader output.
- `blocked`: missing file, unreadable path, missing column, unsupported read
  mode, invalid non-missing X values, row-count mismatch, band-axis mismatch,
  duplicate label keys, or missing required external labels prevents a usable
  output.

Missing X cells are not blocked when the matrix shape is still rectangular.
Keep them as missing values for QC. Missing sample IDs require confirmation
before generated IDs are used.

Low confidence is not a downstream concept. If uncertainty matters, keep the
plan `provisional` or block; do not produce low-confidence `confirmed`.

## Workflow Behavior

`draft_plan` may return `needs_confirmation`. Formal execution modes must stop
before reading unless validation computes `confirmed`. Normal output should
show only required confirmations, ready output summary, or blocked reasons.

# Failure And Recovery Cases

Failures should be short and directly actionable.

## Blocked User Output

```json
{
  "status": "blocked",
  "errors": [{"code": "COLUMN_NOT_FOUND", "message": "Declared label column does not exist."}],
  "next_step_hint": "Fix the reader input or confirm the correct label column."
}
```

## Common Blocking Reasons

- source path is unreadable;
- required column is missing;
- X contains non-numeric values;
- y, sample IDs, or metadata rows do not match X rows;
- band axis length does not match X columns;
- required supervised y is missing;
- external label join key is missing;
- external label file contains duplicate join keys;
- required spectra samples are missing labels;
- requested layout is not supported.

## Recovery Rules

Repair the parameters or source artifact first. Do not reread raw data through
another public workflow, infer replacement columns, drop rows, generate labels,
run QC, split, preprocess, feature engineering, modeling, or optimization.

Optional warnings such as `SMALL_SAMPLE_RISK` may remain as downstream
warnings; they are not reader-blocking failures.

## External Label Blocked Examples

Duplicate key:

```json
{"status": "blocked", "errors": [{"code": "DUPLICATE_LABEL_KEYS"}]}
```

Missing required label:

```json
{"status": "blocked", "errors": [{"code": "MISSING_REQUIRED_LABELS"}]}
```

Repair the label file or alignment parameters, then rerun `read_spectral_dataset`.

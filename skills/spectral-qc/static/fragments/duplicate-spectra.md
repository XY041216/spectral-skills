# Duplicate Spectra

Duplicate spectra and near-duplicate spectra are QC concerns, not reader
concerns.

## Check

Use three tiers. Do not collapse all high-correlation spectra into duplicate
conflicts.

### Exact Duplicate

`X.csv` rows are exactly identical.

- Same label: warning; suggest keeping all rows, merging scans, or deleting
  confirmed duplicate representatives after user confirmation.
- Different labels: blocked; require manual review before downstream use or
  cleaning.

### Strict Near Duplicate

Rows are not exactly identical, but satisfy multiple strict criteria:

- Pearson correlation > 0.9999;
- cosine similarity > 0.9999;
- relative RMSE < 1e-3;
- SAM angle < 0.01 radians.

Same-label strict near duplicates are leakage risks and should usually lead to
group-aware splitting or user-confirmed duplicate handling. Cross-label strict
near duplicates require review before cleaning.

### Global High Similarity

Rows have high correlation or cosine similarity but do not satisfy strict
near-duplicate criteria. This usually means global spectral-shape similarity,
class overlap, or subtle local spectral differences. Record this separately as
`global_similarity_risk`, not as the primary duplicate result. Do not block or
delete by default.

Recommended compact output:

```json
{
  "duplicate_check": {
    "status": "passed",
    "exact_duplicate_pairs": 0,
    "exact_duplicate_label_conflicts": 0,
    "strict_near_duplicate_pairs": 0,
    "strict_near_duplicate_label_conflicts": 0,
    "interpretation": "Exact and strict near-duplicate checks only.",
    "recommended_action": "none",
    "recommended_split_strategy": "stratified_or_group_aware_if_replicates_exist"
  },
  "global_similarity_risk": {
    "status": "warning",
    "high_similarity_pairs": 6735,
    "cross_label_high_similarity_pairs": 5051,
    "interpretation": "Global spectral-shape similarity is not confirmed duplication.",
    "recommended_action": "do_not_delete_by_default",
    "recommended_next_step": "continue_to_splitter; consider preprocessing, PCA, or feature selection during modeling"
  }
}
```

## Action

Do not delete duplicates automatically. Ask the user to confirm whether to keep
all rows, mark candidates, use group-aware splitting, or remove selected
exact/strict duplicate samples.

Duplicate `sample_id` values are not a QC duplicate-spectra issue. They are a
structural package integrity error and should be blocked before QC computation.

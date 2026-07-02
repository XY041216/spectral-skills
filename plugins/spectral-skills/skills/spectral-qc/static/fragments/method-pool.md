# QC Method Pool

Use the standard scheme by default for non-destructive `check` and `mark`
modes. Do not ask the user to choose algorithms just to inspect quality.

## Standard Scheme

- `contract_consistency_check`
- `missing_check`
- `non_numeric_check`
- `band_axis_check`
- `constant_band_check`
- `low_variance_band_check`
- `label_distribution_check`
- `sample_intensity_quality`
- `roughness_check`
- `spike_check`
- `baseline_drift_check`
- `similarity_to_mean`
- `classwise_similarity`
- `near_duplicate_check`
- `pca_hotelling_t2`
- `pca_q_residual`
- `mahalanobis_on_pca`

Write the standard abnormal-sample strategy into `qc_result.json` as
`outlier_detection.strategy = "standard"`, with `methods_run` and each
candidate's `triggered_by` list. The user-facing summary should name the
methods that were actually used.

## Supported Outlier Families

- `robust_zscore`
- `iqr`
- `mad`
- `similarity_to_mean`
- `classwise_similarity`
- `spike_detection`
- `baseline_drift_score`
- `pca_hotelling_t2`
- `pca_q_residual`
- `mahalanobis_on_pca`
- `near_duplicate_check`
- `multi_method_consensus`
- `half_resampling_outlier`
- `mccv_outlier`

## Advanced Resampling Outlier Control

Do not run HR or MCCV in default `check`. Use them only when the user asks for
advanced abnormal-sample control, repeated misclassification/misprediction
stability, or deletion of high-risk abnormal samples after confirmation.

When the user asks only for "outlier detection", offer: default comprehensive
strategy, PCA T2/Q/PCA-MD, MD/PCA-Mahalanobis, HR, MCCV, or custom. Recommend
the default comprehensive strategy unless the user explicitly wants a
resampling stability method.

Supported method IDs:

- `half_resampling_outlier`
- `mccv_outlier`

Shared outcome defaults:

- `outlier_metric`: `misclassification_frequency` for classification or
  `mean_absolute_residual` for regression
- `threshold`: `percentile_95`

HR default recommendation:

- `n_resamples`: 100
- `sample_fraction`: 0.5
- `base_model`: auto (`svm` for classification, `pls` for regression)

MCCV default recommendation:

- `n_resamples`: 100
- `train_ratio`: 0.7
- `base_model`: auto (`svm` or `logistic_regression` for classification,
  `pls` for regression)

Classification metrics:

- `misclassification_frequency`
- `mean_predicted_probability_of_true_class`
- `prediction_instability`
- `classwise_error_frequency`

Regression metrics:

- `mean_absolute_residual`
- `standardized_residual`
- `residual_std`
- `high_leverage_frequency`

Write the compact summary into `qc_result.json` under
`resampling_outlier_control`; do not create extra files for these methods.
For classification, label MCCV/HR outputs as
`classification_instability_candidates` or `resampling_risk_samples`. These are
current-pipeline stability risks, not automatic spectral acquisition outliers.
Record evaluation counts and warn when sample validation counts are low. Do not
recommend deleting MCCV-only candidates; prefer mark-only or an intersection
with standard spectral QC outliers after manual review.

## Confirmed Cleaning Pool

Use these only in `clean` mode after the user confirms action, method,
threshold, and scope.

### Outlier Samples

Supported removal methods:

- `robust_zscore`
- `iqr`
- `mad`
- `similarity_to_mean`
- `classwise_similarity`
- `pca_hotelling_t2`
- `pca_q_residual`
- `mahalanobis_on_pca`
- `multi_method_consensus`
- `half_resampling_outlier`
- `mccv_outlier`

Recommended deletion scope: remove only `high_confidence_outliers`.

### Duplicate And Near-Duplicate Spectra

Supported strategies:

- mark only;
- remove exact duplicates and keep the first sample when labels match;
- remove near duplicates and keep one representative;
- merge replicate scans by averaging spectra when labels are consistent;
- emit group recommendations for splitter instead of deleting.

Prefer group-aware splitting for near duplicates that may be replicate scans.
Block cleaning when duplicate or near-duplicate spectra have conflicting labels.

### Bad Bands

Supported deletion targets:

- constant bands;
- severe low-variance bands;
- high-missing-rate bands;
- high-spike-frequency bands;
- user-specified band ranges.

After deleting bands, update `X.csv`, `band_axis.csv`, `data_contract.json`
`shape.n_features`, top-level `n_features`, and `band_axis.count` together.

### Missing Values

Supported strategies:

- `drop_samples_by_missing_rate`
- `drop_bands_by_missing_rate`
- `band_mean_imputation`
- `band_median_imputation`
- `linear_interpolation`
- `nearest_interpolation`

For sparse missing values on a continuous spectral axis, prefer linear
interpolation or median imputation after confirmation.

### Spikes

Supported strategies:

- `mark_only`
- `hampel_filter`
- `moving_median_replace`
- `local_mad_replace`
- remove severe spike samples

Route smoothing-style requests such as SG smoothing to `spectral-preprocess`.

Legacy method aliases such as `NOE`, `MD`, `PCA_DISTANCE`, `ROBUST_ZSCORE`,
`IQR`, `MAD`, `HR`, and `MCCV` may still be accepted by scripts for
compatibility.

Compatibility vocabulary: Robust Z-score, PCA Hotelling T2, Q residual,
Class-aware checks, HR, MCCV, and PLS residual may appear in older design notes.
Treat HR and MCCV as advanced mark/clean methods, not as default checks.

## Not In Current Required Pool

Do not introduce these as default QC methods in this version:

- Isolation Forest
- One-Class SVM
- DBSCAN or HDBSCAN
- LOF
- deep anomaly detection

## Recommendation Rule

When uncertain, recommend non-destructive marking first. Use confirmed cleaning
only after the user chooses action, method, threshold, and scope.

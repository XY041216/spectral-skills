# Outlier Candidates

Outlier detection in spectral-check produces candidates, not automatic deletions.

Record the strategy in `qc_result.json`:

- `outlier_detection.strategy`
- `outlier_detection.methods_run`
- `outlier_detection.high_confidence_outliers[].triggered_by`
- `outlier_detection.medium_confidence_outliers[].triggered_by`

## Sample Candidates

Possible checks include:

- intensity range or total signal summaries;
- robust Z-score, IQR, or MAD on sample summaries;
- severe spike candidates; keep minor and moderate spike candidates as
  warnings only;
- PCA Hotelling T2;
- PCA Q residual;
- MD when covariance is stable;
- PLS residual or MCCV when modeling context and enough samples exist.

## Band Candidates

Possible checks include:

- missing-rate outlier bands;
- constant or low-variance bands;
- spike-like bands;
- bands with abnormal distribution compared with neighboring bands.

## Destructive Actions

Deleting samples or bands requires confirmation. Explain that candidate status
does not prove an error; it only flags risk for user review.

## Spike Severity

Report spike candidates as `minor`, `moderate`, or `severe`.

- Minor/moderate: warning only; may reflect real absorption peaks or ordinary
  local spectral variation.
- Severe: require local spike evidence plus independent roughness, PCA Q
  residual, or similarity evidence. Severe candidates may enter sample outlier
  reasons, but still require user confirmation before repair or deletion.

Do not recommend deleting medium-confidence outliers by default. Use this
language:

- high-confidence outliers: review manually; eligible as clean candidates after
  confirmation;
- medium-confidence outliers: mark only unless the user confirms a stronger
  method and scope;
- low-confidence outliers: keep in `qc_result.json`, but do not emphasize in
  the user-facing summary unless asked.

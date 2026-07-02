# Confirmation Gates

Run checks and candidate marks without confirmation. Ask for explicit user
confirmation before changing data.

Default boundary:

- `check` and `mark`: inspect, mark, and recommend only; do not modify data.
- `clean`: after confirmation, perform limited QC cleaning and write a new
  standard package under `cleaned_package/`.

## No Confirmation Required

These operations can run directly:

- package consistency check;
- missing-value, illegal-value, constant-band, and low-variance checks;
- band-axis checks;
- label distribution or regression target checks;
- sample intensity, roughness, spike, baseline drift, similarity, duplicate,
  and PCA outlier candidate scoring;
- writing `qc_result.json`.

Candidate detection is not a destructive edit.

## Confirmation Required

Require confirmation before:

- filling missing values;
- deleting samples;
- deleting bands;
- deleting duplicate spectra candidates;
- deleting target outlier candidates;
- deleting minority-class samples;
- repairing spikes or replacing spectral values;
- merging duplicate or replicate spectra;
- rewriting labels or targets;
- writing `cleaned_package/`.

## Two-Level Gate

First confirm the action, then confirm method and parameters.

If the user requests a destructive action but omits method, ask for the method.
If the user gives method but omits parameters, recommend parameters and wait.
If the user gives method and parameters, restate the exact action and wait for
confirmation.

For "detect outlier samples" or "abnormal sample detection", offer method
choices before running an advanced method:

- A. default comprehensive strategy: Robust Z-score, PCA Hotelling T2, Q
  residual, PCA-MD, similarity, spike, and baseline drift;
- B. PCA chemometric strategy: T2, Q residual, and PCA-MD;
- C. MD/PCA-Mahalanobis in PCA score space;
- D. HR half resampling;
- E. MCCV Monte Carlo cross-validation;
- F. custom method and threshold.

Default `check` may run A directly for observation. HR and MCCV are advanced
`mark` methods and should not run unless the user confirms them.

For "delete outlier samples", offer:

- A. default comprehensive strategy, delete only high-confidence outliers;
- B. PCA chemometric strategy, T2/Q/PCA-MD;
- C. MD/PCA-Mahalanobis with confirmed threshold;
- D. robust statistics, Robust Z-score/MAD/IQR;
- E. HR half resampling;
- F. MCCV Monte Carlo cross-validation;
- G. intersection of spectral QC outliers and HR/MCCV risk;
- H. mark only or custom method/threshold.

Before deleting, require all four decisions: method, threshold, deletion scope,
and whether to write `cleaned_package/` plus `qc_cleaning_log.json`. Recommend A
or mark-only. Do not recommend deleting medium-confidence outliers unless the
user gives a specific reason and confirms the scope.
Do not recommend directly deleting MCCV-only or HR-only samples; they may be
classification-boundary or pipeline-instability cases rather than bad spectra.

If the user asks for advanced outlier control, offer HR half resampling, MCCV
Monte Carlo cross-validation, or a custom rule.

Do not default to HR or MCCV when the user only says "delete outlier samples".
If the user selects MCCV, recommend 100 random resamples, train ratio 0.7, and
validation-set misclassification frequency for classification tasks or mean
absolute residual for regression tasks. Ask the user to confirm that only
high-risk samples will be deleted before running `clean`.

For "delete bad bands", offer constant/low-variance bands, missing-rate bands,
spike-frequency bands, edge-noise bands, specified ranges, or custom threshold.

For missing values, offer removing high-missing samples/bands, band mean,
band median, linear interpolation, KNN, or custom handling.

For spikes, offer mark-only, Hampel repair, moving median repair, or route
smoothing-like handling to `spectral-preprocess`. Do not repair or delete all
spike candidates by default; minor and moderate spike candidates are warnings.

For duplicate spectra, prefer group-aware splitting over deletion when the
duplicates may be replicate scans.

Never automatically modify labels. Mark suspected mislabels and ask for manual
review or user-provided corrected labels.

## Duplicate Cleaning Gate

For "delete duplicate spectra" or "delete near duplicates", ask the user to
choose a strategy:

- remove only exact duplicates with matching labels and keep the first sample;
- remove near duplicates and keep one representative;
- merge replicate scans by averaging spectra when labels are consistent;
- do not delete, but emit group recommendations for splitter;
- custom threshold and retention rule.

If exact or near-duplicate spectra have different labels, block cleaning and
require manual review. Do not automatically delete, merge, or relabel them.

## Bad Band Cleaning Gate

For "delete bad bands", ask for method and threshold:

- constant bands;
- severe low-variance bands;
- high-missing-rate bands;
- high spike-frequency bands;
- user-specified band range;
- custom rule.

After confirmed band deletion, update `X.csv`, `band_axis.csv`, and every
feature-count field in `data_contract.json` together.

## Missing And Spike Cleaning Gate

For missing values, ask whether to drop high-missing samples/bands or impute
with band mean, band median, linear interpolation, or nearest interpolation.

For spikes, prefer marking first. If the user confirms repair, require the
method, window, threshold, and affected scope. Route SG smoothing to
`spectral-preprocess`. Severe spike status should already require corroboration
from roughness, PCA Q residual, or similarity evidence; do not treat all local
spike candidates as severe.

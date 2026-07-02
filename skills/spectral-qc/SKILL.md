---
name: spectral-qc
description: >-
  Use when Codex needs to inspect, explain, mark, or user-confirm quality
  control actions on an existing spectral data package produced by
  spectral-reader: data_contract.json, X.csv, sample_ids.csv, band_axis.csv,
  optional y.csv, and optional metadata.csv. This skill is for pre-modeling
  spectral QC such as package consistency, missing or illegal values, spectral
  axis risks, class or target risks, constant or low-variance bands, spike and
  baseline-drift risks, similarity/outlier candidates, near duplicates, PCA
  Hotelling T2/Q-residual candidates, optional HR/MCCV resampling outlier
  stability marking, and confirmed cleaning before splitting, preprocessing,
  feature engineering, or modeling. Do not use this skill to
  read raw source files, split data, preprocess spectra, engineer features,
  train models, or develop the spectral-qc skill itself.
---

# Spectral QC

`spectral-qc` performs pre-modeling quality control on the standard spectral
package written by `spectral-reader`.

It answers three questions:

- whether the package can enter splitting, preprocessing, feature engineering,
  and modeling;
- whether samples, bands, labels, or package contracts have quality or leakage
  risks;
- which data-changing cleaning actions require user confirmation.

## Activation Boundary

Use this skill only for quality control on an existing standard spectral
package:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- optional `y.csv`
- optional `metadata.csv`

Do not use this skill to read raw source files. Do not use this skill to read raw files.
Do not infer source layouts, align external labels, split datasets,
preprocess spectra, engineer features, train models, tune models, write
reports, or develop the skill itself.

## Core Flow

Follow this flow:

`reader package -> run standard QC -> write qc_result.json -> summarize risks -> ask confirmation only for destructive cleaning -> optionally write cleaned_package/`

Default behavior is conservative: check and mark only. `spectral-qc` does not
modify data in `check` or `mark` mode. When the user explicitly requests
cleaning and confirms the action, method, threshold, and scope, `clean` mode may
perform limited QC cleaning and write a new standard package for downstream
skills.

Use one executable entry for deterministic work:

```bash
python skills/spectral-qc/scripts/qc_spectral_package.py --package-dir <reader-output> --mode check --output-dir <qc-output> --json
```

Modes:

- `check`: run the standard non-destructive QC scheme.
- `mark`: run the standard QC scheme and optional advanced method-pool outlier
  scoring, including HR/MCCV when explicitly needed; do not delete or repair.
- `clean`: require confirmation, then write `cleaned_package/` and
  `qc_cleaning_log.json`. Use the cleaned package as the downstream package.
- `methods`: list the implemented first-stage method pool.

Legacy aliases `outliers` and `apply` map to `mark` and `clean`.

## Standard QC Scheme

Run the standard scheme by default; do not ask the user to choose algorithms for
non-destructive inspection.

The standard scheme includes:

- contract consistency check;
- missing, NaN, inf, non-numeric, all-zero, and extreme-range checks;
- numeric band-axis monotonicity, duplicate, gap, and unit checks;
- constant and low-variance band checks;
- classification label distribution or regression target checks;
- sample intensity, roughness, first/second-difference, and adjacent-jump
  checks;
- spike candidates using local MAD/Hampel-style scoring;
- baseline slope, curvature, and mean-offset drift checks;
- similarity to mean spectrum and classwise mean spectrum;
- duplicate spectra in three tiers: exact duplicates, strict near duplicates,
  and separate global high-similarity risk;
- PCA Hotelling T2, Q residual, and Mahalanobis-on-PCA candidate detection.

Do not output a PCA feature package from QC.

`qc_result.json` must expose the standard abnormal-sample strategy under
`outlier_detection`: include `strategy`, `methods_run`, and each high/medium
candidate with `sample_id`, `risk_level`, and `triggered_by`. This lets the
agent explain, for example, that one sample was triggered by PCA-MD and Q
residual while another was triggered only by Q residual.

HR/MCCV resampling methods are not part of default `check`. Use them only as
advanced abnormal-sample stability controls in `mark` mode, or in `clean` mode
after the user confirms method, threshold, and deletion scope.

Supported advanced method IDs:

- `half_resampling_outlier`
- `mccv_outlier`

Default recommendations:

- HR: 100 half-resamples, sample fraction 0.5, base model auto (`svm` for
  classification, `pls` for regression), threshold `percentile_95`.
- MCCV: 100 random resamples, train ratio 0.7, base model auto, threshold
  `percentile_95`.
- Classification scoring: prefer `misclassification_frequency`; optional
  metrics are `mean_predicted_probability_of_true_class`,
  `prediction_instability`, and `classwise_error_frequency`.
- Regression scoring: prefer `mean_absolute_residual`; optional metrics are
  `standardized_residual`, `residual_std`, and `high_leverage_frequency`.

For classification, treat HR/MCCV candidates as
`classification_instability_candidates` or `resampling_risk_samples`, not as
ordinary spectral acquisition outliers. They may be boundary samples,
class-overlap samples, suspected mislabels, or current-pipeline instability
under the current input package and base model. Record the input package/base
model and evaluation-count summary. Do not recommend deleting MCCV-only samples
unless manual review confirms a label or sample problem.

## Output Boundary

Observation-only modes write one lightweight file:

- `qc_result.json`

Default CLI JSON output and default `qc_result.json` must be summary-level:
status, shape, counts, top previews, method summaries, and recommended actions.
Do not print or write full `checks`, all pairwise similarity pairs, all
resampling scores, or all resampling details by default.

Use detail controls explicitly:

- `--json`: terminal summary JSON only;
- `--detail-level full` or `--verbose`: full result when debugging;
- `--export-details`: write `qc_details.json` with full details and keep
  `qc_result.json` compact.

Do not default to sample score CSVs, band score CSVs, candidate CSVs, debug
archives, visualizations, or full reports.

When the user confirms cleaning, write:

- `qc_result.json`
- `cleaned_package/`
- `qc_cleaning_log.json`

The cleaned package must keep the standard package filenames: `X.csv`,
`sample_ids.csv`, `band_axis.csv`, `data_contract.json`, and optional `y.csv`
and `metadata.csv`.

`qc_result.json` must identify the downstream package:

- no cleaning: `output_package: null` and
  `next_package_for_downstream: <input_package>`;
- confirmed cleaning: `output_package: <cleaned_package>` and
  `next_package_for_downstream: <cleaned_package>`.

## Confirmation Rules

Use two-level confirmation for destructive actions.

First identify the action:

- check or mark abnormal samples/bands: execute without confirmation;
- delete samples, delete bands, fill missing values, repair spikes, merge
  duplicates, modify labels, or write `cleaned_package/`: confirmation required.

Then confirm method and parameters:

- algorithm or rule;
- threshold;
- global or classwise scope;
- high-confidence only, intersection, union, or custom candidate rule;
- whether to preserve a cleaning log and write a cleaned package.

If the user only says "delete outliers", return a pending confirmation with a
recommended conservative strategy. If the user gives a method but no threshold,
recommend a threshold and wait. If the user gives method and threshold, restate
the exact cleaning action and wait for confirmation.

For "detect abnormal/outlier samples", offer method choices before running an
advanced strategy unless the user explicitly accepts the default:

- A. Default comprehensive strategy: Robust Z-score + PCA T2 + Q residual +
  PCA-MD + similarity + spike + baseline drift.
- B. PCA chemometric strategy: Hotelling T2 + Q residual + PCA-MD.
- C. MD/PCA-Mahalanobis only.
- D. HR half resampling.
- E. MCCV Monte Carlo cross-validation.
- F. Custom.

Explain that default `check` uses the standard strategy and does not run HR or
MCCV. Run HR/MCCV only in `mark` mode after the user confirms the method and
recommended parameters.

If a user asks "abnormal/outlier sample check" after a `qc_result.json` already
exists, first summarize the existing non-destructive `outlier_detection`
records. Add one sentence: "These results come from the standard comprehensive
QC strategy; HR/MCCV were not run by default. I can rerun mark mode with MD,
HR, or MCCV if you want a specific method." If the user names MD,
PCA-Mahalanobis, HR, or MCCV, run `mark` mode with the requested method instead
of only reprinting the existing result.

For "delete abnormal/outlier samples", ask the user to choose:

- A. Default comprehensive strategy, deleting only high-confidence outliers.
- B. PCA chemometric strategy: T2/Q/PCA-MD.
- C. MD/PCA-Mahalanobis with confirmed threshold.
- D. Robust statistics: Robust Z-score/MAD/IQR.
- E. HR half resampling.
- F. MCCV Monte Carlo cross-validation.
- G. Intersection of spectral QC outliers and HR/MCCV risk.
- H. Mark only or custom method/threshold.

Before clean mode, confirm all four elements: abnormal-sample detection method,
threshold, deletion scope, and writing `cleaned_package/`. Recommend A only
when the user explicitly wants cleaning; otherwise recommend mark-only.
Medium-confidence outliers should be marked only unless the user provides a
specific scientific reason and confirms the scope.

For advanced outlier stability control, offer:

- HR half resampling;
- MCCV Monte Carlo cross-validation;
- custom rule.

If the user chooses MCCV for a classification task, recommend: 100 random
resamples, train ratio 0.7, validation-set misclassification frequency as the
outlier score, and delete only high-risk samples. Ask for confirmation before
running `clean`.

QC may mark suspected mislabels, but it must not automatically modify labels.
Exact duplicate spectra with conflicting labels are blocking and require manual
review. Strict near-duplicate label conflicts require review before cleaning.
High-similarity cross-label spectra are warnings about class overlap or subtle
local differences, not confirmed duplicate conflicts.

Confirmed cleaning is limited to bounded QC actions:

- remove high-confidence outlier samples;
- remove exact duplicates or confirmed near-duplicate representatives;
- remove confirmed bad bands, with `X.csv`, `band_axis.csv`, and
  `data_contract.json` updated together;
- impute missing values with a confirmed strategy;
- repair confirmed spike points or remove severe spike samples.

Do not delete high-similarity spectra by default. Treat them as evidence that
classification may depend on preprocessing, local bands, or low-dimensional
structure rather than global spectral shape.

Do not repair or delete all spike candidates by default. Minor and moderate
spike candidates are warnings. Severe spike status requires local spike
evidence plus independent roughness, PCA Q residual, or similarity evidence.
If the user asks to repair spikes, offer mark-only, Hampel repair, moving median
repair, or routing smoothing-style work to `spectral-preprocess`.

Cap local spike scores when local MAD is near zero or the raw score is
unbounded. Record `score_capped`, `score_capped_count`, and
`local_mad_too_small_count`; do not show enormous raw spike scores to users.

## Agent Summary Style

After QC, summarize the result in a few sentences:

`QC complete. Data have 120 samples and 3401 bands. No missing values, illegal values, constant bands, or low-variance bands were found. No high-confidence outliers were found; 2 medium-risk samples were marked. The data can enter stratified splitting.`

For warning-only QC, explicitly say it does not block the next step:

`QC status is warning, not blocked. The package can enter splitter. Review high-confidence outliers if you want cleaner data, but do not delete medium-confidence or spike candidates without confirmation.`

For blocking issues, lead with the blocker:

`QC blocked downstream use: X.csv has 3401 columns but band_axis.csv has 3400 rows. Fix the standard package before splitting or modeling.`

When many spectra are globally similar, do not describe them as duplicated
sample conflicts unless exact or strict near-duplicate rules support that:

`QC warning: many spectra have highly similar global shape, including cross-label pairs. This more likely indicates class overlap or subtle local spectral differences than duplicated samples. No automatic deletion is recommended.`

Do not dump file internals unless the user asks.

## Read As Needed

- Use `static/core/qc-boundary.md` for scope and neighboring-skill boundaries.
- Use `static/core/standard-package.md` for package and output rules.
- Use `static/core/confirmation-gates.md` before changing data.
- Use `static/fragments/method-pool.md` for standard QC method families.
- Use `static/fragments/missing-values.md` for missing-value QC behavior.
- Use `static/fragments/outlier-candidates.md` for sample and band risk checks.
- Use `static/fragments/class-target-risks.md` for labels, targets, and class distribution.
- Use `references/qc-scenarios.md` for common user intents and expected behavior.

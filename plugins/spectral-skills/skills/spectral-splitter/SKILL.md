---
name: spectral-splitter
description: >-
  Use when Codex needs to split an existing standard spectral data package from
  spectral-reader or spectral-check into reproducible train, validation, and test
  assignments for downstream spectral-preprocess, spectral-feature, or
  spectral-modeling. This skill reads data_contract.json plus X.csv,
  sample_ids.csv, band_axis.csv, optional y.csv, and optional metadata.csv, and
  writes split_indices.csv and split_contract.json. It supports holdout,
  predefined, KFold/StratifiedKFold/LOOCV, repeated random/MCCV, Kennard-Stone,
  SPXY, Duplex, regression-binned, group-aware, and stratified-group splitting
  with fixed random seeds. Do not use this skill to read raw files, perform QC,
  remove outliers, preprocess spectra, engineer features, train models, or
  implement unsupported experimental designs such as time-series splitting or
  nested CV.
---

# Spectral Splitter

`spectral-splitter` creates reproducible train/validation/test assignments for
standard spectral packages.

It sits after `spectral-reader` or `spectral-check`, and before
`spectral-preprocess`, `spectral-feature`, and `spectral-modeling`.

After QC `check` or `mark`, split the original standard package. After
confirmed QC `clean`, split the QC `cleaned_package` standard package. Do not
pass a bare `qc_result.json` directory to splitter unless it also contains the
standard package files.

## Activation Boundary

Use this skill only for splitting an existing standard spectral package:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- optional `y.csv`
- optional `metadata.csv`

Do not use this skill to read raw CSV, Excel, MAT, NPZ, HDF5, or vendor files.
Do not use it for QC, missing-value handling, outlier removal, preprocessing,
feature engineering, modeling, optimization, reporting, logs, or debug systems.

## Core Flow

Follow this flow:

`reader/qc package -> check standard package integrity -> confirm ratio/method -> split sample ids -> write split_indices.csv + split_contract.json`

Use one executable entry for deterministic work:

```bash
python skills/spectral-splitter/scripts/split_spectral_package.py --package-dir <standard-package> --output-dir <split-output> --ratio 8:2 --method random --random-seed 42 --json
```

For classification packages, prefer stratified split:

```bash
python skills/spectral-splitter/scripts/split_spectral_package.py --package-dir <standard-package> --output-dir <split-output> --ratio 6:2:2 --method stratified --random-seed 42 --json
```

## Method Selection

Use the user's requested method when it is supported.

- When asking the user to choose or confirm a split, show the recommended split
  first, then show the full supported split-method menu. Do not present only
  common options. Every user-facing entry must be bilingual and include the
  executable method code, using `涓枃鍚嶇О锛坢ethod_code / English name锛塦.
- Supported split-method menu:
  - 闅忔満鐣欏嚭锛坮andom / random holdout锛?  - 鍒嗗眰鐣欏嚭锛坰tratified / stratified holdout锛?  - 棰勫畾涔夊垝鍒嗭紙predefined_split / predefined split锛?  - K 鎶樹氦鍙夐獙璇侊紙kfold / K-fold cross-validation锛?  - 鍒嗗眰 K 鎶樹氦鍙夐獙璇侊紙stratified_kfold / stratified K-fold cross-validation锛?  - 鐣欎竴娉曪紙leave_one_out / leave-one-out cross-validation锛?  - 钂欑壒鍗℃礇閲嶅鍒掑垎锛坢onte_carlo_cv / Monte Carlo repeated holdout锛?  - 閲嶅闅忔満鍒掑垎锛坮epeated_random_split / repeated random split锛?  - 鍒嗗眰钂欑壒鍗℃礇閲嶅鍒掑垎锛坰tratified_monte_carlo_cv / stratified Monte Carlo repeated holdout锛?  - Kennard-Stone 浠ｈ〃鎬у垝鍒嗭紙kennard_stone / Kennard-Stone split锛?  - SPXY 浠ｈ〃鎬у垝鍒嗭紙spxy / SPXY split锛?  - Duplex 浠ｈ〃鎬у垝鍒嗭紙duplex / Duplex split锛?  - 鍥炲綊鍒嗙鍒嗗眰鍒掑垎锛坮egression_stratified / regression-binned stratified split锛?  - y 鍒嗙鍒嗗眰鍒掑垎锛坹_binned_stratified / y-binned stratified split锛?  - 鍒嗙粍鍒掑垎锛坓roup / group split锛?  - 鍒嗙粍闃叉硠婕忓垝鍒嗭紙group_aware / group-aware split锛?  - 鍒嗗眰鍒嗙粍鍒掑垎锛坰tratified_group / stratified group split锛?- For classification default discussion, recommend `stratified` with `6:2:2`
  or `8:2` and `random_seed=42`.
- For regression default discussion, recommend `kennard_stone` with `8:2`.
  Recommend `spxy` only when the user emphasizes y coverage or representative
  calibration/test coverage.
- Recommend KFold, LOOCV, or MCCV only for small samples, stability evaluation,
  or explicit cross-validation requests.
- Use `random` when the user requests a random split or when no class labels are available.
- Use `stratified` only for classification packages with `y.csv`.
- Use `predefined_split` when the user provides existing split assignments,
  an external `split_indices.csv`, or a metadata split column.
- Use `kfold`, `stratified_kfold`, or `leave_one_out` for cross-validation.
  Default `n_splits` is 5; LOOCV is appropriate only for small samples or when
  explicitly requested.
- Use `monte_carlo_cv`, `repeated_random_split`, or
  `stratified_monte_carlo_cv` for repeated holdout/MCCV. Default
  recommendation is 100 repeats, train/test 0.7/0.3, seed 42.
- Use `kennard_stone` for representative X-space holdout selection.
- Use `spxy` only for regression because it uses y-distance.
- Use `duplex` for representative train/test construction.
- Use `regression_stratified` or `y_binned_stratified` for continuous-y
  quantile-binned splitting.
- Use `group` or `group_aware` when metadata groups, batches, origins, years,
  or replicate IDs should not leak across splits.
- Use `stratified_group` for classification tasks that need both class balance
  and group leakage protection.
- If method is omitted, request confirmation before using any split method.
  When classification labels are present, recommend stratified split.
- If the user asks for time-series split, chronological split, nested CV, or an
  unsupported optimizer-style split, state that it is not implemented.

## Ratio Rules

Require an explicit ratio if the user has not provided one. Accept common forms
such as `8:2`, `7:3`, `7:2:1`, and `6:2:2`.

Record split ratio, method, random seed, and decision source in
`split_contract.json`. Do not silently choose `6:2:2`, `8:2`, `random`, or
`stratified` for the user in an agent conversation.

Train and test must be non-empty. Validation may be zero only when the user
requests train/test splitting.

## Output Boundary

Write only split outputs by default:

- `split_indices.csv`
- `split_contract.json`
- optional human-facing `split_summary.json`

Do not rewrite `X.csv`, `y.csv`, `sample_ids.csv`, `band_axis.csv`, or
`metadata.csv`. Do not create default `train_X.csv`, `test_X.csv`, copied data
packages, logs, debug folders, audit folders, or inventory outputs.

Downstream skills should use `data_contract.json + split_contract.json` and read
the original standard package by sample ID or row index.

If a downstream skill or workflow only supports holdout contracts, stop before
modeling on `cross_validation` or `repeated_holdout` contracts and explain that
the split was created successfully but downstream CV/repeated evaluation support
must be confirmed.

`split_contract.json` supports three split types:

- `holdout`: include `indices` and `sample_ids` for train/val/test.
- `cross_validation`: include `folds` with train/val indices.
- `repeated_holdout`: include `repeats` with train/val/test indices.

Keep outputs light: do not write one file per fold or repeat.

Use a long-table `split_indices.csv` schema for every split type:

`split_type,method,fold_id,repeat_id,role,sample_index,sample_id,label,group_id`

Terminal output should remain a preview/summary; do not dump all folds or
repeats unless the user asks to inspect the file.

## Safety Gates

Block or request confirmation instead of writing unsafe splits when:

- `data_contract.json`, `X.csv`, `sample_ids.csv`, or `band_axis.csv` is missing.
- X row count does not match sample IDs.
- sample IDs are empty or duplicated.
- `y.csv` row count does not match X.
- stratified split is requested without classification labels.
- a class has too few samples to appear in every requested split.
- SPXY is requested for classification or without numeric y.
- group-aware split is requested without a usable metadata group column.
- X contains NaN or infinite values.
- predefined split has duplicate assignments, unknown split labels, unknown
  sample IDs, omitted samples, or empty train/test.
- ratios do not sum to 1 or produce empty train/test splits.
- generated assignments would duplicate or omit samples.

Splitter confirmation gates are lighter than QC, but must cover experiment
design risk. Ask for confirmation when the agent is choosing defaults, when
classification defaults to stratified, when a classification task asks for
random, when KS/SPXY/Duplex are used, when group-aware or stratified-group is
used, when CV replaces holdout, when y participates in splitting, or when QC
potential replicates are used as groups. If the user explicitly specifies
method, ratio/folds/repeats, and seed in a CLI-style request, the script may
execute directly.

## Read As Needed

- Use `static/core/split-boundary.md` for scope and neighboring-skill boundaries.
- Use `static/core/standard-package.md` for input package requirements.
- Use `static/core/output-contract.md` for split output and handoff rules.
- Use `static/fragments/method-selection.md` for MVP method selection.
- Use `static/fragments/safety-checks.md` for blocking and confirmation behavior.
- Use `references/split-scenarios.md` for common user intents.

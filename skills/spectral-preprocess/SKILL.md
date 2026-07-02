---
name: spectral-preprocess
description: >-
  Use when Codex needs to apply leakage-safe spectral preprocessing to an
  already split standard spectral data package from spectral-reader or
  spectral-qc plus split_contract.json from spectral-splitter. Methods that
  learn global parameters are fit on train samples only; per-sample methods
  such as SNV do not fit train-set statistics. For cross-validation or
  repeated-holdout contracts, train-fit methods are refit per fold/repeat. It
  writes preprocess_contract.json for holdout and fold/repeat-wise workflows.
  It supports first-stage none, SNV,
  MSC, detrend, SNV+detrend, Savitzky-Golay smoothing and derivatives, moving
  average, Gaussian smoothing, median filtering, baseline correction,
  common scaling/normalization, absorbance conversion, and physical band-range
  preprocessing. Do not use this skill to read raw files,
  perform QC, split data, remove samples, select features, run PCA/CARS/SPA,
  train models, tune preprocessing by model performance, or create reports,
  logs, debug archives, or optimizer searches.
---

# Spectral Preprocess

`spectral-preprocess` transforms spectra after `spectral-splitter` and before
`spectral-feature` or `spectral-modeling`.

## Activation Boundary

Use this skill only with a standard spectral package and a split contract:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- optional `y.csv`
- optional `metadata.csv`
- `split_contract.json`

Accept `holdout`, `cross_validation`, and `repeated_holdout` split contracts.
For `holdout`, write one standard package plus `preprocess_contract.json`. For
`cross_validation` and `repeated_holdout`, loop through partitions and write
`preprocess_contract.json` plus `iterations/<fold_or_repeat>/...` outputs.

Do not read raw CSV, Excel, MAT, NPZ, HDF5, or vendor files. Do not run QC,
delete samples, split data, select features, train models, tune preprocessing
from model scores, or write reports/log/debug systems.

## Core Flow

Follow this flow:

`reader/qc package + split contract -> confirm method sequence -> fit train-only parameters only for train-fit methods -> apply per-sample/stateless methods independently -> transform partition samples -> write standard package or iteration contract`

Use one executable entry for deterministic work:

```bash
python skills/spectral-preprocess/scripts/preprocess_spectral_package.py --package-dir <standard-package> --split-contract <split-output/split_contract.json> --output-dir <preprocess-output> --methods snv,standardization --json
```

For Savitzky-Golay smoothing or derivatives, provide parameters:

```bash
python skills/spectral-preprocess/scripts/preprocess_spectral_package.py --package-dir <standard-package> --split-contract <split_contract.json> --output-dir <preprocess-output> --methods sg_smoothing,first_derivative --window-length 11 --polyorder 2 --json
```

For high-impact transforms, confirm the intent explicitly:

- baseline methods: pass `--confirm-baseline`.
- absorbance conversion: pass `--confirm-absorbance` after confirming all
  values are positive reflectance/transmittance.
- band range selection/removal: pass `--confirm-band-change` and provide
  `--band-range` or `--remove-band-ranges`.

## Leakage Rule

Fit methods that learn global parameters on train samples only. Apply the same
parameters to validation and test samples.

Train-fit methods:

- `mean_centering`
- `standardization`
- `msc`
- `minmax_scaling`
- `robust_scaling`
- `pareto_scaling`

Per-sample or deterministic methods:

- `none`
- `snv` (per-sample; each spectrum uses its own mean and standard deviation)
- `detrend`
- `snv_detrend`
- `sg_smoothing`
- `first_derivative`
- `second_derivative`
- `moving_average`
- `gaussian_smoothing`
- `median_filter`
- `linear_baseline`
- `polynomial_baseline`
- `rubberband_baseline`
- `als_baseline`
- `l2_normalization`
- `area_normalization`
- `max_abs_normalization`
- `reflectance_to_absorbance`
- `transmittance_to_absorbance`
- `log_transform`
- `band_range_select`
- `remove_band_ranges`

If `split_contract.json` is missing and the user requests a train-fit method,
return `needs_confirmation`; do not fit on all samples unless the user confirms
an unsupervised whole-dataset transform.

## Method Scope

Supported methods are `none`, `snv`, `msc`, `detrend`, `snv_detrend`,
`sg_smoothing`, `first_derivative`, `second_derivative`, `moving_average`,
`gaussian_smoothing`, `median_filter`, `linear_baseline`,
`polynomial_baseline`, `rubberband_baseline`, `als_baseline`,
`mean_centering`, `standardization`, `minmax_scaling`, `robust_scaling`,
`pareto_scaling`, `l2_normalization`, `area_normalization`,
`max_abs_normalization`, `reflectance_to_absorbance`,
`transmittance_to_absorbance`, `log_transform`, `band_range_select`, and
`remove_band_ranges`.

When asking the user to choose or confirm preprocessing, show the full
supported preprocessing menu in bilingual form. Every entry must include the
executable method code using `中文名称（method_code / English name）`:
show the full supported preprocessing menu in bilingual form.

- 无预处理（none / no preprocessing）
- 标准正态变量校正（snv / Standard Normal Variate, SNV）
- 多元散射校正（msc / Multiplicative Scatter Correction, MSC）
- 去趋势（detrend / detrending）
- SNV + 去趋势（snv_detrend / SNV plus detrending）
- Savitzky-Golay 平滑（sg_smoothing / Savitzky-Golay smoothing）
- 一阶导数（first_derivative / first derivative）
- 二阶导数（second_derivative / second derivative）
- 移动平均平滑（moving_average / moving average smoothing）
- 高斯平滑（gaussian_smoothing / Gaussian smoothing）
- 中值滤波（median_filter / median filter）
- 线性基线校正（linear_baseline / linear baseline correction）
- 多项式基线校正（polynomial_baseline / polynomial baseline correction）
- 橡皮筋基线校正（rubberband_baseline / rubberband baseline correction）
- 非对称最小二乘基线校正（als_baseline / asymmetric least squares baseline correction）
- 均值中心化（mean_centering / mean centering）
- 标准化（standardization / standardization）
- 最小-最大缩放（minmax_scaling / min-max scaling）
- 鲁棒缩放（robust_scaling / robust scaling）
- Pareto 缩放（pareto_scaling / Pareto scaling）
- L2 归一化（l2_normalization / L2 normalization）
- 面积归一化（area_normalization / area normalization）
- 最大绝对值归一化（max_abs_normalization / max-absolute normalization）
- 反射率转吸光度（reflectance_to_absorbance / reflectance to absorbance）
- 透射率转吸光度（transmittance_to_absorbance / transmittance to absorbance）
- 对数变换（log_transform / log transform）
- 保留物理波段范围（band_range_select / band range selection）
- 移除物理波段范围（remove_band_ranges / remove band ranges）

If the user does not specify methods, ask them to choose from the supported MVP
set. If the user requests OSC, EMSC, wavelet denoising, airPLS/arPLS,
continuum removal, resampling the band axis, PCA, SPA, CARS, VIP, UVE,
automated search, model-linked selection, or optimization, state that it is
outside this preprocess version.

Recommended method-order rule:

1. `band_range_select` / `remove_band_ranges`
2. `reflectance_to_absorbance` / `transmittance_to_absorbance`
3. smoothing / denoising
4. baseline correction / `detrend`
5. scatter correction: `snv`, `msc`, or `snv_detrend`
6. derivatives
7. scaling: `mean_centering`, `standardization`, `minmax_scaling`,
   `robust_scaling`, or `pareto_scaling`

Warn if a chain includes both `snv` and `msc`, because both are scatter
corrections. Warn if a chain has more than five active methods.

For modeling workflows, `none` is an explicit baseline preprocessing decision.
Do not skip preprocessing implicitly. If the user asks to build a model but has
not specified whether to preprocess, return `needs_confirmation` with
alternatives such as `none`, `snv`, `msc`, `sg_smoothing`, and common method
sequences.

## Output Boundary

Write a new standard package:

- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- `data_contract.json`
- optional `y.csv`
- optional `metadata.csv`
- `preprocess_state.json`

Do not write `X_preprocessed.csv`, train/test matrix copies, reports, logs,
audit folders, debug folders, or optimizer inventories.

Downstream skills should use `preprocess_output/data_contract.json` plus the
original `split_contract.json` only for legacy compatibility. Prefer
`preprocess_output/preprocess_contract.json` for both holdout and
fold/repeat-wise workflows.

Write confirmation status and selected methods into `preprocess_state.json` and
the output `data_contract.json`.

## Safety Gates

Return `blocked` or `needs_confirmation` instead of writing outputs when:

- required standard package files are missing or misaligned.
- `split_contract.json` is missing for train-fit methods.
- split indices duplicate, omit, or reference unknown samples.
- train split is empty.
- SG/derivative parameters are missing or invalid.
- baseline correction is requested without confirmation.
- absorbance conversion is requested without confirmation or includes
  non-positive values.
- band range selection/removal is requested without confirmation or removes
  every band.
- X contains missing or non-numeric values.
- a requested method would change sample count.

## Read As Needed

- Use `static/core/preprocess-boundary.md` for neighboring-skill boundaries.
- Use `static/core/leakage-rules.md` for train-only fit behavior.
- Use `static/core/output-contract.md` for handoff rules.
- Use `static/fragments/method-selection.md` for MVP method choices.
- Use `static/fragments/safety-checks.md` for blocked and confirmation cases.
- Use `references/preprocess-scenarios.md` for common user intents.

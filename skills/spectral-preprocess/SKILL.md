---
name: spectral-preprocess
description: >-
  Use when Codex needs to apply leakage-safe spectral preprocessing to an
  already split standard spectral data package from spectral-reader or
  spectral-check plus split_contract.json from spectral-splitter. Methods that
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
executable method code using `涓枃鍚嶇О锛坢ethod_code / English name锛塦:
show the full supported preprocessing menu in bilingual form.

- 鏃犻澶勭悊锛坣one / no preprocessing锛?- 鏍囧噯姝ｆ€佸彉閲忔牎姝ｏ紙snv / Standard Normal Variate, SNV锛?- 澶氬厓鏁ｅ皠鏍℃锛坢sc / Multiplicative Scatter Correction, MSC锛?- 鍘昏秼鍔匡紙detrend / detrending锛?- SNV + 鍘昏秼鍔匡紙snv_detrend / SNV plus detrending锛?- Savitzky-Golay 骞虫粦锛坰g_smoothing / Savitzky-Golay smoothing锛?- 涓€闃跺鏁帮紙first_derivative / first derivative锛?- 浜岄樁瀵兼暟锛坰econd_derivative / second derivative锛?- 绉诲姩骞冲潎骞虫粦锛坢oving_average / moving average smoothing锛?- 楂樻柉骞虫粦锛坓aussian_smoothing / Gaussian smoothing锛?- 涓€兼护娉紙median_filter / median filter锛?- 绾挎€у熀绾挎牎姝ｏ紙linear_baseline / linear baseline correction锛?- 澶氶」寮忓熀绾挎牎姝ｏ紙polynomial_baseline / polynomial baseline correction锛?- 姗＄毊绛嬪熀绾挎牎姝ｏ紙rubberband_baseline / rubberband baseline correction锛?- 闈炲绉版渶灏忎簩涔樺熀绾挎牎姝ｏ紙als_baseline / asymmetric least squares baseline correction锛?- 鍧囧€间腑蹇冨寲锛坢ean_centering / mean centering锛?- 鏍囧噯鍖栵紙standardization / standardization锛?- 鏈€灏?鏈€澶х缉鏀撅紙minmax_scaling / min-max scaling锛?- 椴佹缂╂斁锛坮obust_scaling / robust scaling锛?- Pareto 缂╂斁锛坧areto_scaling / Pareto scaling锛?- L2 褰掍竴鍖栵紙l2_normalization / L2 normalization锛?- 闈㈢Н褰掍竴鍖栵紙area_normalization / area normalization锛?- 鏈€澶х粷瀵瑰€煎綊涓€鍖栵紙max_abs_normalization / max-absolute normalization锛?- 鍙嶅皠鐜囪浆鍚稿厜搴︼紙reflectance_to_absorbance / reflectance to absorbance锛?- 閫忓皠鐜囪浆鍚稿厜搴︼紙transmittance_to_absorbance / transmittance to absorbance锛?- 瀵规暟鍙樻崲锛坙og_transform / log transform锛?- 淇濈暀鐗╃悊娉㈡鑼冨洿锛坆and_range_select / band range selection锛?- 绉婚櫎鐗╃悊娉㈡鑼冨洿锛坮emove_band_ranges / remove band ranges锛?
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

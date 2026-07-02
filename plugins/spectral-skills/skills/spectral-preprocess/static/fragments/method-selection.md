# Method Selection

Supported methods:

When asking the user to confirm preprocessing, show the recommended method
first but also show the full supported preprocessing-method menu. Do not show
only `none`, `snv`, `msc`, derivatives, and standardization. Use these visible
headings:

- `Recommended preprocessing`
- `Supported preprocessing methods`
- `Methods requiring extra parameters or caution`
- `You may choose`

- Every user-facing entry must be bilingual and include the executable method
  code, using `中文名称（method_code / English name）`. English-only preprocessing
  menus are invalid.
- 无预处理（`none` / no preprocessing）: copy the standard package shape with no spectral transform.
- 标准正态变量校正（`snv` / Standard Normal Variate, SNV）: per-spectrum Standard Normal Variate.
- 多元散射校正（`msc` / Multiplicative Scatter Correction, MSC）: fit the reference spectrum on train samples, then correct all samples.
- 去趋势（`detrend` / detrending）: remove a per-spectrum linear trend while preserving the row mean.
- SNV + 去趋势（`snv_detrend` / SNV plus detrending）: apply SNV, then per-spectrum detrend.
- Savitzky-Golay 平滑（`sg_smoothing` / Savitzky-Golay smoothing）: smoothing with confirmed `window_length` and `polyorder`.
- 一阶导数（`first_derivative` / first derivative）: Savitzky-Golay derivative order 1.
- 二阶导数（`second_derivative` / second derivative）: Savitzky-Golay derivative order 2.
- 移动平均平滑（`moving_average` / moving average smoothing）: deterministic per-spectrum moving average smoothing.
- 高斯平滑（`gaussian_smoothing` / Gaussian smoothing）: deterministic per-spectrum Gaussian smoothing.
- 中值滤波（`median_filter` / median filter）: deterministic per-spectrum median filtering.
- 线性基线校正（`linear_baseline` / linear baseline correction）: subtract a per-spectrum endpoint baseline.
- 多项式基线校正（`polynomial_baseline` / polynomial baseline correction）: subtract a per-spectrum polynomial baseline.
- 橡皮筋基线校正（`rubberband_baseline` / rubberband baseline correction）: subtract a per-spectrum rubberband lower-hull baseline.
- 非对称最小二乘基线校正（`als_baseline` / asymmetric least squares baseline correction）: subtract a per-spectrum asymmetric least-squares baseline.
- 均值中心化（`mean_centering` / mean centering）: subtract train-set band means.
- 标准化（`standardization` / standardization）: subtract train-set band means and divide by train-set band standard deviations.
- 最小-最大缩放（`minmax_scaling` / min-max scaling）: fit train-set band min/max and scale all samples.
- 鲁棒缩放（`robust_scaling` / robust scaling）: fit train-set band medians/IQR and scale all samples.
- Pareto 缩放（`pareto_scaling` / Pareto scaling）: fit train-set band means/stds and divide by sqrt(std).
- L2 归一化（`l2_normalization` / L2 normalization）: per-spectrum L2 normalization.
- 面积归一化（`area_normalization` / area normalization）: per-spectrum area normalization.
- 最大绝对值归一化（`max_abs_normalization` / max-absolute normalization）: per-spectrum max-absolute normalization.
- 反射率转吸光度（`reflectance_to_absorbance` / reflectance to absorbance）: convert positive reflectance values with `-log10(x)`.
- 透射率转吸光度（`transmittance_to_absorbance` / transmittance to absorbance）: convert positive transmittance values with `-log10(x)`.
- 对数变换（`log_transform` / log transform）: natural-log transform for strictly positive spectra.
- 保留物理波段范围（`band_range_select` / band range selection）: keep physical band ranges and update `band_axis`.
- 移除物理波段范围（`remove_band_ranges` / remove band ranges）: remove physical band ranges and update `band_axis`.

If methods are missing, ask the user to choose from this list. If SG or
derivative parameters are missing, ask for `window_length` and `polyorder`.
Baseline correction, absorbance conversion, and band-axis-changing methods need
explicit confirmation before execution.

Prefer concise defaults:

- Unknown spectral type: recommend `snv`.
- NIR/FT-NIR: recommend `snv` or `msc`; optionally `sg_smoothing,first_derivative`.
- Raman: recommend baseline correction plus smoothing and normalization.
- FTIR/MIR: recommend baseline correction plus `mean_centering` or `standardization`.
- Reflectance hyperspectral data: consider `reflectance_to_absorbance` or
  `band_range_select` after confirming the physical meaning of X.

Recommended chain order:

1. band range selection/removal
2. absorbance conversion
3. smoothing/denoising
4. baseline correction or detrend
5. scatter correction
6. derivative
7. scaling

Do not implement or fake OSC, EMSC, wavelets, airPLS/arPLS, continuum removal,
band-axis resampling, PCA, SPA, CARS, VIP, UVE, or model-performance-driven
preprocessing search in this version.

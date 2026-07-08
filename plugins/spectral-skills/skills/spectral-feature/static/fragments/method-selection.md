# Method Selection

Use one feature method per execution. When the method is not specified, show a
transparent confirmation card before running anything.

## Confirmation-card structure

1. `推荐方案`: state the budget-limited recommendation and exact parameters.
2. `为什么推荐`: relate the choice to sample count, feature count, task, and split.
3. `本轮默认纳入`: list every included method.
4. `skill 还支持但本轮默认不纳入`: show the complete grouped menu below.
5. `需要额外确认前才能执行`: separate deep training and discovery-only methods.
6. `你可以选择`: give executable reply examples.

Every user-facing entry must use `中文名称（method_code / English name）：说明`.
Never show a code-only list or imply that the recommendation is the full
capability surface.

Every feature menu item must explain four things when shown to the user:
purpose, when it fits the current profile, key parameters, and risk/limitation.
If older text is garbled or too terse, use this canonical explanation style.

## Recommended traditional starting set

- 主成分分析（`pca` / Principal Component Analysis, PCA）：`PCA(10)`。
- PLS 潜变量（`pls_latent_variables` / PLS latent variables）：`PLS-LV(3)`。
- VIP 变量重要性筛选（`vip` / Variable Importance in Projection）：`VIP(100)`。
- KBest 统计筛选（`select_k_best` / SelectKBest）：`KBest(80)`。
- SPA 连续投影算法（`spa` / Successive Projections Algorithm）：`SPA(80)`。

These are budget-limited defaults, not the full capability surface.

## Explanation completeness for supported feature methods

When listing supported feature methods, do not leave methods such as CARS, UVE,
MCUVE, interval PLS, correlation filters, ANOVA/F-regression, variance
threshold, band selection, or deep embeddings as unexplained names. Use concise
entries like these:

- CARS variable selection (`cars` / Competitive Adaptive Reweighted Sampling):
  iterative wavelength selection for chemometric modeling; confirm `top_k` or
  selection threshold, number of runs, PLS components, and random seed; can be
  unstable on small data, so validate with repeated splits.
- UVE (`uve` / Uninformative Variable Elimination): removes variables with weak
  stability or low information relative to noise; confirm PLS components,
  `top_k` or score threshold, and random seed.
- MCUVE (`mcuve` / Monte Carlo UVE): repeated-sampling UVE for more stable
  importance estimates; confirm `n_runs`, PLS components, `top_k`, and seed.
- Interval PLS (`interval_pls` / interval PLS): compares contiguous spectral
  intervals instead of individual wavelengths; useful when chemical information
  is localized; confirm interval width/count and PLS components.
- Correlation filter (`correlation_filter` / correlation filter): keeps bands
  correlated with the target; fast screening but can be redundant and unstable
  under collinearity; confirm `top_k` or correlation threshold.
- ANOVA F (`anova_f` / ANOVA F-test): classification univariate screening;
  useful as a quick high-dimensional filter; confirm `top_k`; ignores
  wavelength interactions.
- F-regression (`f_regression` / F-regression): regression univariate screening;
  confirm `top_k`; not appropriate for classification labels.
- Variance threshold (`variance_threshold` / variance threshold): removes
  near-constant bands; unsupervised and cheap; confirm threshold and remember it
  does not prove predictive relevance.
- Band range selection (`select_by_band_range` / select by band range): keeps a
  physical wavelength/wavenumber range; confirm start/end and units.
- Band index selection (`select_by_band_indices` / select by band indices):
  keeps explicit band columns; confirm zero/one-based indexing and preserve
  band-axis lineage.
- Kernel PCA (`kernel_pca` / Kernel PCA): nonlinear projection; confirm kernel,
  gamma, and components; fit on train only and watch overfit.
- Sparse PCA (`sparse_pca` / Sparse PCA): sparse latent projection; confirm
  components and sparsity/alpha; convergence must be recorded.
- NMF (`nmf` / Non-negative Matrix Factorization): parts-based non-negative
  decomposition; require non-negative input and do not silently shift spectra.
- ICA (`ica_embedding` / ICA): independent component projection; confirm
  components and convergence; components may be unstable across splits.
- LDA projection (`lda_projection` / LDA projection): supervised projection
  limited to `n_classes - 1`; use only inside train-fitted validation design.
- DCT/FFT features (`dct_features`, `fft_features`): deterministic frequency
  compression; confirm number of coefficients and whether high-frequency noise
  should be truncated.
- Dictionary learning (`dictionary_learning` / Dictionary Learning): sparse
  dictionary representation; confirm components, alpha, iterations, and
  convergence.
- t-SNE/UMAP/Isomap/LLE (`tsne_embedding`, `umap_embedding`,
  `isomap_embedding`, `lle_embedding`): discovery/visualization-first manifold
  embeddings; do not treat visual separation as performance evidence.
- Deep train-fitted embeddings (`autoencoder_embedding`,
  `denoising_autoencoder_embedding`, `cnn_1d_embedding`, `resnet1d_embedding`,
  `cls_former_embedding`, `masked_spectral_autoencoder_embedding`,
  `contrastive_spectral_embedding`, `self_supervised_spectral_embedding`):
  learned representations; confirm dimension, epochs, early stopping, batch,
  learning rate, weight decay, seed, device, and downstream model.

For small-sample high-dimensional data, explicitly offer a representative
self-developed/deep feature branch such as `cls_former_embedding_svm` or
`contrastive_spectral_embedding + linear_svm/svm`, but keep `none`, PCA, PLS-LV,
and VIP as stability baselines.

## Full supported menu

### 传统特征与化学计量方法（traditional and chemometric features）

- 无特征工程（`none` / no feature engineering）：保留当前全光谱基线。
- 主成分分析（`pca` / Principal Component Analysis, PCA）：无监督线性降维。
- PLS 潜变量（`pls_latent_variables` / PLS latent variables）：监督式潜变量特征。
- VIP 变量重要性筛选（`vip` / Variable Importance in Projection）：基于 PLS 的波段筛选。
- KBest 统计筛选（`select_k_best` / SelectKBest）：按统计检验选择 top-k 特征。
- SPA 连续投影算法（`spa` / Successive Projections Algorithm）：选择低共线代表波段。
- CARS 变量筛选（`cars` / Competitive Adaptive Reweighted Sampling）：迭代式波段筛选。
- UVE 变量筛选（`uve` / Uninformative Variable Elimination）：剔除无信息变量。
- MCUVE 变量筛选（`mcuve` / Monte Carlo UVE）：重复采样的 UVE。
- 区间 PLS（`interval_pls` / interval PLS）：比较光谱区间的 PLS 表现。
- 相关性筛选（`correlation_filter` / correlation filter）：按相关性保留变量。
- ANOVA F 筛选（`anova_f` / ANOVA F-test）：分类统计筛选。
- 回归 F 筛选（`f_regression` / F-regression）：回归任务专用。
- 方差阈值筛选（`variance_threshold` / variance threshold）：移除低方差变量。
- 波段范围筛选（`select_by_band_range` / select by band range）：按物理范围保留波段。
- 波段索引筛选（`select_by_band_indices` / select by band indices）：按列索引保留波段。

### 投影、信号变换与流形方法（projection, signal-transform, and manifold methods）

- 核 PCA（`kernel_pca` / Kernel PCA）：非线性 PCA。
- 稀疏 PCA（`sparse_pca` / Sparse PCA）：带稀疏约束的 PCA。
- 非负矩阵分解（`nmf` / Non-negative Matrix Factorization, NMF）：要求输入非负。
- 独立成分分析（`ica_embedding` / Independent Component Analysis, ICA）：独立成分投影。
- LDA 监督投影（`lda_projection` / Linear Discriminant Analysis projection）：维度不超过 `n_classes - 1`。
- DCT 特征（`dct_features` / Discrete Cosine Transform features）：确定性频域压缩。
- FFT 特征（`fft_features` / Fast Fourier Transform features）：确定性频域特征。
- 字典学习（`dictionary_learning` / Dictionary Learning）：学习稀疏字典。
- Isomap 嵌入（`isomap_embedding` / Isomap embedding）：流形探索；建模需额外确认。
- LLE 嵌入（`lle_embedding` / Locally Linear Embedding）：流形探索；建模需额外确认。
- t-SNE 可视化嵌入（`tsne_embedding` / t-SNE embedding）：仅用于探索性可视化。
- UMAP 嵌入（`umap_embedding` / UMAP embedding）：默认用于探索；建模需额外确认。

### 深度嵌入方法（deep train-fitted embeddings）

- 自编码器嵌入（`autoencoder_embedding` / Autoencoder embedding）：无监督重构表征；分类建议比较 8/16 维。
- 去噪自编码器嵌入（`denoising_autoencoder_embedding` / Denoising autoencoder embedding）：加噪重构；确认 `noise_std`。
- 一维 CNN 光谱嵌入（`cnn_1d_embedding` / 1D CNN spectral embedding）：学习局部峰形；小样本使用浅层网络和较强正则。
- ResNet1D 光谱嵌入（`resnet1d_embedding` / ResNet1D spectral embedding）：残差卷积表征；样本少于 200 时提示高过拟合风险。
- CLS-former 光谱嵌入（`cls_former_embedding` / CLS-former spectral embedding）：Transformer/CLS token 表征；确认 `patch_size`。
- 掩码光谱自编码器嵌入（`masked_spectral_autoencoder_embedding` / Masked spectral autoencoder embedding）：遮蔽重构；确认 `mask_ratio` 和 `patch_size`。
- 对比光谱嵌入（`contrastive_spectral_embedding` / Contrastive spectral embedding）：对比学习；确认增强强度和 `temperature`。

## Data-aware deep-training card

Do not reuse one fixed default bundle for every deep method. Build the card from
`n_samples`, `n_train`, `n_features`, class count/balance, intended use, and
available device. Show the observed profile first.

For a balanced four-class dataset with `n=120`, `n_train<=72`, and `p>=3401`,
use these recommendations as a starting point, not a promise of optimality:

- classification embedding: `n_components=16`; compare `[8,16,32]` only in an explicitly confirmed deep search;
- visualization embedding: `n_components=2`, clearly marked as visualization-only;
- `batch_size=16` because the training split is small;
- `learning_rate=0.001`, with `0.0003` as a confirmed fallback when validation is unstable;
- autoencoder: `epochs=100`, `weight_decay=1e-5`;
- denoising autoencoder: `epochs=100`, `noise_std=0.03` (reasonable range 0.02-0.05);
- CNN1D: `epochs=80`, `weight_decay=1e-4`, shallow/small-kernel design;
- ResNet1D: `epochs=60`, `weight_decay=1e-4`, plus an explicit high-risk warning;
- CLS-former: `epochs=80`, `patch_size=16`, `weight_decay=1e-4`;
- masked autoencoder: `epochs=100`, `mask_ratio=0.15` (reasonable range 0.10-0.25), `patch_size=16`;
- contrastive embedding: `epochs=100`, `temperature=0.2`, `noise_std=0.03`, `mask_ratio=0.10`.

Treat `epochs` as a maximum training budget. If the implementation supports
validation early stopping, show `patience` and use it; otherwise state that a
fixed epoch count does not prove convergence. Always confirm `n_components`,
`epochs`, `batch_size`, `learning_rate`, `weight_decay`, split, seed, device,
and method-specific parameters before execution.

Deep embeddings are excluded from default optimizer searches. Classification
must not rely on a 2D embedding merely because it was used for visualization.
Select embedding dimensions and downstream classifier parameters using only
validation or train-only CV, then evaluate the locked combination on test.

Use t-SNE/UMAP/Isomap/LLE primarily for report-oriented discovery. Visual
separation is not classification or regression performance evidence.

Before NMF, verify non-negative input and never silently add an offset. For
UVE/MCUVE, confirm `n_components`, `n_runs`, `top_k` or `score_threshold`, and
`random_state` in one card. Route method comparison or tuning to
`spectral-optimizer`.

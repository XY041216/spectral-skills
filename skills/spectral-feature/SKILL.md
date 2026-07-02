---
name: spectral-feature
description: >-
  Use when Codex needs leakage-safe spectral feature engineering, variable
  selection, dimensionality reduction, or report-oriented embeddings from an
  already split standard spectral package or preprocess contract. It supports
  traditional chemometric selectors, projection/signal/manifold methods, and
  confirmed PyTorch deep embeddings. Do not use it to read raw files, run QC or
  splitting, preprocess spectra, train/evaluate models, optimize by
  performance, or treat visual embedding separation as performance evidence.
---

# Spectral Feature

Use this skill only after a standard package and `split_contract.json` exist.
Fit train-learned transforms on train only, then transform validation/test.
For CV/repeated holdout, refit independently per fold/repeat.

## Primary execution

Use `scripts/feature_spectral_package.py` or the corresponding core workflow.
Inputs are a package/preprocess contract, split contract, output directory,
one method, method parameters, task type, and confirmation flags.

Do not compare methods here. Route parameter/method search to
`spectral-optimizer`.

## Selection gate

If method is missing, read `static/fragments/method-selection.md` and render its
full grouped bilingual menu. The card must show recommendation, rationale,
included methods, supported but excluded methods, extra-confirmation methods,
and reply examples.

Every entry must use `中文名称（method_code / English name）：说明`. Never show a
flat code-only list.

Supported groups:

- traditional/chemometric: `none`, `pca`, `pls_latent_variables`, `vip`,
  `select_k_best`, `spa`, `cars`, `uve`, `mcuve`, `interval_pls`,
  `correlation_filter`, `anova_f`, `f_regression`, `variance_threshold`,
  `select_by_band_range`, `select_by_band_indices`;
- projection/signal/manifold: `kernel_pca`, `sparse_pca`, `nmf`,
  `ica_embedding`, `lda_projection`, `dct_features`, `fft_features`,
  `dictionary_learning`, `isomap_embedding`, `lle_embedding`,
  `tsne_embedding`, `umap_embedding`;
- deep embeddings: `autoencoder_embedding`,
  `denoising_autoencoder_embedding`, `cnn_1d_embedding`,
  `resnet1d_embedding`, `cls_former_embedding`,
  `masked_spectral_autoencoder_embedding`,
  `contrastive_spectral_embedding`.

## Parameter confirmation

Confirm method-shaping parameters before writing files. Examples include PCA/
PLS dimensions, VIP/KBest/SPA top-k, iPLS intervals/components/CV, CARS/UVE/
MCUVE runs and selection rule, and manifold dimensions/neighbors.

`--auto-confirm-feature-defaults` may use recorded recommended defaults, but it
is not a substitute for user interaction when the workflow requires explicit
confirmation.

For UVE/MCUVE, ask once for `n_components`, `n_runs`, `top_k` or
`score_threshold`, and `random_state`.

## Deep embedding gate

Require `--confirm-deep-embedding-training` and show actual values before
execution. Confirm:

- intended use: classification embedding or visualization embedding;
- `n_components`, maximum `epochs`, early-stopping status/patience;
- `batch_size`, `learning_rate`, `weight_decay`;
- split, seed, device;
- `noise_std`, `mask_ratio`, `temperature`, or `patch_size` where relevant.

Do not use one common default bundle for every deep method. Generate
recommendations from `n_samples`, `n_train`, `n_features`, class profile,
device, and intended use. For `n<=120`, `n_train<=72`, `p>=3401`:

- classification embedding starts at 16 dimensions; deep search may compare 8/16/32;
- visualization uses 2 dimensions and must be labeled visualization-only;
- batch size starts at 16;
- AE/DAE/masked/contrastive may use about 100 maximum epochs;
- CNN1D starts shallower with about 80 epochs and `weight_decay=1e-4`;
- ResNet1D starts around 60 epochs with a high-overfit warning;
- CLS-former starts around 80 epochs, `patch_size=16`, `weight_decay=1e-4`;
- DAE noise starts at 0.03 (reasonable 0.02-0.05);
- masked ratio starts at 0.15 (reasonable 0.10-0.25);
- contrastive temperature starts at 0.2 (reasonable 0.1-0.2).

These are data-aware starting points, not optimality claims. Fixed epochs do
not prove convergence. Record training trace, seed, device, parameter sources,
and whether early stopping was available.

Fit standardization and neural encoders on train only. PyTorch is required; do
not silently substitute PCA. Deep embeddings remain excluded from default
optimizer spaces unless the user confirms a deep-search budget.

## Path resolution and contracts

Resolve all consumed paths to absolute paths before loading contracts. Explicit
CLI/API paths such as `--split-contract split/split_contract.json` are resolved
against the current working directory. Paths stored inside a contract are
resolved relative to that contract file.

Record `resolved_paths` in `feature_contract.json`, including `input_package`,
`output_package`, `package_dir`, and `split_contract`. Downstream modeling must
use these resolved contracts instead of accidentally nesting a relative split
path under the preprocess directory.

## Discovery and safety semantics

- t-SNE is visualization-only and has no reusable out-of-sample transform.
- UMAP/Isomap/LLE default to discovery; modeling use needs explicit validation design.
- Visual separation is not classification/regression evidence.
- NMF requires non-negative input; never silently offset data.
- LDA projection dimension must not exceed `n_classes - 1`.
- Record convergence status for Sparse PCA, NMF, ICA, and Dictionary Learning.
- Feature engineering must not change sample count, order, or labels.

## Outputs

Write `feature_contract.json`, output package files, method state/artifacts,
`feature_manifest.csv`, and leakage/out-of-sample audits. Deep methods also
write `training_trace.csv`, training audit, transformer artifact when reusable,
and deep-training confirmation.

Use `feature_contract.json` as the downstream handoff to modeling/reporting.

## Block when

- package/split files are missing or inconsistent;
- train is empty or supervised methods lack labels/task type;
- critical parameters are unconfirmed;
- requested deep training is unconfirmed or PyTorch is unavailable;
- NMF input is negative;
- a method would leak validation/test or change sample alignment.

## Read as needed

- `static/fragments/method-selection.md`
- method registry/schema and script help for exact executable parameters

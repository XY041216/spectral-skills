# Feature Scenarios

Common requests and behavior:

- "Do PCA on preprocessed data": require a split contract and PCA retention rule.
- "Try Kernel PCA / Sparse PCA / NMF / ICA / Dictionary Learning": require a
  split contract, confirm `n_components`, fit on train only, and transform
  validation/test with the train-fitted mapping. For NMF, block if the current
  matrix contains negative values and do not shift it automatically. For Sparse
  PCA/NMF/ICA/Dictionary Learning, retain convergence evidence in all audit
  artifacts.
- "Use LDA projection": require classification labels, confirm `n_components`,
  cap it at `n_classes - 1`, and record requested/effective components plus
  `supervised_y_used=true` and `val_test_y_used_for_fit=false`.
- "Keep first 10 PCs": pass `n_components=10`.
- "Keep 95% cumulative variance": pass `explained_variance=0.95`.
- "Use DCT or FFT features": use `dct_features` or `fft_features` with
  `n_components`; these are deterministic per-sample signal transforms, not
  learned feature selection.
- "Select 900 to 1700 nm": use `select_by_band_range` with lower/upper bounds.
- "Select variables 20 to 80": use `select_by_band_indices` with explicit indices.
- "Remove low-variance bands": use `variance_threshold` fitted on train samples.
- "Use VIP": require y and split, then ask for `top_k` or VIP threshold plus
  `n_components`. Do not silently apply VIP >= 1.0.
- "Use SelectKBest": infer `f_classif` for classification and `f_regression`
  for regression.
- "Use iPLS": score contiguous intervals inside train-only CV.
- "Use SPA": select representative, low-collinearity wavelengths.
- "Use CARS/UVE/MCUVE": run train-only repeated selection and write a compact
  trace; do not use validation/test labels.
- "Use UVE" without parameters: ask once for `n_components`, `n_runs`,
  `top_k` or `score_threshold`, and `random_state`; recommend
  `10, 50, top_k=50, 42` for a typical wide-table baseline.
- "Make a UMAP/t-SNE plot": route through `spectral-feature` only to create a
  report-oriented embedding package, then `spectral-report` for the scatter
  figure. Mark the figure as discovery only. For `tsne_embedding`, require an
  all-sample exploratory confirmation because t-SNE has no stable
  train/test-transform handoff.
- "Use Isomap or LLE": allow as discovery embeddings; do not claim visual
  separation is model accuracy. Record out-of-sample support as limited and
  require explicit confirmation before modeling.
- "Use UMAP for modeling": explain that UMAP supports transform when
  `umap-learn` is available, but is visualization-first and excluded from the
  optimizer default space; require explicit confirmation.
- "Use t-SNE for modeling": block. t-SNE is an all-sample discovery embedding
  without a stable standard out-of-sample transform.
- "Use autoencoder / denoising autoencoder / contrastive / masked autoencoder /
  Transformer / CLS-former / CNN / ResNet1D embedding": use the implemented
  deep embedding path only after explicit `--confirm-deep-embedding-training`.
  Confirm `n_components`, `epochs`, `batch_size`, `learning_rate`,
  split design, `random_state`, `device`, and method-specific
  augmentation/masking/patch parameters.
  For contrastive spectral embedding, confirm `noise_std`, `mask_ratio`, and
  `temperature`; if using recommended defaults, state that they are official
  recommended defaults and record parameter sources.
  Require PyTorch, fit the encoder on train samples only, record
  `training_audit`, `training_trace.csv`, `feature_transformer.pkl`, and
  `deep_training_confirmation`, and state that fixed-epoch training does not
  prove convergence.
- "Use attention pooling / generic self-supervised embedding": still gated.
  Ask for a dedicated training protocol, sample-size justification, validation
  plan, and artifact contract; do not silently substitute PCA.
- "Use LASSO or automated feature search": route to a later feature extension
  or optimizer rather than faking it here.
- "Do PCA before splitting": warn that fitting PCA before split leaks validation/test distribution information.

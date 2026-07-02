# Feature Interpretation

Read `references/chart-decision-matrix.md`, `references/axis-and-unit-rules.md`, `references/statistical-reporting.md`, and `references/visual-system.md`.

- PCA score axes must include explained variance when available. If unavailable, use component semantics without inventing percentages.
- PLS score and loading panels must state that the projection is supervised and train-fitted when the contract says so.
- Kernel PCA, Sparse PCA, NMF, ICA, LDA projection, dictionary-learning, DCT,
  and FFT feature figures must use the method names and coordinate labels from
  `feature_contract.json` / `feature_state.json`; do not relabel them as PCA.
- Autoencoder, denoising autoencoder, CNN1D, ResNet1D, CLS-former/Transformer,
  masked autoencoder, and contrastive spectral embeddings must use the method
  names and coordinate labels from `feature_contract.json` /
  `feature_state.json`; do not relabel them as PCA/PLS or imply supervised
  discrimination unless the contract says labels were used.
- Map loadings, VIP, SPA, CARS, UVE, MCUVE, and selected intervals back to physical band values, never only column indices.
- Draw selected bands as translucent spans, rugs, or top ticks over a mean spectrum; avoid thick vertical lines that hide peaks.
- Use a dot/lollipop plot for ranked VIP or score values. Keep the threshold visible only when it was part of the recorded method.
- For interval methods, show continuous intervals and preserve start/end boundaries in source data.
- Treat t-SNE/UMAP/Isomap/LLE figures as discovery evidence; title and caption
  must state that visual separation is not performance evidence. Prefer these
  figures for `figure_role=discovery` or `interpretation`, not `validation`.
- If `feature_mode=visualization_embedding` or `manifold_embedding`, read the embedding coordinates
  from the spectral-feature output package and draw a scatter plot with class,
  split, or batch labels from existing source data. Do not train a new
  embedding inside spectral-report; spectral-report must only visualize
  existing embedding coordinates and their source data.
- do not train a new embedding inside spectral-report.
- Read `intended_use`, `out_of_sample_transform`, and modeling/report handoff
  fields. For t-SNE, state that no stable new-sample transform exists. For UMAP,
  state whether the package is visualization-only or explicitly confirmed for
  modeling. For Isomap/LLE, describe out-of-sample support as limited.
- If the method is NMF, report the recorded non-negative-input check. If the
  method is Sparse PCA, NMF, ICA, or Dictionary Learning, surface convergence
  warnings from `feature_manifest.csv` / `feature_state.json`; do not hide them
  because the plot rendered successfully.
- If `feature_state.json` says `transform_available_for_new_samples=false`
  (for example t-SNE), the caption must state that the embedding is all-sample
  exploratory and not deployable for new-sample inference.
- If `feature_state.json` contains `deep_training_confirmation` or
  `training_audit`, surface the training protocol in the caption or QA:
  train-only fitting, epoch count, final/best loss, random seed, PyTorch device,
  and any small-sample or fixed-epoch warning. Do not describe a deep embedding
  as converged unless the contract explicitly supports that claim.
- Deep embedding score plots are interpretation or discovery evidence unless
  paired with separate modeling metrics from `spectral-modeling`. Visual class
  separation alone is not classifier performance.
- For multi-method 2D embedding scatter plots, use a compact multi-panel layout
  with three columns when there are seven or more methods. For six methods,
  prefer a 2 x 3 layout with six data panels and one shared legend below the
  grid or centered outside the panel area. For seven methods, prefer a 3 x 3 layout:
  seven method panels, one legend panel, and one blank or training/QC
  note panel.
- Multi-method embedding scatter panels must use a white background, no
  gridlines, full black axis frames, lowercase `(a)` to `(g)` panel labels, and
  shared figure-level axis labels such as `Embedding 1` and `Embedding 2`.
  Panel labels must sit outside the plotting region at the upper-left
  (`panel_label_position=outside_upper_left`), not inside the data area. Keep
  legends outside data panels or in an empty panel.
- Keep method titles concise and centered. When the contract method names are
  too long for balanced multi-panel layout, shorten only to unambiguous forms
  and record the mapping in `display_name_map`, for example `Denoising AE`,
  `Masked spectral AE`, or `Contrastive spectral`.
- Use independent axis limits per embedding method unless the contract states
  that coordinates are directly comparable. Captions must state that embedding
  coordinates are method-specific, have no physical units, and axis scales are
  not directly comparable across methods.
- When class count is small (for example four classes) and the user explicitly
  confirms color-only class encoding, a high-contrast palette without marker
  redundancy is acceptable only if CVD/grayscale QA still passes. Otherwise add
  marker shape or direct-label redundancy.
- For deep embedding reports, include a training audit table in the final
  response or report package with method, epochs, batch size, learning rate,
  objective, `supervised_y_used`, train-only status, fixed-epoch status, random
  seed, and device. State fixed-epoch training, no convergence claim, and
  small-sample risk.

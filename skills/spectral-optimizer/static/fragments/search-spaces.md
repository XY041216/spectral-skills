# Search Spaces

Default compact spaces:

- `vip`: `top_k=[10,20,30,50,80,100]`
- `pca`: `n_components=[5,10,20,30]`
- `spa` and `select_k_best`: `top_k=[20,30,50,80]`
- `svm`: `C=[0.1,1,10]`, `gamma=["scale","auto"]`
- `plsr` and `pls_da`: `n_components=[3,5,10]`
- feature comparison recommended small grid:
  - `none`
  - `pca n_components=[5,10,20,30]`
  - `pls_latent_variables n_components=[3,5,10]`
  - `vip top_k=[10,20,30,50,80,100]`
  - `select_k_best top_k=[20,30,50,80]`
  - `spa top_k=[20,30,50,80]`
- classification `optimize_pipeline` compact default:
  - preprocess `none/snv/msc`
  - feature `none/pca10/pls_latent_variables[3,5,10]/vip30`
  - models `svm(C=[1,10], gamma=scale)`, `linear_svm(C=1)`, `pls_da(5)`

Use small spaces first. Broader spaces require budget confirmation. Do not add
experimental models unless the downstream modeling confirmation gate is also
explicitly satisfied.

## Tuning Levels

When the user asks for automatic tuning, show a layered confirmation card
before creating a candidate space:

- Level 1: classifier parameter tuning only. Tune locked downstream model
  parameters such as SVM `C/gamma`, KNN `k`, RF depth/trees, or PLS-DA
  components inside the confirmed train/validation or train-only CV design.
- Level 2: traditional feature plus classifier tuning. Compare bounded
  feature candidates such as `PCA(10)`, `PLS-LV(3)`, `VIP(100)`,
  `KBest(80)`, and `SPA(80)` with classifier parameters. List supported but
  excluded feature methods so the user sees that omissions are budget choices.
- Level 3: deep embedding plus classifier tuning. Compare confirmed deep
  embedding dimensions, for example `n_components=[8,16,32]`, then tune the
  downstream classifier within the same leakage-safe validation design.

Level 3 requires explicit extra budget confirmation for embedding dimensions,
epochs, batch size, learning rate, device, seed, repeats, and the selection
metric. No level may use final-test metrics for candidate selection.

## Budget Profiles

- `quick`: narrow traditional search for smoke/interactive use.
- `regular`: recommended bounded traditional search. For feature comparison it
  expands PCA/PLS-LV/VIP/KBest/SPA grids instead of testing one arbitrary point.
- `extended`: broader traditional feature/model grids; follow close candidates
  with repeated validation and report Macro-F1 mean ± SD.
- `deep`: explicit Level 3 protocol. Start with deep embedding dimensions
  `[8,16,32]`, method-specific epoch/patch/noise/mask/temperature grids, and
  locked downstream `[linear_svm, svm, lda]`. The raw default can exceed 300
  trials, so preview and prune it before confirmation.

`recommended` remains a compatibility alias for `regular`. Deep search must not
run without `--confirm-comparison-design`, `--confirm-parameter-grid`, and an
explicitly confirmed expanded trial budget. Test data remains excluded.

Do not include discovery-only feature embeddings in default optimizer search
spaces. `tsne_embedding`, `umap_embedding`, `isomap_embedding`, and
`lle_embedding` are primarily for `spectral-report` visualization and must not
be selected by validation/test model performance unless the user explicitly
confirms a modeling-oriented embedding experiment with a leakage-safe
validation plan. Gated deep/self-supervised feature methods such as
`contrastive_spectral_embedding`, `masked_spectral_autoencoder_embedding`,
`autoencoder_embedding`, `transformer_embedding`, `cls_former_embedding`,
`cnn_1d_embedding`, `resnet1d_embedding`, and
`self_supervised_spectral_embedding` require a separate protocol and should be
recorded as excluded from compact/recommended optimizer spaces.

For low-performing deep embeddings that were trained with `n_components=2`,
do not assume classifier hyperparameters are the main problem. Treat 2D deep
embeddings primarily as visualization embeddings that may discard classification
information. Before tuning only the classifier, recommend a separate confirmed
deep-embedding protocol that compares higher modeling embedding dimensions,
for example `n_components=[8,16,32]`, then tunes the downstream classifier only
inside the training/validation design using the confirmed selection metric
such as Macro-F1. This protocol is not part of the compact optimizer default
and requires explicit budget, epochs, device, seed, and leakage-safe validation
confirmation.

If a user asks whether automatic tuning is available after seeing poor
2D CLS-former or other deep-embedding classification results, answer that
spectral-optimizer supports bounded auditable tuning without test leakage, but
recommend checking the embedding dimension/protocol before classifier-only
tuning. Do not use previously inspected test metrics to choose the dimension.

When using a compact default `optimize_pipeline` space, write
`candidate_space_policy` with included and excluded methods. For example, if
`select_k_best`, `spa`, `logistic_regression`, or `random_forest_classifier`
are not included, record that they were omitted to stay within the compact
budget and are available in an extended/custom search. `pls_latent_variables`
is core enough for spectral work that it should be included in the classification
compact default unless a custom budget card explicitly excludes it.

For `compare_step`, include the relevant "none" baseline unless the user
explicitly asks to compare only non-empty transformations.

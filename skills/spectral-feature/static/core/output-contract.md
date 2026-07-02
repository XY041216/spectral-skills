# Output Contract

Write a new standard spectral package with transformed `X.csv` and unchanged
sample alignment.

Required output:

- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- `data_contract.json`
- `feature_state.json`
- `feature_contract.json`
- `feature_manifest.csv` when convergence or fold/repeat audit rows are needed

Write `y.csv` and `metadata.csv` only when present in the input package.

For PCA, `band_axis.csv` should list `PC1`, `PC2`, etc. For band selection,
retain the selected original band-axis rows.

Write compact method artifacts as CSV when relevant. Keep only artifact paths
and method summaries in `feature_state.json` or `feature_contract.json`; do not
embed full score vectors in JSON.

`data_contract.json` must include `processing_stage: feature`,
`parent_contract`, `split_contract`, and compact `feature_summary`. Do not write
reports, logs, debug folders, or copied train/test matrices by default.

For projection and embedding extensions, record `feature_mode`, `intended_use`,
`out_of_sample_transform`, `allowed_for_optimizer_default`, and downstream
handoff status. `spectral_modeling.ready` must be false for t-SNE, and gated
methods must say that explicit confirmation is required. Keep
`spectral_report.ready=true` for valid discovery embeddings.

For confirmed deep embeddings, also record `training_audit`,
`deep_training_confirmation`, `artifacts.training_trace.csv`, the PyTorch
device, fixed `random_seed`, and `transform_available_for_new_samples=true`.
Write `feature_transformer.pkl` so downstream modeling or deployment can apply
the train-fitted encoder. Mark `allowed_for_optimizer_default=false` and
include a warning that fixed-epoch training is not a convergence guarantee.

Repeat method-specific audit fields consistently in `feature_state.json`,
`feature_contract.json`, and `feature_manifest.csv`:

- convergence-audited methods: `converged`, `n_iter`, `max_iter`,
  `random_seed`, and warning text;
- deep embeddings: training status, epoch count, final loss, best loss, random
  seed, and warnings.

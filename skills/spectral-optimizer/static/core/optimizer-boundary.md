# Optimizer Boundary

`spectral-optimizer` plans candidate spaces and selects from validation/CV
trial results. It must not reimplement child skill algorithms.

Allowed:

- recommend candidate preprocessing, feature, and modeling methods
- build `candidate_space.json` only after the user confirms the full card
- build `trial_manifest.csv` only after design and budget confirmation
- select a best trial from `val_*` or `cv_*` metrics
- write `best_pipeline.json`
- preview a comparison with `--preview-only`, which must write no files

Forbidden:

- read raw spectral files
- create or modify splits
- fit preprocessing, feature, or model algorithms directly
- select by test metrics
- run unbounded AutoML
- create any optimizer files or child-skill outputs before user confirmation
- after confirmation, let the Agent directly run child skills to prepare
  comparison intermediates such as fixed SNV preprocessing, PCA feature
  packages, or trial model packages
- let the Agent hand-write `candidate_space.json`, `trial_manifest.csv`,
  `optimization_plan.json`, `optimizer_contract.json`, or `trial_inputs.json`
- materialize fixed upstream preprocessing or feature outputs before the
  comparison/optimization card is confirmed

After the user confirms a comparison or optimization card, the Agent should
still call only the top-level optimizer/workflow entry. The optimizer executor
must call spectral-preprocess, spectral-feature, and spectral-modeling
validation-only internally so lineage stays inside optimizer artifacts.

---
name: spectral-optimizer
description: >-
  Use when Codex needs to recommend, tune, compare, or plan leakage-safe
  spectral preprocessing, feature, and modeling candidates after reader/QC and
  splitting contracts exist. This skill supports recommend_from_profile,
  tune_method, compare_step, and budgeted optimize_pipeline modes. It creates
  optimizer_contract.json, optimization_plan.json, candidate_space.json,
  trial_manifest.csv, optional trial_results.csv, best_pipeline.json, and
  recommendation_report.md. Candidate scoring should call official child
  skills, including spectral-modeling validation_only mode. Do not use this
  skill to read raw files, run QC, split data, implement preprocessing,
  feature, or model algorithms, bypass child skills, use test metrics for
  selection, or launch unbounded AutoML searches.
---

# Spectral Optimizer

`spectral-optimizer` chooses what to try next. It does not implement spectral
preprocessing, feature engineering, or model training. It builds candidate
spaces, writes auditable trial plans, and selects a best pipeline only from
validation or cross-validation trial results.

Do not use this skill to read raw files, run QC, split data, or replace child
skills.

## Modes

- `recommend_from_profile`: inspect data profile only; do not run models.
- `tune_method`: fix the workflow and vary parameters for one method, such as
  `feature=vip top_k=[10,20,30,50,80,100]`.
- `compare_step`: fix the workflow and compare one stage, such as feature
  methods `pca`, `pls_latent_variables`, `vip`, `select_k_best`, `spa`.
- `optimize_pipeline`: search multiple stages under a confirmed trial budget.

User intent routing is strict:

- `自动选优组合`, `自动组合选优`, `组合选优`, `pipeline optimization`, or "best
  preprocessing + feature + model combination" means `optimize_pipeline`, not
  `compare_step`.
- For this intent, the default recommendation is the regular 72-trial pipeline
  preview: preprocess `none/snv/msc`; feature
  `none`, `pca(n_components=10)`, `pls_latent_variables(n_components=3,5,10)`,
  `vip(top_k=30)`; models `svm(C=1,10,gamma=scale)`, `linear_svm(C=1)`,
  `pls_da(n_components=5)`; selection metric `val_macro_f1`.
- Do not answer an automatic combination request with only "SNV + feature
  comparison + fixed SVM" or a 22-trial feature comparison. That is a
  `compare_step` route and must be used only when the user explicitly asks
  "which feature method is best" or fixes preprocess/model stages.

Use the deterministic entry:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode recommend_from_profile --task-type classification --n-samples 120 --n-features 3401 --n-classes 4 --output-dir <optimizer-output> --json
```

For parameter tuning:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode tune_method --target-step feature --method vip --task-type classification --n-samples 120 --n-features 3401 --output-dir <optimizer-output> --json
```

Before any compare/tune/optimize design is confirmed, use no-file preview:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode compare_step --target-step feature --fixed-preprocess-methods snv --comparison-depth regular --task-type classification --n-samples 120 --n-features 3401 --max-trials 22 --preview-only --json
```

`--preview-only` must write only stdout JSON. It must not create `output_dir`,
`candidate_space.json`, `trial_manifest.csv`, `optimization_plan.json`,
`optimizer_contract.json`, trial inputs, or any preprocess/feature/modeling
outputs.

When the user provides candidate method names or locked validator/model
parameters in conversation, pass them as CLI lists rather than hand-writing
optimizer artifacts:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode compare_step --target-step preprocess --preprocess-candidates none,snv,msc,detrend,snv_detrend,sg_smoothing,first_derivative --validator-model svm --validator-param C=1.0 --validator-param gamma=scale --task-type classification --n-samples 120 --n-features 3401 --max-trials 7 --preview-only --json
```

Use `--model-param-grid model.param=value` for named model grids, for example
`--model-param-grid svm.C=1|10 --model-param-grid svm.gamma=scale`. Do not
create `compare_*_input.json`, `model_config.json`, or candidate-space files
just to express model parameters.

After confirmation, the optimizer may execute validation-only scoring through
official child skills. Do not prepare SNV, PCA, VIP, or other intermediate
packages outside the optimizer; pass the original reader package and split
contract and let the optimizer executor prepare every trial input:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode compare_step --target-step feature --fixed-preprocess-methods snv --comparison-depth regular --execute-trials --package-dir <reader-package> --split-contract <split_contract.json> --output-dir <optimizer-output> --task-type classification --selection-metric val_macro_f1 --max-trials 22 --confirm-comparison-design --confirm-parameter-grid --confirm-budget --validator-model-source user_confirmed_recommendation --json
```

Validation trials must call `spectral-modeling` in `validation_only` mode with
internal model selection disabled. The model parameters recorded for each
optimizer trial are the parameters that must be scored; if a trial lacks the
parameters needed to reproduce a model, block or fail that trial rather than
letting modeling retune it.

For model comparison after a feature stage is fixed, pass the fixed
`feature_contract.json` to the optimizer; do not create
`model_compare_candidate_space.json` by hand:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode compare_step --target-step modeling --fixed-feature-contract <feature_output/feature_contract.json> --model-candidates logistic_regression,linear_svm,svm,lda --execute-trials --package-dir <reader-package> --split-contract <split_contract.json> --output-dir <optimizer-output> --task-type classification --selection-metric val_macro_f1 --max-trials 4 --confirm-comparison-design --confirm-budget --json
```

Always preview model comparison first. Candidate-list CLI input automatically
adds the deterministic validation-only parameter card: logistic/linear SVM
`C=1`, kernel SVM `C=1 gamma=scale`, LDA `solver=svd`, QDA
`reg_param=0.1`, Gaussian NB `var_smoothing=1e-9`, KNN `n_neighbors=5`,
and random forest `n_estimators=100 max_depth=5`. Show these values as pending
user confirmation. If a custom candidate space omits any required locked
parameter, preview must return `missing_locked_params` and
`recommended_locked_params`, write no files, and refuse plan materialization
or execution until corrected. Never launch trials merely to discover this
configuration error.

If the candidate space expands method parameters, such as PCA components,
PLS latent variables, VIP `top_k`, KBest `top_k`, or SPA `top_k`, add
`--confirm-parameter-grid` only after the user has confirmed the exact grid.

For ordinary holdout runs, prefer letting the optimizer prepare trial inputs
inside `output_dir`:

```bash
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode optimize_pipeline --candidate-space <candidate_space.json> --execute-trials --package-dir <reader-package> --split-contract <split_contract.json> --output-dir <optimizer-output> --task-type classification --max-trials 20 --confirm-candidate-space --confirm-budget --json
```

For default feature comparison, `--comparison-depth regular` expands the
small grid `none`, `PCA=[5,10,20,30]`, `PLS-LV=[3,5,10]`,
`VIP top_k=[10,20,30,50,80,100]`, and `KBest/SPA top_k=[20,30,50,80]`
for 22 trials under one validator model. Use `--comparison-depth quick` only
when the user confirms a single-point screen; use `extended` for a broader
grid or repeated-holdout follow-up. `recommended` remains an alias for
`regular`. Use `deep` only after explicit Level 3 confirmation; its raw
embedding/classifier grid may exceed 300 trials and must be previewed and
pruned before execution.

For any optimization comparison, do not describe the default as "conventional
optimization comparison." Describe it as the recommended regular comparison,
then explicitly list excluded but supported built-in self-developed
small-sample/deep candidates: DKL-GP, prototype spectral models, CLS-former
models, CLS-former embedding + SVM, and feature-stage deep embeddings with
confirmed downstream classifiers.

If the user asks for all supported methods, full search, 全量支持方法选优, or
exhaustive comparison, do not collapse the request to the regular compact space.
Use an `all-supported-preview` route: enumerate all implemented preprocess,
feature, model, optional boosting, self-developed, and deep candidate groups,
estimate expanded trials, mark unavailable optional dependencies and
visualization-only methods, then ask the user to prune or confirm a high-budget
run before execution.

Every optimizer recommendation must compute and show data-profile flags:
`small_sample`, `high_dimensional`, `very_high_dimensional`, and
`class_imbalance`. For small-sample high-dimensional profiles such as
`n=120, p=3401`, recommend regular first as the stability baseline and offer
representative self-developed/deep add-ons such as `cls_former_embedding_svm`
or `contrastive_spectral_embedding + linear_svm/svm` before a full deep or
all-supported search.

The confirmation card is incomplete unless it contains a section named
`内置自创/深度候选是否加入选优组合` and explicitly asks:
`我们推荐先跑 regular 组合；同时 skill 内置自创小样本特征提取/表示学习和深度学习方法，是否加入到本轮组合选优？`
List candidate codes item by item, including `contrastive_spectral_embedding`,
`masked_spectral_autoencoder_embedding`, `self_supervised_spectral_embedding`,
`autoencoder_embedding`, `transformer_embedding`, `cls_former_embedding`,
`cnn_1d_embedding`, `resnet1d_embedding`,
`spectral_dkl_gp_classifier/regressor`, `proto_spectral_classifier/regressor`,
`cls_former_classifier/regressor`, and `cls_former_embedding_svm`.

Do not present these as a code-only list. Group and explain them:

- self-developed small-sample models: CLS-former classifier/regressor
  (`cls_former_classifier/regressor`), prototype spectral
  classifier/regressor (`proto_spectral_classifier/regressor`), and DKL-GP
  (`spectral_dkl_gp_classifier/regressor`);
- self-developed/deep feature extraction or representation learning:
  `cls_former_embedding`, `contrastive_spectral_embedding`,
  `masked_spectral_autoencoder_embedding`, `self_supervised_spectral_embedding`,
  `autoencoder_embedding`, `transformer_embedding`, `cnn_1d_embedding`, and
  `resnet1d_embedding`;
- bridge option: CLS-former embedding + SVM (`cls_former_embedding_svm`).

For each group, state the intended use and risk in plain Chinese: small-sample
or nonlinear spectral modeling, uncertainty-aware modeling, train-fitted deep
representations, extra epochs/device/early-stopping confirmation, and higher
overfitting risk on small high-dimensional datasets. For profiles like
`n=120, p=3401`, recommend regular first as the stability baseline and offer to
add only 1-2 representative self-developed/deep candidates unless the user asks
for a full deep search.

Second-stage candidate selection is mandatory. If the user replies only
`regular + 加 1-2 个深度特征候选`, `regular + 加 1-2 个自创/深度模型候选`, `B`, or
`C`, do not choose candidates for the user and do not execute trials. Return a
candidate-selection card with checkboxes/options and wait for exact method
confirmation.

For deep feature add-ons, offer at least:

- `cls_former_embedding_svm`: CLS-former embedding + SVM, recommended
  representative self-developed bridge for small high-dimensional spectra;
- `contrastive_spectral_embedding + linear_svm/svm`: contrastive spectral
  embedding plus confirmed downstream classifier;
- `masked_spectral_autoencoder_embedding + linear_svm/svm`;
- `autoencoder_embedding + linear_svm/svm`;
- custom: user-specified deep feature plus downstream classifier.

For self-developed/deep model add-ons, offer at least:

- `cls_former_classifier`;
- `proto_spectral_classifier`;
- `spectral_dkl_gp_classifier`;
- `cls_former_embedding_svm`;
- custom: user-specified model list.

When a user confirms any add-on for `optimize_pipeline`, insert it into the
same three-axis candidate space before previewing or executing. Never run
`regular` and the self-developed/deep candidates as separate optimizer jobs.
Deep feature add-ons belong on the `feature` axis and must be crossed with
every confirmed preprocessing method and downstream model. Self-developed/deep
model add-ons belong on the `modeling` axis and must be crossed with every
confirmed preprocessing and feature method. Example: regular classification
space is `3 preprocess * 6 expanded feature choices * 4 expanded model choices
= 72` trials. Adding `cls_former_classifier`, `proto_spectral_classifier`, and
`spectral_dkl_gp_classifier` changes the unified space to `3 * 6 * 7 = 126`
trials, not `72 + 3`. Adding `cls_former_embedding` as a feature add-on changes
the unified space to `3 * 7 * 4 = 84` trials unless the user also prunes or
changes the downstream model axis. The output must be one optimizer directory
with one `candidate_space.json`, one `trial_manifest.csv`, one
`trial_results.csv`, and one `best_pipeline.json`.

The card must also offer `全量运行所有支持方法（传统 + 自创/深度）` as a separate
high-budget option. This is different from `全量支持方法选优预览`: preview shows
the universe and trial count, while full run requires explicit high-budget
confirmation, dependency availability, pruning policy, metric, split/repeats,
device, and deep-training parameters.

Dependency rule: do not install PyTorch, XGBoost, LightGBM, CatBoost, UMAP, or
any optional dependency merely because the user selected a deep/full option.
If dependencies are missing or fail to import, stop at `needs_confirmation` or
`blocked`, list unavailable candidates, and ask whether to install dependencies,
skip unavailable methods, or run only executable traditional candidates. Never
write deep configs or launch regular trials as a substitute for the missing
candidate confirmation.

End with choices: A `仅 regular 推荐组合`, B `regular + 选择的自创/深度特征`,
C `regular + 选择的自创/深度模型`, D `先预览 extended/deep 方案再确认`. Do not ask
only for `确认 regular 72`; that is an incomplete confirmation and must remain
`needs_confirmation`.

## Leakage Rule

Never use test metrics to choose a method, parameter, or pipeline. Selection
must use validation metrics or inner CV metrics. If final test evaluation is
needed, run it once after the best pipeline is fixed.

Default selection metrics:

- classification: primary `val_macro_f1`, then `val_accuracy`,
  `val_balanced_accuracy`, and AUC when available.
- regression: primary `val_rmse`, then `val_mae` and `val_r2` when available.

Every optimizer contract must record:

- `selection_metric`
- `selection_protocol`
- `test_used_for_selection: false`
- `final_test_evaluated_once: true`

If a supplied `trial_results.csv` contains test metrics, do not select by them.
Use only `val_*` or `cv_*` metrics.

If `test_access_log.json` shows test metrics were already viewed before an
optimization request, record that downstream final test metrics are
confirmatory, not fully blind.

When the user later asks to evaluate the optimizer best pipeline on test, route
to `spectral-modeling` with `--best-pipeline <best_pipeline.json>` and
`--lock-best-pipeline-params`. Do not ask modeling to run its normal internal
hyperparameter search for an optimizer-best reproduction. If the test split was
already accessed, require a separate confirmatory-test confirmation before
running final evaluation.

`best_pipeline.json` must preserve full trial lineage, not only the model name
and parameters. Include enough information for modeling to recover the locked
upstream inputs, preferably `trial_id`, `trial_dir`, `modeling_output`,
`preprocess_method`, `feature_method`, and the resolved feature/preprocess
contract path produced during the winning trial. If the selected pipeline has a
non-none feature stage, final replay must consume that `feature_contract.json`;
if it has only preprocessing, final replay must consume that
`preprocess_contract.json`. A best-pipeline replay that silently drops the
feature stage is invalid.

When validation/CV metrics tie, use a deterministic tie-breaker and record it
in `best_pipeline.json`, `optimizer_contract.json`, and
`recommendation_report.md`:

1. prefer the best primary validation/CV metric;
2. if tied, prefer better secondary validation/CV metrics;
3. then fewer preprocessing methods;
4. then predefined preprocessing priority; when SNV and MSC tie, prefer SNV
   because it is per-sample and does not require a train-set reference
   spectrum;
5. then fewer output features/components;
6. then fewer tuned parameters;
7. then unsupervised feature methods over supervised selectors;
8. then lower-compute traditional models over experimental models.

For small datasets with a single holdout validation split, record a structured
follow-up `repeated_holdout` check with `n_repeats=5` or `10`. Keep the selected
preprocess, feature, model, and parameters locked, then report mean/std
Macro-F1 for classification or RMSE for regression. Elevate this recommendation
when train accuracy is near 1.0.

## Output Boundary

Write optimizer outputs only, and place them under the current workflow run
directory when the optimizer is part of a broader analysis. Use
`<run_dir>/optimizer_output` for optimizer artifacts; per-trial intermediates
must remain below `<run_dir>/optimizer_output/trials/<trial_id>/...`. Do not
create sibling folders beside the raw file or reader package such as
`<stem>_optimizer_regular72`, `<stem>_opt_experimental3`, or separate deep
model output directories for one comparison.

The optimizer output directory may contain only:

- `optimizer_contract.json`
- `optimization_plan.json`
- `candidate_space.json`
- `trial_manifest.csv`
- `trial_results.csv` only when real validation/CV trial scores are supplied or
  produced by `--execute-trials`
- `best_pipeline.json`
- `recommendation_report.md`

Do not create transformed matrices, cleaned packages, split contracts, or
report figures. Per-trial modeling outputs are allowed only under
optimizer-managed `trials/<trial_id>/model_output` when `--execute-trials`
uses `evaluation_mode=validation_only`; those trial outputs must not contain
test metrics or final model artifacts.

Hard red line before user confirmation:

- Do not create or modify any file or directory.
- Do not manually create `candidate_space.json`, `trial_manifest.csv`,
  `optimization_plan.json`, `optimizer_contract.json`, or `trial_inputs.json`.
- Do not run upstream materialization such as SNV preprocessing merely because
  the user named a fixed upstream step.
- Only show the comparison card in conversation, or call optimizer with
  `--preview-only` and verify `files_written=[]` and `directories_created=[]`.

After the user confirms the full card, the optimizer state may advance:

`needs_confirmation -> design_confirmed -> plan_materialized -> trials_completed -> best_selected -> final_test_evaluated`.

Only `plan_materialized` and later states may write optimizer artifacts. Only
`trials_completed` and later states may write trial results. Final test still
requires a separate user confirmation.

Hard red line after user confirmation:

- The Agent still must not directly run child skills to create comparison
  intermediates such as `preprocess_output_snv`, `feature_output_pca`, or a
  hand-written `trial_inputs.json`.
- After confirmation, call only the top-level workflow or optimizer command.
  The optimizer executor must prepare preprocess, feature, and modeling
  validation-only trial outputs and record full lineage in
  `optimizer_contract.json`, `trial_manifest.csv`, and `best_pipeline.json`.
- If the optimizer cannot prepare a required fixed upstream stage itself, stop
  with `blocked`; do not silently fall back to Agent-side subskill orchestration.

## Confirmation Gates

Return `needs_confirmation` when:

- `--execute-trials` is requested for `compare_step` or `tune_method` without a
  user-confirmed comparison card. The card must list candidate methods, fixed
  upstream/downstream stages, validator model, primary metric, auxiliary
  metrics, and max_trials. A validator such as SVM is allowed only after user
  confirmation because it directly affects stage rankings.
- `--execute-trials` is requested for `compare_step` or `tune_method` and any
  method has a parameter grid that was not confirmed. A method list such as
  `none / PCA / PLS latent variables / VIP / select_k_best / SPA` is not enough;
  the confirmation card must also show grids like
  `PCA n_components=[5,10,20,30]`, `PLS n_components=[3,5,10]`,
  `VIP top_k=[10,20,30,50,80,100]`, and `KBest/SPA top_k=[20,30,50,80]`.
  Only then may the runner pass `--confirm-parameter-grid`.
- `--execute-trials` is requested for `optimize_pipeline` without a confirmed
  candidate space policy and confirmed trial budget. Do not treat
  `--confirm-budget` as a substitute for user confirmation of included methods.
- `optimize_pipeline` or a custom candidate space exceeds `--max-trials`; this
  check must happen immediately after candidate expansion and before any child
  skill runs.
- a workflow comparison would need preprocessing method parameters that were
  not supplied; optimizer compare spaces may include explicit recommended
  values such as `sg_smoothing window_length=11 polyorder=2`, but must record
  them as optimizer-confirmed defaults.
- the Agent has not confirmed a comparison card listing candidate methods,
  fixed stages, primary metric, auxiliary metrics, and max_trials.
- model comparison preview finds a candidate without every parameter required
  by validation-only locked execution. Return all missing fields at once plus
  executable recommended values; keep `files_written=[]` and
  `directories_created=[]`.
- experimental models appear in a candidate space without explicit downstream
  model parameter confirmation.
- the user asks to select by test score.

If a compare step inherits a validator model from a previous confirmed
comparison, record it in `optimizer_contract.json` and
`optimization_plan.json` as `validator_model.source =
inherited_from_previous_user_confirmed_comparison` plus
`previous_confirmation_stage`. When in doubt, ask the user whether to reuse
the validator.

After a stage comparison, the selected method is a recommendation only. Ask the
user to adopt it before fixing it for later stages or before final test
evaluation. Do not run final test automatically from an optimizer
recommendation.

Return `blocked` when:

- mode is unsupported.
- target step is unsupported.
- candidate space has no modeling candidates.
- supplied trial results have no validation/CV selection metric.

Return `trials_failed` when `--execute-trials` writes `trial_results.csv` but
all trials fail. Include `n_trials`, `n_success`, `n_failed`, `failure_reason`,
and the `trial_results.csv` path in `optimizer_contract.json`; do not report a
normal `plan_materialized` or `ready` status.

## Read As Needed

- Use `static/core/optimizer-boundary.md` for neighboring-skill boundaries.
- Use `static/core/leakage-rules.md` for test isolation.
- Use `static/core/output-contract.md` for output files.
- Use `static/fragments/mode-selection.md` for routing user intents.
- Use `static/fragments/search-spaces.md` for compact candidate spaces.
- Use `references/optimizer-scenarios.md` for example user requests.

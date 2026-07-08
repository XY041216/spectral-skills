# Mode Selection

- User asks "recommend methods" or gives only data size/profile:
  `recommend_from_profile`.
- User names one method but asks for best parameter:
  `tune_method`.
- User asks which feature/modeling/preprocess step is best:
  `compare_step`.
- User asks for fully automatic best pipeline:
  `optimize_pipeline` with a confirmed trial budget.
- User asks "自动选优组合", "自动组合选优", "组合选优", or asks for the best
  preprocessing + feature + model combination:
  `optimize_pipeline` with the regular 72-trial pipeline preview by default.
  Do not downgrade this intent to fixed-SNV feature comparison or a 22-trial
  `compare_step` unless the user explicitly asks to compare only features or
  explicitly fixes upstream/downstream stages.
- User asks to run all supported methods, full search, 全量支持方法选优, or
  exhaustive comparison: `optimize_pipeline` in `all-supported-preview` policy
  first. Preview the complete implemented candidate universe, report expanded
  trial count, then require pruning or a high-budget confirmation before
  execution.

Before executing `compare_step`, ask the user to confirm the full comparison
card: candidates, fixed stages, validator model, primary metric, auxiliary
metrics, and max_trials. The validator model is not neutral; for example,
preprocess comparison under SVM means "best under the confirmed SVM validator",
not "globally best for every model".

For modeling comparison, preview must also prove that every validation-only
trial has a complete locked parameter set. Candidate-list input should receive
the optimizer's deterministic fixed defaults automatically. A custom space
with missing fields must return every missing field and recommended value in
`needs_confirmation`; do not materialize or execute the plan.

Before that confirmation, do not write files. Do not create a fixed SNV package,
candidate-space file, optimizer output directory, or trial manifest. If a
machine-readable card is useful, call optimizer with `--preview-only`; the
result must report `files_written=[]` and `directories_created=[]`.

If any candidate method expands parameters, confirm the parameter grid in the
same card before execution. Confirming method names alone is not enough:
`PCA / PLS / VIP / KBest / SPA` must also show the exact component or `top_k`
grid and the expanded trial count. Use `--confirm-parameter-grid` only after
that confirmation.

For optimizer execution, offer four explicit budget profiles:

- `quick`: smoke/interactive search; one point per traditional feature method.
- `regular`: recommended bounded traditional grid; backward-compatible alias
  `recommended` is accepted. Treat it as the default recommendation, not as the
  full optimizer capability surface.
- `extended`: broader traditional grids plus repeated-holdout follow-up.
- `deep`: opt-in deep embedding plus classifier search. It requires explicit
  device, dimension, epoch, augmentation, seed, metric, and expanded-trial
  confirmation and is never selected automatically.
- `all-supported-preview`: full supported-method universe preview across all
  implemented preprocess, feature, classifier/regressor, optional boosting,
  self-developed, and deep methods. It is a route recommendation, not automatic
  execution; after preview, ask the user to prune by method groups, cap trials,
  or confirm a high-budget run.

If the user fixes an upstream preprocess step such as SNV for feature
comparison, pass it to optimizer as `--fixed-preprocess-methods snv`. Do not
run `spectral-preprocess` yourself. The optimizer executor must prepare SNV
inside each trial after confirmation.

If a later comparison reuses the validator model from an earlier comparison,
either ask whether to reuse it or record
`validator_model.source=inherited_from_previous_user_confirmed_comparison` and
the previous confirmation stage in the optimizer contract.

Before executing `optimize_pipeline`, ask the user to confirm the candidate
space policy and the trial budget. Do not pass `--confirm-budget` unless the
user explicitly confirmed the expanded trial count.

For automatic compact search, show included and excluded methods before
execution. Say when `pls_latent_variables`, SPA/CARS/UVE, optional boosting, or
experimental small-sample/deep models are excluded from the compact default.
List the built-in self-developed candidates (`spectral_dkl_gp_*`,
`proto_spectral_*`, `cls_former_*`, `cls_former_embedding_svm`) and relevant
deep embedding options from `spectral-feature`, then ask whether the user wants
to keep the regular comparison only or include selected candidates through an
extended/custom/deep search.

For the regular automatic-combination default, show exactly this bounded
pipeline space unless the user changes it:

- split: stratified 6:2:2, seed 42, test excluded from selection;
- preprocess: `none`, `snv`, `msc`;
- feature: `none`, `pca(n_components=10)`,
  `pls_latent_variables(n_components=3,5,10)`, `vip(top_k=30)`;
- model: `svm(C=1,10,gamma=scale)`, `linear_svm(C=1)`,
  `pls_da(n_components=5)`;
- trial count: 72 validation trials;
- metric: `val_macro_f1`.

Never call this "fixed SNV + feature comparison"; never report 22 trials for
this intent.

For "组合选优", "优化比较", `optimize_pipeline`, or `compare_step`, a regular
preview is incomplete unless the user-facing confirmation card includes
`内置自创/深度候选是否加入选优组合`. It must recommend the bounded regular plan
and then ask whether to add built-in self-developed small-sample/deep feature
or model candidates. End with choices A `仅 regular 推荐组合`, B
`regular + 选择的自创/深度特征`, C `regular + 选择的自创/深度模型`, D
`先预览 extended/deep 方案再确认`. Do not ask only for `确认 regular 72`.

Also offer choice E `全量支持方法选优预览`: enumerate every supported method group,
estimate the trial count, and ask for pruning before execution. Do not silently
exclude methods from this route; exclusions must be labeled as unavailable in
the runtime environment, discovery-only, or blocked by missing confirmation.
Also offer choice F `全量运行所有支持方法（传统 + 自创/深度）`: this is a high-budget
execution request, not a default. It requires a second confirmation card with
dependency status, unavailable optional methods, exact grids, split/repeats,
metric, device, and deep-training parameters before any files or trials are
created.

If the user chooses B or C, show a second-stage candidate-selection card and
wait. Do not select the 1-2 candidates yourself. Do not execute regular trials
while the add-on choice is still unresolved.

For B, list selectable deep/self-developed feature branches:
`cls_former_embedding_svm`, `contrastive_spectral_embedding + linear_svm/svm`,
`masked_spectral_autoencoder_embedding + linear_svm/svm`,
`autoencoder_embedding + linear_svm/svm`, and custom.

For C, list selectable model branches:
`cls_former_classifier`, `proto_spectral_classifier`,
`spectral_dkl_gp_classifier`, `cls_former_embedding_svm`, and custom.

Before any B/C/F execution, check dependency availability. If PyTorch or other
optional dependencies are unavailable, do not install them automatically and do
not write deep config files. Ask whether to install dependencies, skip those
methods, or run only executable traditional methods.
Dependency gate: do not install them automatically; do not write deep config files.

Do not describe built-in self-developed/deep candidates as a bare code list.
Use grouped explanations:

- self-developed small-sample models: CLS-former
  (`cls_former_classifier/regressor`), prototype spectral models
  (`proto_spectral_classifier/regressor`), and DKL-GP
  (`spectral_dkl_gp_classifier/regressor`);
- self-developed/deep feature extraction or representation learning:
  `cls_former_embedding`, `contrastive_spectral_embedding`,
  `masked_spectral_autoencoder_embedding`, `self_supervised_spectral_embedding`,
  `autoencoder_embedding`, `transformer_embedding`, `cnn_1d_embedding`, and
  `resnet1d_embedding`;
- bridge option: `cls_former_embedding_svm`.

For each group, add a short Chinese note explaining what it is useful for and
why it is not in regular by default. On small high-dimensional data, recommend
regular first as the stability baseline and offer adding 1-2 representative
self-developed/deep candidates before offering a full deep search.

Use data-profile flags in every recommendation card:
`small_sample` (`n_samples<200`, `n_train<100`, or min train class `<20`),
`high_dimensional` (`p/n_train>=10` or `p/n>=10`), `very_high_dimensional`
(`>=30`), and `class_imbalance` (max/min class `>=1.5`). When the profile is
small-sample high-dimensional, explicitly offer `cls_former_embedding_svm` and
`contrastive_spectral_embedding + linear_svm/svm` as representative optional
self-developed/deep candidates.

The compact classification default includes `pls_latent_variables` because
PLS-LV is a core spectral feature extraction method. If budget pressure removes
it from a custom compact card, say so explicitly and offer extended/custom
search.

If the user already specified every method and parameter, do not invoke
optimizer; route directly through `spectral-workflow`.

After a small-sample single-holdout search selects a best pipeline, recommend a
locked stratified repeated-holdout check with 5 or 10 repeats and mean/std
Macro-F1. Treat near-perfect train accuracy as an additional overfit signal.

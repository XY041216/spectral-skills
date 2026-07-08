# Confirmation Gates

Ask only for the smallest missing decision. Do not execute a data-changing or
model-fitting step while its gate is open.

## Generic route gate

For generic requests such as "处理这个光谱数据", "处理 <file>", or a bare raw data
file path with no route choice, it is allowed to run `spectral-reader` and write
the standard package when the reader boundary is clear. Then stop and show seven
compact routes:

1. recommended baseline;
2. read/check only;
3. manual method workflow;
4. optimization comparison;
5. small-sample/deep-learning model experiment;
6. deep embedding plus traditional classifier/regressor comparison;
7. visualization-only exploration.

This reader-completion route card is mandatory. The final answer is incomplete
unless it says `下一步可以选一个路线继续` and shows the seven routes. Do not run
spectral-check, split, preprocess, feature extraction, modeling, optimizer, or
report stages until the user chooses a route. Do not show full stage menus until
the user enters a stage.

The optimization comparison route may recommend regular/traditional
combinations first, but it must also list built-in self-developed
small-sample/deep options and ask whether to include them.

User-facing terminology rule: never write `QC` in route cards, status summaries,
or next-step prompts. Use `check`, `spectral-check`, or `质量检查`. Internal
legacy file names such as `qc_result.json` may appear only as paths or API
identifiers.

The route card must be profile-aware rather than a fixed menu. Show the observed
profile first: `n_samples`, `n_features`, `p/n`, task, class balance or target
range, spectral axis, and check status when available. Then mark one
recommended route and keep all alternatives visible:

1. `推荐基线`
2. `只读取和质量检查`
3. `逐步手动选择方法`
4. `自动优化比较`
5. `小型自创/深度候选加入`
6. `全量支持方法选优`
7. `小样本/深度模型实验`
8. `深度嵌入 + 传统模型比较`
9. `可视化探索`

`全量支持方法选优` means previewing the complete implemented candidate universe
across preprocess, feature, model, optional boosting, self-developed, and deep
methods. It must report the expanded trial count and require pruning or a
high-budget confirmation before execution.

Use these data-profile flags for recommendations:

- `small_sample`: `n_samples < 200`, `n_train < 100`, or smallest train class
  size `< 20`.
- `high_dimensional`: `n_features / n_train >= 10` or `n_features / n_samples >= 10`.
- `very_high_dimensional`: either ratio `>= 30`.
- `class_imbalance`: largest class / smallest class `>= 1.5`; severe at `>= 2.0`.

When `small_sample && high_dimensional`, regular baselines remain the stability
recommendation, but the route card and manual stage cards must also offer a
small self-developed/deep add-on. For `n=120, p=3401`, explicitly say the data
qualifies and recommend considering `cls_former_embedding_svm` or
`contrastive_spectral_embedding + linear_svm/svm` as optional representative
candidates.

## Stage-card contract

Use these visible headings:

- `推荐方案`
- `为什么这样推荐`
- `本轮默认纳入`
- `本轮默认不纳入但 skill 内置支持`
- `需要用户确认的参数`
- `你可以选择`

For modeling also show `可选模型菜单`.

Every method item must be bilingual with executable code:
`中文名 (method_code / English name): risk or use note`. Never use a code-only
list, "regular", or category placeholders in place of supported methods.

## Split gate

Confirm split type, design-specific ratios/folds/repeats/groups, shuffle, and
seed. Show the complete bilingual splitter menu. Do not ask for a holdout ratio
for K-fold/LOOCV.

The split recommendation must be data-aware. For classification, prefer
stratified holdout or stratified repeated holdout; when `small_sample=true`,
mention 5- or 10-repeat stratified holdout or stratified K-fold as a stability
follow-up. If group metadata exists, recommend group-aware or stratified-group
splitting to avoid leakage.

## Preprocess gate

Confirm an explicit method, including `none`. Show the complete bilingual
preprocess menu and all method-shaping parameters. Train-fitted transforms use
train only and refit per fold/repeat.

The preprocess recommendation must not be a fixed `snv`. For unknown scatter or
baseline risk, recommend comparing `none`, `snv`, and `msc` in optimizer, or
choose `snv` only as a single manual starting point while saying why it fits the
observed spectra. For derivative/smoothing/baseline methods, require window,
order, and baseline parameters before execution.

## Feature gate

Confirm an explicit feature method, including `none`. Group the complete menu
as traditional/chemometric, projection/signal/manifold, and deep embeddings.
Deep methods require data-aware confirmation of dimensions, epochs, early
stopping status, batch, learning rate, weight decay, seed, device, and
method-specific parameters. Use 2D only for visualization unless the user
explicitly confirms a modeling experiment.

The feature recommendation must be profile-aware. If `small_sample &&
high_dimensional`, recommend a full-spectrum `none` baseline plus a compact
dimension-reduction/selection option such as PCA, PLS-LV, or VIP. Also offer a
small self-developed/deep add-on such as `cls_former_embedding_svm` or
`contrastive_spectral_embedding` paired with a confirmed downstream classifier
when the user wants to test built-in deep capability.

## Modeling gate

Group the complete menu as traditional/chemometric, optional boosting, and
small-sample deep/experimental models. List each deep model with Chinese name,
code, English name, and practical risk. A classifier/regressor set such as
`regular-fast` must show both included and excluded supported models.

The model recommendation must be data-aware. For `small_sample &&
high_dimensional`, prefer regularized/linear or chemometric baselines such as
Linear SVM, Logistic Regression, shrinkage LDA, RBF-SVM, and PLS-DA before broad
tree/deep searches. Explicitly offer CLS-former, prototype, DKL-GP, and
CLS-former embedding + SVM as optional self-developed/deep additions when the
profile qualifies, with overfitting and budget warnings.

When asking the user to choose classification/regression models, explicitly
mention built-in self-developed small-sample models (`spectral_dkl_gp_*`,
`proto_spectral_*`, `cls_former_*`, `cls_former_embedding_svm`) and ask whether
to use them in addition to regular classifiers/regressors.

Expose fixed defaults versus classifier-only tuning. State the exact grid,
selection metric, validation design, and final-test policy. Default
classification selection is Macro-F1. Test is never used for selection.

## Optimizer gate

Show Level 1 classifier tuning, Level 2 traditional pipeline tuning, and Level
3 built-in small-sample/deep model or deep embedding/classifier tuning. Offer
`quick`, `regular`, `extended`, and `deep` budgets. Confirm candidates, grids,
expanded trials, metric, repeats, seed, device, and included/excluded methods.
`deep` is opt-in only.

For `regular` recommendations, state that they are regular-method defaults
rather than the full supported model pool. The confirmation card is incomplete
unless it includes a section named `内置自创/深度候选是否加入选优组合` and asks:
`我们推荐先跑 regular 组合；同时 skill 内置自创小样本特征提取/表示学习和深度学习方法，是否加入到本轮组合选优？`

List feature/embedding candidates such as `contrastive_spectral_embedding`,
`masked_spectral_autoencoder_embedding`, `self_supervised_spectral_embedding`,
`autoencoder_embedding`, `transformer_embedding`, `cls_former_embedding`,
`cnn_1d_embedding`, and `resnet1d_embedding`. List modeling candidates such as
`spectral_dkl_gp_classifier/regressor`, `proto_spectral_classifier/regressor`,
`cls_former_classifier/regressor`, and `cls_former_embedding_svm`.

Do not present the built-in self-developed/deep candidates as a code-only list.
Introduce them as grouped options:

- `自创小样本模型`: CLS-former classifier/regressor
  (`cls_former_classifier/regressor` / CLS-former spectral model), prototype
  spectral classifier/regressor (`proto_spectral_classifier/regressor` /
  prototype spectral model), and DKL-GP
  (`spectral_dkl_gp_classifier/regressor` / deep-kernel Gaussian process).
  State that these are designed for small-sample spectral modeling or
  uncertainty-aware nonlinear modeling, but require extra confirmation of
  epochs, early stopping, seed, device, and validation design.
- `自创/深度特征提取与表示学习`: CLS-former embedding
  (`cls_former_embedding`), contrastive spectral embedding
  (`contrastive_spectral_embedding`), masked spectral autoencoder
  (`masked_spectral_autoencoder_embedding`), self-supervised spectral embedding
  (`self_supervised_spectral_embedding`), autoencoder, Transformer, CNN1D, and
  ResNet1D embeddings. State that these create train-fitted representations and
  must be evaluated with a confirmed downstream classifier/regressor inside the
  same leakage-safe split.
- `组合桥接`: CLS-former embedding + SVM (`cls_former_embedding_svm`), useful
  when the user wants a deep representation but a conventional downstream
  classifier.

For the current data profile, add a short recommendation sentence, for example:
`n=120 且 p=3401，建议先跑 regular 作为稳定基线；若要展示自创/深度能力，可先加入 1-2 个代表性候选，而不是一次性加入全部深度网格。`

End the card with choices: A `仅 regular 推荐组合`, B `regular + 选择的自创/深度特征`,
C `regular + 选择的自创/深度模型`, D `先预览 extended/deep 方案再确认`. Do not ask
only for `确认 regular 72`; that is an incomplete confirmation.

Before confirmation, use only `--preview-only`; write no candidate-space or
trial files. After selection, keep parameters locked for final test.

## Modeling completion gate

Before sending the final answer after a modeling stage, read the modeling
outputs. For multi-model comparisons, the final answer must include a Markdown
comparison table with every evaluated classifier, not just the selected model.
Minimum columns are Model, Train Macro-F1, Validation Macro-F1, Validation
Accuracy, selected parameters, and Test accessed. Explain why the winner was
selected and whether the test set remains isolated.

For `regular-fast`, explicitly list Logistic Regression, Linear SVM, RBF-SVM,
LDA, KNN, Random Forest, and Extra Trees in the final result summary.

## Report gate

Confirm chart grammar, backend, language/font, palette, layout, statistical
unit, metric unit, and size. Default to white/no-grid/full-frame/Times New
Roman/outside lowercase labels/low-saturation high-distinction colors. Assign
distinct colors to categorical methods and record the sort key. Do not default
to colorblind or monochrome palettes.

## Test gate

Ask before final test evaluation and log access. If test was already viewed,
label later evaluation confirmatory. Visual separation is not performance
evidence. Near-perfect train accuracy with lower validation/test is an
overfitting signal and triggers repeated holdout/CV follow-up.

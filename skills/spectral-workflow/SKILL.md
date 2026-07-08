---
name: spectral-workflow
description: >-
  Use when Codex needs to route a multi-stage spectral analysis across
  spectral-reader, spectral-check, spectral-splitter, spectral-preprocess,
  spectral-feature, spectral-modeling, spectral-optimizer, and spectral-report.
  It selects the minimal child skills, asks for missing confirmations,
  preserves data_contract.json and split_contract.json handoffs, and prevents
  leakage. Do not use it to implement child algorithms, use test data for
  selection, draw figures itself, or run unconfirmed optimizer searches.
---

# Spectral Workflow

Use this skill as the routing and confirmation policy. Execute algorithms only
through the child skill that owns the stage.

## Routing

Start with `static/core/route-index.md` for the stage-owner map. Load only the
active child skill and its method-selection fragment; do not load full menus for
future stages.

## Output Layout

All files created for one user-facing spectral analysis must converge under one
run directory. Prefer `spectral-workflow --output-root <base>/spectral_runs
--run-name <run_id>` so the workflow writes:

- `<output-root>/<dataset>/<run_id>/reader_package`
- `<output-root>/<dataset>/<run_id>/qc_output`
- `<output-root>/<dataset>/<run_id>/split_output`
- `<output-root>/<dataset>/<run_id>/preprocess_output`
- `<output-root>/<dataset>/<run_id>/feature_output`
- `<output-root>/<dataset>/<run_id>/model_output`
- `<output-root>/<dataset>/<run_id>/optimizer_output`
- `<output-root>/<dataset>/<run_id>/report_output`
- `<output-root>/<dataset>/<run_id>/logs`
- `<output-root>/<dataset>/<run_id>/workflow_result.json`

For a raw-file request such as "处理 <file>", do not write sibling folders such
as `<stem>_standard_package`, `<stem>_qc`, `<stem>_split`, or
`<stem>_optimizer_regular72`. Run the reader through `spectral-workflow
--task-goal read` or, if invoking `spectral-reader` directly, set its
`--output-dir` to the current run directory's `reader_package`. Every later
check, split, preprocess, feature, modeling, optimizer, or report artifact for
the same analysis must reuse that run directory and write to its stage
subfolder. If the user continues from an older package that is already outside a
run directory, create one new run directory and record the external package as
`reused_from`, rather than scattering new sibling outputs next to the raw file.

Final answers must show the run directory first, then stage outputs relative to
that directory. Avoid presenting a list of unrelated sibling folders as the
analysis result.

1. Raw file/folder -> `spectral-reader`.
2. Standard package -> validate `data_contract.json`; skip reader.
3. Before split/model workflows -> check-only `spectral-check`, unless explicitly skipped.
4. Any train-fitted step -> `spectral-splitter` first.
5. Preprocessing -> `spectral-preprocess`.
6. Feature engineering/embedding -> `spectral-feature`.
7. Classification/regression -> `spectral-modeling`.
8. Recommendation/tuning/comparison -> `spectral-optimizer`.
9. Publication figure/report -> `spectral-report`.

Prefer the one-command workflow after all decisions are confirmed. During
manual selection, invoke stages one at a time and preserve contracts.

Before execution, create `workflow_plan.json`. Mark each stage `execute`,
`skip`, or `pending_user_decision`; record decision source and expected output.
Use `update_workflow_decision.py` and `update_workflow_result.py`, not ad-hoc
JSON. These state tools perform locked atomic read-modify-write updates. Do not
edit `workflow_plan.json` by appending text or by hand-rewriting partial JSON.
If a plan cannot be parsed, stop and report the corrupted file instead of
continuing with an inferred state. An explicit `none` preprocessing/feature
choice is a skipped stage, not an executable transformation.
Task-goal aliases such as `classification_baseline` and
`baseline_classification` normalize to `classification`; baseline routing must
still execute split, preprocess, optional feature skip, and modeling stages when
their parameters are already confirmed.

When forwarding reader boundaries, distinguish source column names from
positions. `--reader-spectral-start-column` and `--reader-spectral-end-column`
name headers such as `3600` and `200`. If the user means zero-based positions,
use `--reader-spectral-start-column-index` and
`--reader-spectral-end-column-index` instead.

## Generic top-level route card

Generic raw-file handling has a mandatory reader-completion route card. For a
generic request such as "处理这个光谱数据", "处理 <file>", or a bare raw data file
path with no chosen analysis route, it is allowed to run `spectral-reader` and
write the standard package when the reader boundary is clear. After that, stop:
do not run spectral-check, splitting, preprocessing, feature extraction,
modeling, optimizer, report generation, or any downstream workflow artifact
until the user chooses a route. If reader boundaries are ambiguous, ask before
writing. If a standard package already exists, inspect its metadata and show the
same route card without rewriting it.

The final answer for this generic handling is incomplete unless it says
`下一步可以选一个路线继续` and shows the route card below. This rule supersedes
any older or garbled route text in this file.

User-facing terminology rule: never write `QC` in route cards, status summaries,
or next-step prompts. Use `check`, `spectral-check`, or `质量检查` instead.
Internal legacy names such as `qc_result.json`, `qc_output`, CLI flags, or Python
module names may remain only as file/API identifiers.

Use only the canonical route card below; ignore any older route examples in this
file that contain garbled text or the old `QC` name. The card must be
data-aware: first show the observed profile (`n_samples`, `n_features`, `p/n`,
task, class balance or target range, axis, and check status if available), then
mark the recommended route for that profile.

Canonical route card:

1. `推荐基线`: reader standard package -> spectral-check/质量检查 -> split -> data-aware preprocess -> feature baseline -> data-aware regular model comparison.
2. `只读取和质量检查`: standardize the raw file and run spectral-check only; do not split or model.
3. `逐步手动选择方法`: choose split, preprocess, feature, and model one stage at a time. Each recommendation must be derived from the current data profile, not a fixed script.
4. `自动优化比较`: run a bounded regular optimizer search; show included/excluded methods and ask about self-developed/deep add-ons.
5. `小型自创/深度候选加入`: add 1-2 representative self-developed/deep candidates such as CLS-former embedding + SVM or contrastive embedding + SVM when the data profile supports it.
6. `全量支持方法选优`: preview the all-supported candidate universe across all implemented preprocess, feature, model, optional boosting, self-developed, and deep methods; require pruning or a high-budget confirmation before execution.
7. `小样本/深度模型实验`: separately confirm prototype spectral models, CLS-former, DKL-GP, or other end-to-end/experimental models.
8. `深度嵌入 + 传统模型比较`: train confirmed deep embeddings, then compare confirmed downstream classifiers/regressors.
9. `可视化探索`: PCA/t-SNE/UMAP/deep 2D embedding for exploration only; state that visual separation is not performance evidence.

Data-aware recommendation rule:

- Compute `small_sample=true` when `n_samples < 200`, `n_train < 100`, or the
  smallest train class has fewer than 20 samples.
- Compute `high_dimensional=true` when `n_features / max(n_train,1) >= 10` or
  `n_features / max(n_samples,1) >= 10`; compute `very_high_dimensional=true`
  when the ratio is at least 30.
- Compute `class_imbalance=true` when the largest class count divided by the
  smallest class count is at least 1.5; call it severe at 2.0.
- If `small_sample && high_dimensional`, recommend regular baselines first, then
  explicitly offer a small self-developed/deep add-on. For a profile like
  `n=120, p=3401`, say it qualifies as small-sample high-dimensional
  (`p/n≈28`, `p/n_train≈47`) and recommend CLS-former embedding + SVM or
  contrastive spectral embedding + SVM as optional representative add-ons, not
  as the default replacement for regular methods.
- If `very_high_dimensional`, prefer `none` baseline plus PCA/PLS-LV/VIP or
  linear/regularized models before broad tree/deep sweeps.
- If check status is warning, say the warnings are marked and whether they
  block the next step; do not say `QC`.

Show this visible route card in Chinese:

1. `确认推荐基线`: reader standard package -> check -> stratified 6:2:2 seed=42 -> SNV -> feature=none -> SVM.
2. `只读取和检查`: only standardize the data and run quality/data checks.
3. `手动选择方法`: choose split, preprocess, feature, and classifier/regressor step by step; mention regular methods first, then self-developed small-sample/deep options.
4. `优化比较`: compare preprocessing, feature, and modeling combinations. Recommend a regular bounded plan first, then ask whether to include built-in self-developed small-sample/deep candidates.
5. `小样本/深度模型实验`: separately confirm prototype spectral models, CLS-former, DKL-GP, or other end-to-end/experimental models.
6. `深度嵌入 + 传统模型比较`: train AE/CNN/ResNet/CLS-former/masked/contrastive embeddings, then compare SVM/LDA/RF/ET or other confirmed downstream models.
7. `可视化探索`: PCA/t-SNE/UMAP/deep 2D embedding for exploration only; state that visual separation is not performance evidence.

For a generic request such as "澶勭悊杩欎釜鍏夎氨鏁版嵁", inspect the file read-only,
summarize `n`, `p`, axis, task, and class/target profile, then stop for one route
choice. Do not show full method menus at this point.

Show exactly these seven routes:

1. `纭鎺ㄨ崘鍩虹嚎`锛氳鍙栨爣鍑嗗寘 -> QC -> stratified 6:2:2 seed=42 -> SNV -> feature=none -> SVM銆?2. `鍙鍙栧拰 QC`锛氬彧鐢熸垚鏍囧噯鍏夎氨鍖呭苟鍋氳川閲忔鏌ャ€?3. `鎵嬪姩閫夋嫨鏂规硶`锛氶€愭閫夋嫨鍒掑垎銆侀澶勭悊銆佺壒寰佸拰鍒嗙被/鍥炲綊妯″瀷锛涘彲鍏堟帹鑽愬父瑙勬柟娉曪紝浣嗗繀椤昏鏄庤繕鏀寔灏忔牱鏈繁搴?瀹為獙妯″瀷銆?4. `浼樺寲姣旇緝`锛氱敤 optimizer 姣旇緝棰勫鐞嗐€佺壒寰佸拰妯″瀷缁勫悎銆傞粯璁ゅ彲鎺ㄨ崘甯歌缁勫悎锛涘悓鏃跺垪鍑哄唴缃嚜鍒涘皬鏍锋湰妯″瀷鍜屾繁搴﹀涔犳ā鍨嬶紝骞惰闂敤鎴锋槸鍚︾撼鍏ョ粍鍚堟瘮杈冦€?5. `娣卞害瀛︿範妯″瀷瀹為獙`锛氬崟鐙€夋嫨鍘熷瀷鍒嗙被鍣ㄣ€丆LS-former銆丏KL-GP 绛夌鍒扮/瀹為獙妯″瀷銆?6. `娣卞害宓屽叆 + 浼犵粺鍒嗙被鍣ㄦ瘮杈僠锛氳缁?AE/CNN/ResNet/CLS-former/masked/contrastive embeddings锛屽啀姣旇緝 SVM/LDA/RF/ET 绛夈€?7. `鍙鍖栨帰绱锛歅CA/t-SNE/UMAP/deep 2D embedding锛涙槑纭笉浣滀负鍒嗙被鎬ц兘璇佹嵁銆?
Do not collapse routes 5-7 into ordinary model selection. Deep routes carry
their own training protocol, device, random-seed, budget, and risk gates.

For route 1 baselines, use `--task-goal classification` or
`--task-goal classification_baseline`, plus `--require-test-confirmation` unless
the user has already explicitly approved final test access. The workflow should
run modeling in `validation_only` mode first, write validation artifacts and
`workflow_result.json`, then ask for one explicit final-test confirmation. After
confirmation, rerun the same locked workflow with `--confirm-test-evaluation`.

## Stage confirmation cards

Use these visible headings for every stage card; they supersede any older or
garbled heading text in this file:

- `推荐方案`
- `为什么这样推荐`
- `本轮默认纳入`
- `本轮默认不纳入但 skill 内置支持`
- `需要用户确认的参数`
- `你可以选择`

Modeling cards must also show `可选模型菜单`.

Read the selected child `SKILL.md` and its method-selection fragment. Every
stage card must use these visible headings:

- `鎺ㄨ崘鏂规`
- `涓轰粈涔堟帹鑽恅
- `鏈疆榛樿绾冲叆`
- `skill 杩樻敮鎸佷絾鏈疆榛樿涓嶇撼鍏
- `闇€瑕侀澶栫‘璁ゅ墠鎵嶈兘鎵ц`
- `浣犲彲浠ラ€夋嫨`

Modeling cards must also show `鑷姩璋冨弬鑳藉姏`.

Use `涓枃鍚嶇О锛坢ethod_code / English name锛夛細璇存槑` for every method. English-only
lists, "绠€鐗?, "绛?, or category placeholders are invalid substitutes for the
complete stage menu. Long menus may be grouped.

Ask only for the smallest current decision: split -> preprocess -> feature ->
model. Do not collect all choices at once unless the user explicitly requests a
compact confirmation card.

### Split

Show all supported holdout, K-fold/LOOCV, repeated holdout/MCCV,
Kennard-Stone/SPXY/Duplex, regression-binned, and group-aware designs in
bilingual form. Confirm method-specific parameters and seed. Do not request a
holdout ratio for CV designs.

### Preprocess

Show the complete bilingual preprocess menu from `spectral-preprocess`. Confirm
SG/derivative, baseline, absorbance/log, and band-range parameters explicitly.
Fit train-learned transforms on train only; refit per fold/repeat.

### Feature

Separate the full menu into:

- traditional/chemometric selection;
- projection/signal/manifold methods;
- deep train-fitted embeddings.

List deep methods item by item with Chinese name, code, English name, and risk
note. Discovery embeddings and 2D deep embeddings are visualization-first.

### Model

Separate:

- traditional ML/chemometric classifiers;
- optional boosting;
- small-sample deep/experimental spectral models.

List every supported model item by item. `regular-fast` is a budget choice, not
the whole capability surface. When recommending a regular/traditional model
set, explicitly say it excludes the built-in self-developed small-sample/deep
options and ask whether to include them: `spectral_dkl_gp_classifier/regressor`,
`proto_spectral_classifier/regressor`, `cls_former_classifier/regressor`,
`cls_former_embedding_svm`, and feature-stage deep embeddings such as
`cnn_1d_embedding`, `resnet1d_embedding`, `transformer_embedding`,
`autoencoder_embedding`, `masked_spectral_autoencoder_embedding`, and
`contrastive_spectral_embedding` with confirmed downstream classifiers.

## Deep-route protocol

Before any deep embedding or experimental classifier, show a data-aware card
based on `n_samples`, `n_train`, `n_features`, class count/balance, split,
intended use, and device.

Confirm:

- embedding/feature dimension;
- maximum epochs and early-stopping status/patience;
- batch size, learning rate, weight decay;
- seed and device;
- model-specific noise, masking, temperature, patch, dropout, distance, kernel,
  or downstream-classifier parameters;
- selection metric and final-test policy.

For `n<=120`, `n_train<=72`, `p>=3401`, and balanced four-class classification,
recommend 16 dimensions for a first classification embedding and reserve 2D for
visualization. Typical small-sample starting points are batch 8-16,
`weight_decay=1e-4` for CNN/ResNet/Transformer families, and bounded epochs with
validation early stopping when implemented. ResNet1D must carry a high-overfit risk warning.
A fixed epoch count does not prove convergence.

Do not apply one shared default bundle to every deep method. Use the
method-specific recommendations in `spectral-feature` or `spectral-modeling`.

## Tuning routes and budgets

Do not say only "automatic tuning". Show:

- Level 1锛歝lassifier-only tuning.
- Level 2锛歵raditional preprocessing/feature + classifier tuning.
- Level 3锛歜uilt-in small-sample/deep model or deep embedding + classifier tuning; extra confirmation required.

Then offer optimizer budgets:

- `quick`锛歴moke/interactive search; small single-point or narrow grids.
- `regular`锛歳ecommended bounded traditional search; primary default, but not the full skill capability surface.
- `extended`锛歜roader traditional grids and repeated-validation follow-up.
- `deep`锛歟xplicit deep-search protocol; never a default.

Every optimizer card must state candidates, fixed stages, exact grids, expanded
trial count, selection metric, validation design, repeats, seed, device, and
included/excluded methods. Use validation Macro-F1 or repeated validation
Macro-F1 mean 卤 SD for classification unless the user confirms another metric.
Never use final-test metrics for candidate selection.

For any optimization-comparison request, do not frame the route as only
conventional methods. The confirmation card is incomplete unless it contains a
required section named `内置自创/深度候选是否加入选优组合`. Show the recommended
regular/traditional combination first, then list built-in self-developed or
experimental small-sample models and deep learning options separately, with
practical risks and added confirmation needs. The card must explicitly say:
`我们推荐先跑 regular 组合；同时 skill 内置自创小样本特征提取/表示学习和深度学习方法，是否加入到本轮组合选优？`

List candidate codes item by item: feature/embedding candidates
`contrastive_spectral_embedding`, `masked_spectral_autoencoder_embedding`,
`self_supervised_spectral_embedding`, `autoencoder_embedding`,
`transformer_embedding`, `cls_former_embedding`, `cnn_1d_embedding`,
`resnet1d_embedding`; modeling candidates
`spectral_dkl_gp_classifier/regressor`, `proto_spectral_classifier/regressor`,
`cls_former_classifier/regressor`, and `cls_former_embedding_svm`.

Do not present these candidates as a bare code list. Group and explain them in
Chinese: self-developed small-sample models (CLS-former, prototype spectral
models, DKL-GP), self-developed/deep feature extraction or representation
learning (CLS-former embedding, contrastive, masked autoencoder,
self-supervised, autoencoder, Transformer, CNN1D, ResNet1D), and the bridge
option CLS-former embedding + SVM. For each group, state intended use and risk:
small-sample/nonlinear spectral modeling, uncertainty-aware modeling,
train-fitted representations, extra epochs/device/early-stopping confirmation,
and overfitting risk. For `n=120, p=3401`, recommend regular first and offer
adding 1-2 representative self-developed/deep candidates before a full deep
search.

End the card with choices: A `仅 regular 推荐组合`, B `regular + 选择的自创/深度特征`,
C `regular + 选择的自创/深度模型`, D `先预览 extended/deep 方案再确认`. Do not ask
only for `确认 regular 72`; that is an incomplete confirmation.

Level 3/deep must confirm a compact search such as:
`n_components=[8,16,32]`, method-specific epochs/patch/augmentation parameters,
and downstream classifiers `[linear_svm, svm, lda]`. Do not launch it merely
because the user asked for "best".

Use optimizer `--preview-only` before materializing a plan. Require complete
locked parameters and explicit budget/grid confirmation. After selecting a
pipeline, run final modeling with locked parameters; do not retune on test.
When evaluating an optimizer `best_pipeline`, pass
`--best-pipeline <best_pipeline.json>` and `--lock-best-pipeline-params` to
`spectral-modeling`. The locked replay must include preprocess, feature, model,
and parameters from the optimizer trial lineage. If any non-none upstream stage
cannot be resolved to a contract, stop instead of manually rebuilding a partial
pipeline.

For small single-holdout results, recommend a locked 5- or 10-repeat stratified
holdout confirmation. Near-perfect train accuracy plus lower validation/test is
an overfitting signal, not proof that the tuner failed.

For locked-parameter robustness after a single holdout result, first clarify the
design name: `stratified_monte_carlo_cv` is repeated stratified holdout, not
strict repeated K-fold CV. Preserve the original ratio when the user asks to
confirm a `6:2:2` baseline; pass `--split-ratio 6:2:2` or explicit
`--train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2`, not the repeated-holdout
default `0.7/0.3`.

Use workflow-level locked repeated confirmation when the selected model and
parameters are already fixed. Write a compact model config such as:

```json
{"models":{"svm":{"C":1.0,"gamma":"scale","class_weight":null}}}
```

Then run the workflow with
`--split-method stratified_monte_carlo_cv --split-ratio 6:2:2 --n-repeats 10
--preprocess-methods snv --feature-method none --models svm --model-config
<locked-model-config.json> --modeling-mode repeated_classifier_comparison
--candidate-model-set-source locked_previous_svm
--confirm-confirmatory-test-evaluation`. This route forwards
`--disable-model-selection` to `spectral-modeling`; do not rerun internal
hyperparameter search during confirmatory robustness analysis.

## Modeling completion handoff

After `spectral-modeling` completes, read `metrics.json`,
`modeling_summary.json`, and `modeling_contract.json` before replying. If more
than one model was evaluated, the final answer must include a Markdown
comparison table with one row per model. Do not summarize only the selected model.

For holdout validation-only classifier comparisons, read
`classifier_validation_summary.csv` first; it is the authoritative one-row-per-
classifier summary and avoids rerunning each classifier separately.

For classifier comparisons, include at least:

- Model;
- Train Macro-F1;
- Validation Macro-F1;
- Validation Accuracy;
- Validation Balanced Accuracy if available;
- selected/tuned parameters;
- Test accessed.

Then explain the selection criterion, the winning model, why it won, whether
test remains isolated, any overfit/instability signal, output paths, and the
next safe step. For `regular-fast`, explicitly name Logistic Regression, Linear
SVM, RBF-SVM, LDA, KNN, Random Forest, and Extra Trees in the result summary.

## Report routing

Before plotting, confirm chart grammar, Python/Matplotlib-Seaborn backend,
language/font, palette, panel layout, statistical unit, metric unit, and final
size. Default style: white background, no grid, full black frames, Times New Roman,
outside lowercase panel labels, and low-saturation high-distinction colors. Do not default to a colorblind palette.

When categories are methods, assign distinct low-saturation colors by method.
When categories are metrics, use stable metric colors. Do not give every method
the same fill unless monochrome is confirmed. Record abbreviation mapping and
sorting rule in the caption and `report_contract.json`.

## Leakage and test policy

- Never fit preprocessing/features on full data before split.
- Never use test metrics for method, dimension, epoch, or hyperparameter choice.
- Validation-only optimizer trials must not access test.
- Ask before final test evaluation; log test access.
- If test was viewed earlier, label later results confirmatory, not blind.
- If a confirmatory test score looks high after prior test access, do not call
  it final best evidence. Recommend 10 repeated held-out splits or repeated CV
  with selection performed inside each repeat/fold.
- Visual embedding separation is not predictive-performance evidence.
- Existing scripts/results/cache are not user confirmation.

## Handoffs

Preserve canonical contracts: `data_contract.json`, `qc_result.json`,
`split_contract.json`, `preprocess_contract.json`, `feature_contract.json`,
`modeling_contract.json`, `optimizer_contract.json`, and `report_contract.json`.
Final `workflow_result.json` must point to actual outputs and record stage
decisions, test-access status, and final artifacts.

## Read as needed

- `static/fragments/confirmation-gates.md`
- `static/core/route-index.md`
- child skill `SKILL.md` and method-selection fragment for the active stage
- optimizer search-space rules for tuning requests
- report figure-contract and classification rules for plotting requests

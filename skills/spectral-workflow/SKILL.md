---
name: spectral-workflow
description: >-
  Use when Codex needs to route a multi-stage spectral analysis across
  spectral-reader, spectral-qc, spectral-splitter, spectral-preprocess,
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

1. Raw file/folder -> `spectral-reader`.
2. Standard package -> validate `data_contract.json`; skip reader.
3. Before split/model workflows -> check-only `spectral-qc`, unless explicitly skipped.
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

For a generic request such as "处理这个光谱数据", inspect the file read-only,
summarize `n`, `p`, axis, task, and class/target profile, then stop for one route
choice. Do not show full method menus at this point.

Show exactly these seven routes:

1. `确认推荐基线`：读取标准包 -> QC -> stratified 6:2:2 seed=42 -> SNV -> feature=none -> SVM。
2. `只读取和 QC`：只生成标准光谱包并做质量检查。
3. `手动选择常规方法`：逐步选择划分、预处理、传统特征和传统分类器。
4. `常规优化比较`：用 optimizer 比较预处理、传统特征和传统分类器组合。
5. `深度学习模型实验`：单独选择原型分类器、CLS-former、DKL-GP 等端到端/实验模型。
6. `深度嵌入 + 传统分类器比较`：训练 AE/CNN/ResNet/CLS-former/masked/contrastive embeddings，再比较 SVM/LDA/RF/ET 等。
7. `可视化探索`：PCA/t-SNE/UMAP/deep 2D embedding；明确不作为分类性能证据。

Do not collapse routes 5-7 into ordinary model selection. Deep routes carry
their own training protocol, device, random-seed, budget, and risk gates.

For route 1 baselines, use `--task-goal classification` or
`--task-goal classification_baseline`, plus `--require-test-confirmation` unless
the user has already explicitly approved final test access. The workflow should
run modeling in `validation_only` mode first, write validation artifacts and
`workflow_result.json`, then ask for one explicit final-test confirmation. After
confirmation, rerun the same locked workflow with `--confirm-test-evaluation`.

## Stage confirmation cards

Read the selected child `SKILL.md` and its method-selection fragment. Every
stage card must use these visible headings:

- `推荐方案`
- `为什么推荐`
- `本轮默认纳入`
- `skill 还支持但本轮默认不纳入`
- `需要额外确认前才能执行`
- `你可以选择`

Modeling cards must also show `自动调参能力`.

Use `中文名称（method_code / English name）：说明` for every method. English-only
lists, "简版", "等", or category placeholders are invalid substitutes for the
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
the whole capability surface.

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

- Level 1：classifier-only tuning.
- Level 2：traditional preprocessing/feature + classifier tuning.
- Level 3：deep embedding + classifier tuning; extra confirmation required.

Then offer optimizer budgets:

- `quick`：smoke/interactive search; small single-point or narrow grids.
- `regular`：recommended bounded traditional search; primary default.
- `extended`：broader traditional grids and repeated-validation follow-up.
- `deep`：explicit deep-search protocol; never a default.

Every optimizer card must state candidates, fixed stages, exact grids, expanded
trial count, selection metric, validation design, repeats, seed, device, and
included/excluded methods. Use validation Macro-F1 or repeated validation
Macro-F1 mean ± SD for classification unless the user confirms another metric.
Never use final-test metrics for candidate selection.

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

---
name: spectral-modeling
description: >-
  Use when Codex needs to train, tune, compare, and evaluate classification or
  regression models from an already split standard spectral package,
  preprocess contract, or feature contract. It fits train only, selects with
  validation or train-only CV, and evaluates test only after locking the model.
  It supports traditional ML, chemometric, optional boosting, and gated
  experimental small-sample spectral models. Do not use it to read raw files,
  run QC/splitting/preprocessing/features, tune upstream stages, use test for
  selection, or create reports.
---

# Spectral Modeling

Use this skill only after a split-aware package, preprocess contract, or feature
contract exists. Preserve upstream lineage and test isolation.

## Primary execution

Use `scripts/model_spectral_package.py` or the core modeling workflow. Inputs
include package/feature contract, split contract, task type, model set,
parameters, tuning mode, seed, evaluation mode, and output directory.

## Model selection gate

If model(s) are missing, read `static/fragments/model-selection.md` and show a
complete grouped bilingual menu:

- traditional ML and chemometric models;
- optional boosting models;
- small-sample deep/experimental spectral models.

Every entry must use `中文名称（method_code / English name）：用途、依赖或风险说明`.
A set such as `regular-fast` must list included classifiers and all supported
but excluded options. It is a budget recommendation, not the full capability
surface.

Classification includes LR, Linear/RBF SVM, LDA/QDA, Gaussian NB, PLS-DA,
SIMCA, ET, Gradient Boosting, RF, KNN, MLP, optional XGBoost/LightGBM/CatBoost,
and gated DKL-GP/prototype/CLS-former models.

CNN1D/ResNet1D are feature-stage embeddings rather than direct modeling
classifier codes; pair them with a confirmed downstream classifier.

## Automatic tuning

Offer:

- fixed parameters (`--no-param-search`);
- Level 1 bounded classifier-only tuning;
- Level 2 traditional feature/pipeline tuning via `spectral-optimizer`;
- Level 3 deep embedding/classifier tuning via explicitly confirmed optimizer deep search.

Use validation or train-only CV only. Classification selection defaults to
Macro-F1; disclose the metric and grid. Never use final-test metrics for
selection.

The regular Level 1 grids cover:

- Logistic: `C`, L2 penalty, `class_weight`;
- RBF-SVM: `C`, `gamma`, `class_weight`;
- Linear SVM: `C`, `class_weight`;
- KNN: neighbors, weights, distance metric;
- RF/ET: trees, depth, max features, minimum leaf size;
- LDA/QDA: shrinkage or regularization;
- optional boosting: learning rate, depth/leaves, sampling, and regularization.

Optional boosting must not use a two-parameter-only grid. XGBoost includes
`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `reg_lambda`;
LightGBM includes trees, leaves, learning rate, regularization; CatBoost
includes iterations, depth, learning rate, and L2 leaf regularization.

## Experimental small-sample models

These require explicit parameter and runtime confirmation:

- 光谱 DKL-GP 分类器（`spectral_dkl_gp_classifier` / Spectral DKL-GP classifier）：confirm kernel, preprojection, embedding dimension, training budget, early stopping, and device.
- 原型光谱分类器（`proto_spectral_classifier` / Prototype spectral classifier）：confirm `embedding_dim`, epochs, batch, learning rate, distance metric, early stopping, and device.
- CLS-former 分类器（`cls_former_classifier` / CLS-former classifier）：confirm feature dimension, network width/depth, dropout, epochs, batch, learning rate, weight decay, patience, and device.
- CLS-former 嵌入 + SVM（`cls_former_embedding_svm` / CLS-former embedding plus SVM）：confirm both the embedding protocol and the downstream SVM grid.

Experimental models with explicit user-confirmed parameters are valid locked
models. `--disable-model-selection` must accept confirmed experimental
parameters when the method has no registry grid; use a single-candidate locked
configuration rather than rejecting the parameters as unknown.

For `n<=120`, `p>=3401`, recommend small embeddings (8/16), batch 8-16,
nontrivial regularization, and validation early stopping when the implementation
supports it. Treat train=1.0 with lower validation/test as an overfit signal.
Recommend repeated holdout/CV; do not call it proof that tuning failed.

## Evaluation modes

- Validation-only: optimizer/model comparison; no test access.
- Final locked test: test exactly once after model/parameters are fixed.
- CV/repeated holdout: train/evaluate independently per fold/repeat.
- Repeated classifier comparison: disable per-repeat model selection and write
  one row per `(repeat_id, model_method)` using identical split IDs.

For holdout validation-only multi-classifier comparisons, write
`classifier_validation_summary.csv` with one row per classifier family. Include
train/validation accuracy, balanced accuracy, macro-F1, ROC-AUC when available,
selected parameters, selection metric, selection score, whether it was selected,
and `test_accessed=false`. This file is the authoritative comparison table.

For repeated classifier comparison, require an explicit classifier set and
pipeline. Write `classifier_repeat_metrics.csv`,
`classifier_metric_summary.csv`, predictions when available, and a comparison
contract. Report Accuracy, Balanced accuracy, and Macro-F1 mean ± SD with
statistical unit `repeat`.

## Best-pipeline replay

`--best-pipeline --lock-best-pipeline-params` must reproduce the full locked
pipeline selected by the optimizer: preprocess contract, feature contract,
model method, and model parameters. Recover upstream contracts from optimizer
trial lineage such as `trial_dir/feature_output/feature_contract.json` or
`trial_dir/preprocess_output/preprocess_contract.json`.

If `best_pipeline` records a non-none preprocess/feature stage but the upstream
contract cannot be resolved, block with an explicit error instead of silently
evaluating only the model on the wrong input matrix. Record
`best_pipeline_reproduction` in `modeling_contract.json`.

## Final response contract

When a modeling or classifier-comparison stage completes, the final answer must
not collapse the result to only the selected/best model. Always include:

1. a Markdown comparison table for every evaluated model;
2. columns for `Model`, `Train Macro-F1`, `Val Macro-F1`, `Val Accuracy`,
   `Val Balanced Accuracy` when available, `Selection metric`, selected
   parameters, and `Test accessed`;
3. a short explanation of why the selected model won, using validation/CV
   metrics only;
4. test status: `not accessed`, `final locked test`, or `confirmatory test`;
5. overfitting/instability notes, especially train-vs-validation gaps;
6. paths to `metrics.json`, `modeling_summary.json`,
   `modeling_contract.json`, and comparison CSV files if present;
7. a next-step recommendation, for example locked final-test evaluation,
   repeated holdout confirmation, or report plotting.

Prefer `classifier_validation_summary.csv` when it exists; do not rerun each
classifier merely to reconstruct a comparison table.

For a `regular-fast` validation-only comparison on a CLS-former embedding, the
answer must explicitly mention all evaluated classifiers: Logistic Regression,
Linear SVM, RBF-SVM, LDA, KNN, Random Forest, and Extra Trees. If the script only
prints the best model to stdout, read the metrics/summary/trace files before
responding and construct the table from those files.

## Test and leakage policy

- Fit train only; select on validation/train-only CV.
- Do not access test during optimizer trials.
- Ask before final/confirmatory test evaluation and write `test_access_log.json`.
- If test was already viewed, label later results confirmatory rather than blind.
- A saved optimizer best pipeline must remain parameter-locked during test.
- A high single-split confirmatory test score is not final-best evidence after
  prior test access; recommend 10 repeated held-out splits or repeated CV with
  selection performed inside each repeat/fold.

## Outputs

Write modeling contract, metrics, predictions, selection trace, confusion
matrix/artifacts where applicable, parameter sources, test-access log, and a
deployable `pipeline_bundle/` for final holdout models. Holdout validation-only
multi-classifier runs also write `classifier_validation_summary.csv`. Repeated
designs also write per-iteration contracts and aggregate summaries.

## Block when

- split/package/labels are missing or inconsistent;
- task type or model set is unresolved;
- experimental-model parameters are unconfirmed;
- final-test access is unconfirmed;
- an optimizer-selected model would be retuned on test;
- a best pipeline declares non-none upstream preprocess/feature stages but lacks resolvable contracts;
- repeated comparison lacks per-classifier outputs or a confirmed candidate set.

## Read as needed

- `static/fragments/model-selection.md`
- registry and script help for exact model names/parameters

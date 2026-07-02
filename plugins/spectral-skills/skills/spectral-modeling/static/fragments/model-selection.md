# Model Selection

When the model is unspecified, show a recommendation first and then the full
grouped bilingual menu. Every entry must use
`中文名称（method_code / English name）：用途、依赖或风险说明`.

## Supported classification models

### 传统机器学习与化学计量分类器（traditional ML and chemometric classifiers）

- 逻辑回归（`logistic_regression` / Logistic Regression）：线性、可解释基线。
- 线性 SVM（`linear_svm` / Linear SVM）：高维小样本强基线。
- RBF 支持向量机（`svm` / RBF-SVM）：常用非线性光谱基线。
- 线性判别分析（`lda` / Linear Discriminant Analysis）：低维嵌入或收缩后更稳定。
- 二次判别分析（`qda` / Quadratic Discriminant Analysis）：小样本需要正则化。
- 高斯朴素贝叶斯（`gaussian_nb` / Gaussian Naive Bayes）：概率基线。
- PLS-DA（`pls_da` / Partial Least Squares Discriminant Analysis）：化学计量分类基线。
- SIMCA（`simca` / Soft Independent Modeling of Class Analogy）：类别建模分类器。
- Extra Trees 分类器（`extra_trees_classifier` / Extra Trees classifier）：随机化树集成。
- 梯度提升分类器（`gradient_boosting_classifier` / Gradient Boosting classifier）：宽光谱上可能较慢。
- 随机森林分类器（`random_forest_classifier` / Random Forest classifier）：树集成基线。
- K 近邻分类器（`knn_classifier` / KNN classifier）：对缩放和特征几何敏感。
- 多层感知机分类器（`mlp_classifier` / MLP classifier）：浅层神经基线；小样本需谨慎。

### 可选 boosting 分类器（optional boosting classifiers）

- XGBoost 分类器（`xgboost_classifier` / XGBoost classifier）：需要 XGBoost；调学习率、深度、采样和正则。
- LightGBM 分类器（`lightgbm_classifier` / LightGBM classifier）：需要 LightGBM；调叶子数、学习率和正则。
- CatBoost 分类器（`catboost_classifier` / CatBoost classifier）：需要 CatBoost；调深度、学习率和 L2 leaf 正则。

### 小样本深度学习与实验光谱模型（small-sample deep/experimental models）

- 光谱 DKL-GP 分类器（`spectral_dkl_gp_classifier` / Spectral DKL-GP classifier）：深度核学习 + 高斯过程；确认 kernel、嵌入维度、早停预算和设备。
- 原型光谱分类器（`proto_spectral_classifier` / Prototype spectral classifier）：prototype/metric learning；从 `embedding_dim=8/16` 起步并确认距离度量。
- CLS-former 分类器（`cls_former_classifier` / CLS-former classifier）：端到端 Transformer 分类；确认特征维度、dropout、epochs、batch、学习率、早停和设备。
- CLS-former 嵌入 + SVM（`cls_former_embedding_svm` / CLS-former embedding plus SVM）：同时确认嵌入训练协议和下游 SVM 参数搜索。

CNN1D/ResNet1D are currently `spectral-feature` deep embedding methods rather
than direct modeling classifier codes; after embedding, pair them with
SVM/LDA/RF/ET or another confirmed classifier.

## Transparent recommendation card

Every classifier recommendation must show:

- `推荐方案`
- `为什么推荐`
- `本轮默认纳入`
- `skill 还支持但本轮默认不纳入`
- `需要额外确认前才能执行`
- `自动调参能力`
- `你可以选择`

Recommended `regular-fast` contains Logistic Regression, Linear SVM, RBF-SVM,
LDA, KNN, Random Forest, and Extra Trees. State explicitly that it excludes QDA,
Gaussian NB, PLS-DA, SIMCA, Gradient Boosting, MLP, optional boosting, and
experimental/deep models. Offer `regular-full`, optional boosting, experimental
models, or a custom list.

## Classifier tuning

Classifier tuning is bounded and leakage-safe. Tune only inside validation or
train-only CV; never use final-test metrics for selection. Use Macro-F1 as the
default classification selection metric, especially for small or imbalanced
datasets, and disclose the metric in the confirmation card and contract.

Offer three levels:

- Level 1：只调分类器参数。
- Level 2：传统预处理/特征 + 分类器联合调优，由 `spectral-optimizer` 执行。
- Level 3：深度嵌入 + 分类器调优，必须额外确认计算预算。

For Level 1, expose the actual grid. A formal regular grid should cover:

- RBF-SVM：`C`, `gamma`, `class_weight`.
- Linear SVM：`C`, `class_weight`.
- Logistic：`C`, `penalty`, `class_weight`.
- KNN：`n_neighbors`, `weights`, `metric`.
- RF/ET：`n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
- LDA/QDA：`shrinkage` or `reg_param`.
- XGBoost：`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `reg_lambda`.
- LightGBM：`n_estimators`, `num_leaves`, `learning_rate`, `reg_lambda`.
- CatBoost：`iterations`, `depth`, `learning_rate`, `l2_leaf_reg`.

If train accuracy is near 1.0 but validation/test is lower, call it an
overfitting signal under the confirmed pipeline, not proof that tuning is broken.
Recommend repeated holdout/CV and Level 2/3 comparison. Experimental
deep models must show data-aware parameters: for `n<=120, p>=3401`, prefer
small embeddings (8/16), batch 8-16, nontrivial weight decay, validation early
stopping when implemented, and an explicit high-variance warning.

## Completion summary for model comparisons

After any multi-model comparison, final user-facing output must include a
Markdown table with one row per evaluated model/classifier. Minimum columns:

- Model
- Train Macro-F1
- Validation Macro-F1
- Validation accuracy
- Validation balanced accuracy, if available
- Selected/tuned parameters
- Test accessed

Use `classifier_validation_summary.csv` when it exists; it is the official
holdout validation-only one-row-per-classifier output, so do not rerun every
classifier just to build the table.

Then add a short explanation:

- which model was selected and by which validation/CV metric;
- whether the test set remains isolated;
- whether train-vs-validation gaps suggest overfitting;
- where to find `metrics.json`, `modeling_summary.json`,
  `modeling_contract.json`, predictions, and comparison CSV outputs;
- what the next safe step is.

Do not report only the selected model when multiple classifiers were evaluated.
For `regular-fast`, explicitly mention Logistic Regression, Linear SVM, RBF-SVM,
LDA, KNN, Random Forest, and Extra Trees.

If test was already viewed and a confirmatory test score looks high, do not call
it final-best evidence. Recommend 10 repeated held-out splits or repeated CV
with selection performed inside each repeat/fold.

## Supported regression models

Show regression methods with the same bilingual structure: `plsr`, `pcr`,
`linear_regression`, `ridge`, `lasso`, `elastic_net`, `bayesian_ridge`, `svr`,
`knn_regressor`, `random_forest_regressor`, `extra_trees_regressor`,
`gradient_boosting_regressor`, `gpr`, optional XGBoost/LightGBM/CatBoost, and
experimental DKL-GP/prototype/CLS-former regressors.

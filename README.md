# Spectral Skills

面向 Codex、Claude 等智能体的端到端光谱数据分析技能集合。项目覆盖光谱读取、质量控制、数据划分、预处理、特征工程、分类/回归建模、受预算约束的自动优化，以及论文级图表与报告输出。

Spectral Skills is a leakage-aware agent skill collection for end-to-end spectral analysis. It ships Codex and Claude-compatible plugin metadata from one source tree.

## 核心特点

- **端到端但不做成黑箱**：九个 skill 各自负责一个清晰阶段，由 `spectral-workflow` 路由组合。
- **防止数据泄漏**：先划分，再拟合预处理、特征和模型；测试集不参与方法选择或调参。
- **完整方法菜单**：推荐方案只是预算内默认值，不代表系统只支持推荐列表中的方法。
- **自动调参可审计**：候选空间、trial、选择指标、最佳参数和最终管线均写入合同文件。
- **适配小样本光谱**：同时提供化学计量学、传统机器学习、可选 boosting 和需确认的实验深度模型。
- **结果可追溯**：阶段间通过 JSON contract 和标准 CSV 交接，而不是依靠临时内存状态。
- **论文级绘图**：输出可编辑 SVG/PDF、PNG 预览、源数据、图注、绘图代码和 QA 记录。

## 工作流总览

```text
原始光谱文件或文件夹
        |
        v
spectral-reader  ->  spectral-check  ->  spectral-splitter
                                            |
                                            v
spectral-preprocess  ->  spectral-feature  ->  spectral-modeling
        \                    |                       /
         \                   v                      /
          +---------- spectral-optimizer ----------+
                               |
                               v
                        spectral-report

spectral-workflow：负责路由、确认、合同传递与测试集隔离
```

典型分类链路：

```text
读取标准包 -> 非破坏性质量检查 -> 分层划分 -> SNV -> PCA/VIP/全光谱 -> SVM/PLS-DA/模型比较 -> 最终测试 -> 论文图
```

典型回归链路：

```text
读取标准包 -> 质量检查 -> Kennard-Stone/KFold -> MSC/SNV -> PLS-LV/SPA/CARS -> PLSR/SVR/集成回归 -> 最终测试 -> 回归图
```

## 九个 Skill 的详细能力

### 1. `spectral-reader`：光谱数据读取与标准化

将不同来源和布局的光谱数据转换为下游统一使用的标准数据包。它只负责“正确读入”，不做质量检查、插补、删样本、预处理或建模。

**支持的输入格式**

- 文本表格：CSV、TSV、TXT；
- 工作簿：Excel、ODS；
- 数组/科学容器：NPY、NPZ、MAT（非 v7.3）、HDF5、NetCDF；
- 外部标签、目标值、元数据和独立波段轴文件；
- 每个样本一个 CSV/TSV/TXT 文件的文件夹；
- 含光谱、标签、元数据和波段轴候选文件的混合目录。

**支持的数据布局**

- 样本按行或经确认后样本按列；
- 数值波长/波数列名，如 `350–2500 nm`、`3600–200 cm-1`；
- 表头前有注释或仪器导出说明；
- 光谱列前后夹有样本 ID、标签、目标值或元数据；
- 多行表头、Excel 多 sheet、NPZ/MAT 变量选择、HDF5/NetCDF dataset path；
- 按 `sample_id` 对齐外部标签，或经确认后按行序对齐；
- 从文件名或文件夹名提取标签（必须显式要求）。

**标准输出**

| 文件 | 作用 |
| --- | --- |
| `X.csv` | 数值光谱矩阵，行是样本，列是波段 |
| `y.csv` | 可选的分类标签或回归目标 |
| `sample_ids.csv` | 样本唯一标识 |
| `band_axis.csv` | 波长、波数或索引轴 |
| `metadata.csv` | 可选样本元数据 |
| `data_contract.json` | 数据形状、来源、单位、文件引用和警告 |

遇到多个可解释的 sheet、变量、dataset、标签文件或样本方向时，skill 会要求最小必要确认，不会静默猜测。

### 2. `spectral-check`：光谱质量检查

读取标准数据包，先检查和标记问题；默认不删除样本或波段。只有用户确认动作、阈值和范围后，才进入 `clean` 模式。

**默认综合检查**

- 合同、文件、行列和波段轴一致性检查；
- 缺失值、非数值、常量波段和低方差波段；
- 类别样本量、类别不平衡和回归目标异常；
- 光谱强度、粗糙度、尖峰和基线漂移风险；
- 全局/类内平均光谱相似性；
- 完全重复和近重复光谱；
- PCA Hotelling T²、Q residual 和 PCA 空间 Mahalanobis 距离。

**支持的异常样本方法**

- 稳健 Z 分数（`robust_zscore` / robust Z-score）；
- 四分位距（`iqr` / interquartile range）；
- 中位数绝对偏差（`mad` / median absolute deviation）；
- 均值相似性（`similarity_to_mean`）；
- 类内相似性（`classwise_similarity`）；
- 尖峰检测（`spike_detection`）；
- 基线漂移评分（`baseline_drift_score`）；
- PCA Hotelling T²（`pca_hotelling_t2`）；
- Q 残差（`pca_q_residual`）；
- PCA-Mahalanobis（`mahalanobis_on_pca`）；
- 近重复检查（`near_duplicate_check`）；
- 多方法共识（`multi_method_consensus`）；
- 半采样异常稳定性（`half_resampling_outlier` / HR）；
- 蒙特卡洛交叉验证异常稳定性（`mccv_outlier` / MCCV）。

HR/MCCV 只在用户要求高级异常稳定性分析时运行；它们识别的是当前管线下的不稳定样本，不等同于自动判定“坏光谱”。

**经确认后支持的清理动作**

- 删除高置信异常样本；
- 删除完全重复或近重复光谱，或输出分组建议交给 splitter；
- 删除常量、高缺失率、严重低方差或高尖峰频率波段；
- 波段均值/中位数插补、线性/最近邻插值；
- Hampel、移动中位数或局部 MAD 尖峰修复；
- 更新清理后的标准数据包和 `qc_cleaning_log.json`。

主要输出为 `qc_result.json`；清理后还会生成 `cleaned_package/`，并通过 `next_package_for_downstream` 指向正确的下游输入。

### 3. `spectral-splitter`：可复现数据划分

在不复制光谱矩阵的情况下生成样本归属和划分合同，供后续所有阶段复用。

| 中文名称 | 方法代码 / English name | 适用场景 |
| --- | --- | --- |
| 随机留出 | `random` / random holdout | 回归或无分层要求的普通留出 |
| 分层留出 | `stratified` / stratified holdout | 分类任务，尽量保持类别比例 |
| 预定义划分 | `predefined_split` / predefined split | 外部验证集或已有 split 列 |
| K 折交叉验证 | `kfold` / K-fold CV | 常规交叉验证 |
| 分层 K 折 | `stratified_kfold` / stratified K-fold CV | 小样本分类 |
| 留一法 | `leave_one_out` / LOOCV | 极小样本的显式需求 |
| 蒙特卡洛重复留出 | `monte_carlo_cv` / Monte Carlo CV | 重复随机评估 |
| 重复随机划分 | `repeated_random_split` / repeated random split | 重复 holdout |
| 分层蒙特卡洛 | `stratified_monte_carlo_cv` / stratified MCCV | 重复分类评估 |
| Kennard–Stone | `kennard_stone` / Kennard–Stone split | 基于 X 空间覆盖的代表性划分 |
| SPXY | `spxy` / SPXY split | 同时考虑 X 与连续 y 的回归划分 |
| Duplex | `duplex` / Duplex split | 代表性 train/test 构造 |
| 回归分箱分层 | `regression_stratified` / regression-binned split | 连续 y 分箱后分层 |
| y 分箱分层 | `y_binned_stratified` / y-binned split | 连续 y 的分位数分箱 |
| 分组划分 | `group` / group split | 同组样本不跨集合 |
| 分组防泄漏划分 | `group_aware` / group-aware split | 批次、受试者或重复测量隔离 |
| 分层分组划分 | `stratified_group` / stratified group split | 同时保持类别和组边界 |

输出：`split_indices.csv`、`split_contract.json` 和 `split_summary.json`。当前不宣称支持时间序列划分、按时间顺序划分或 nested CV。

### 4. `spectral-preprocess`：防泄漏光谱预处理

输入必须包含 `split_contract.json`。需要学习总体统计量的方法只在训练集拟合，然后应用到验证集和测试集；CV/重复留出时按 fold/repeat 重新拟合。

**散射与趋势校正**

- 无预处理（`none` / no preprocessing）；
- 标准正态变量校正（`snv` / Standard Normal Variate）；
- 多元散射校正（`msc` / Multiplicative Scatter Correction）；
- 去趋势（`detrend` / detrending）；
- SNV + 去趋势（`snv_detrend` / SNV plus detrending）。

**平滑与导数**

- Savitzky–Golay 平滑（`sg_smoothing`）；
- 一阶导数（`first_derivative`）；
- 二阶导数（`second_derivative`）；
- 移动平均（`moving_average`）；
- 高斯平滑（`gaussian_smoothing`）；
- 中值滤波（`median_filter`）。

**基线校正**

- 线性基线（`linear_baseline`）；
- 多项式基线（`polynomial_baseline`）；
- 橡皮筋基线（`rubberband_baseline`）；
- 非对称最小二乘基线（`als_baseline` / asymmetric least squares）。

**缩放与归一化**

- 均值中心化（`mean_centering`）；
- 标准化（`standardization`）；
- 最小–最大缩放（`minmax_scaling`）；
- 鲁棒缩放（`robust_scaling`）；
- Pareto 缩放（`pareto_scaling`）；
- L2 归一化（`l2_normalization`）；
- 面积归一化（`area_normalization`）；
- 最大绝对值归一化（`max_abs_normalization`）。

**物理转换与波段处理**

- 反射率转吸光度（`reflectance_to_absorbance`）；
- 透射率转吸光度（`transmittance_to_absorbance`）；
- 对数变换（`log_transform`）；
- 保留物理波段范围（`band_range_select`）；
- 移除物理波段范围（`remove_band_ranges`）。

MSC、中心化和缩放类方法属于 train-fit 方法。SG/导数、基线、吸光度转换和波段范围方法需要确认关键参数或物理含义。输出为新的标准包、`preprocess_state.json` 和 `preprocess_contract.json`。

### 5. `spectral-feature`：特征工程、降维与变量选择

每次执行一个特征方法。所有需要拟合的方法只用训练集；深度嵌入需额外确认训练预算和专属参数。

#### 传统特征与化学计量方法

| 中文名称 | 方法代码 / English name |
| --- | --- |
| 不做特征工程 | `none` / no feature engineering |
| 主成分分析 | `pca` / Principal Component Analysis |
| PLS 潜变量 | `pls_latent_variables` / PLS latent variables |
| VIP 变量重要性 | `vip` / Variable Importance in Projection |
| KBest 统计筛选 | `select_k_best` / SelectKBest |
| SPA 连续投影算法 | `spa` / Successive Projections Algorithm |
| CARS 竞争性自适应重加权采样 | `cars` / Competitive Adaptive Reweighted Sampling |
| UVE 无信息变量剔除 | `uve` / Uninformative Variable Elimination |
| MCUVE 蒙特卡洛 UVE | `mcuve` / Monte Carlo UVE |
| 区间 PLS | `interval_pls` / interval PLS |
| 相关性筛选 | `correlation_filter` / correlation filter |
| ANOVA F 筛选 | `anova_f` / ANOVA F-test |
| 回归 F 筛选 | `f_regression` / F-regression |
| 方差阈值 | `variance_threshold` / variance threshold |
| 指定波段范围 | `select_by_band_range` / select by band range |
| 指定波段索引 | `select_by_band_indices` / select by band indices |

#### 投影、信号变换与流形方法

| 中文名称 | 方法代码 / English name | 说明 |
| --- | --- | --- |
| 核 PCA | `kernel_pca` / Kernel PCA | 非线性投影 |
| 稀疏 PCA | `sparse_pca` / Sparse PCA | 稀疏载荷 |
| 非负矩阵分解 | `nmf` / NMF | 输入必须非负 |
| 独立成分分析 | `ica_embedding` / ICA | 独立成分投影 |
| LDA 监督投影 | `lda_projection` / LDA projection | 分类监督降维 |
| DCT 特征 | `dct_features` / Discrete Cosine Transform | 确定性逐样本变换 |
| FFT 特征 | `fft_features` / Fast Fourier Transform | 确定性频域特征 |
| 字典学习 | `dictionary_learning` / Dictionary Learning | 稀疏字典表示 |
| Isomap 嵌入 | `isomap_embedding` / Isomap | 默认探索用途；建模需确认 |
| LLE 嵌入 | `lle_embedding` / Locally Linear Embedding | 默认探索用途；建模需确认 |
| t-SNE 嵌入 | `tsne_embedding` / t-SNE | 仅用于可视化，不进入性能证明 |
| UMAP 嵌入 | `umap_embedding` / UMAP | 默认探索用途；建模需确认 |

#### 深度光谱嵌入

| 中文名称 | 方法代码 / English name | 方法专属确认 |
| --- | --- | --- |
| 自编码器嵌入 | `autoencoder_embedding` / Autoencoder embedding | 维度、训练轮数等 |
| 去噪自编码器嵌入 | `denoising_autoencoder_embedding` / Denoising AE | `noise_std` |
| 一维 CNN 光谱嵌入 | `cnn_1d_embedding` / 1D CNN spectral embedding | 网络预算与正则化 |
| ResNet1D 光谱嵌入 | `resnet1d_embedding` / ResNet1D embedding | 小样本过拟合风险 |
| CLS-former 光谱嵌入 | `cls_former_embedding` / CLS-former embedding | `patch_size` |
| 掩码光谱自编码器嵌入 | `masked_spectral_autoencoder_embedding` / Masked spectral AE | `mask_ratio`, `patch_size` |
| 对比光谱嵌入 | `contrastive_spectral_embedding` / Contrastive spectral embedding | `temperature` 和增强强度 |

深度方法会确认 `n_components`、`epochs`、`batch_size`、`learning_rate`、`weight_decay`、随机种子、设备和方法专属参数。用于分类的嵌入建议比较 8/16/32 维；2 维通常只用于可视化。视觉分离不能代替分类或回归指标。

输出为新的标准包、`feature_state.json` 和 `feature_contract.json`，并保留选中波段、载荷、训练审计等可追溯信息。

### 6. `spectral-modeling`：分类、回归、模型比较与调参

输入可以是 split 后的标准包、`preprocess_contract.json` 或 `feature_contract.json`。模型只在训练集拟合，使用验证集或训练集内部 CV 选模型/参数，锁定后才访问测试集。

#### 分类模型

**传统机器学习与化学计量分类器**

- 逻辑回归（`logistic_regression` / Logistic Regression）；
- 线性 SVM（`linear_svm` / Linear SVM）；
- RBF 支持向量机（`svm` / RBF-SVM）；
- 线性判别分析（`lda` / Linear Discriminant Analysis）；
- 二次判别分析（`qda` / Quadratic Discriminant Analysis）；
- 高斯朴素贝叶斯（`gaussian_nb` / Gaussian Naive Bayes）；
- PLS-DA（`pls_da` / Partial Least Squares Discriminant Analysis）；
- SIMCA（`simca` / Soft Independent Modeling of Class Analogy）；
- K 近邻（`knn_classifier` / KNN classifier）；
- 随机森林（`random_forest_classifier` / Random Forest classifier）；
- Extra Trees（`extra_trees_classifier` / Extra Trees classifier）；
- 梯度提升（`gradient_boosting_classifier` / Gradient Boosting classifier）；
- 多层感知机（`mlp_classifier` / MLP classifier）。

**可选依赖 boosting 分类器**

- XGBoost（`xgboost_classifier`）；
- LightGBM（`lightgbm_classifier`）；
- CatBoost（`catboost_classifier`）。

**实验性小样本深度/概率分类器**

- 深核学习高斯过程分类器（`spectral_dkl_gp_classifier` / spectral DKL-GP classifier）；
- 原型光谱分类器（`proto_spectral_classifier` / prototypical spectral classifier）；
- CLS-former 分类器（`cls_former_classifier` / CLS-former classifier）；
- CLS-former 嵌入 + SVM（`cls_former_embedding_svm` / CLS-former embedding SVM）。

常用快速比较集 `regular-fast` 包含：逻辑回归、线性 SVM、RBF-SVM、LDA、KNN、随机森林和 Extra Trees。它是速度与覆盖面的默认折中，不包含 QDA、Gaussian NB、PLS-DA、SIMCA、Gradient Boosting、MLP、可选 boosting 或实验深度模型；这些方法仍可通过自定义列表或更完整的比较方案纳入。

#### 回归模型

**传统机器学习与化学计量回归器**

- PLS 回归（`plsr` / Partial Least Squares Regression）；
- 主成分回归（`pcr` / Principal Component Regression）；
- 线性回归（`linear_regression` / Linear Regression）；
- 岭回归（`ridge` / Ridge Regression）；
- Lasso（`lasso` / Lasso Regression）；
- 弹性网络（`elastic_net` / Elastic Net）；
- 贝叶斯岭回归（`bayesian_ridge` / Bayesian Ridge）；
- 支持向量回归（`svr` / Support Vector Regression）；
- K 近邻回归（`knn_regressor` / KNN regressor）；
- 随机森林回归（`random_forest_regressor`）；
- Extra Trees 回归（`extra_trees_regressor`）；
- 梯度提升回归（`gradient_boosting_regressor`）；
- 高斯过程回归（`gpr` / Gaussian Process Regression）。

**可选依赖回归器**：`xgboost_regressor`、`lightgbm_regressor`、`catboost_regressor`。

**实验性小样本回归器**：`spectral_dkl_gp_regressor`、`proto_spectral_regressor`、`cls_former_regressor`。

#### 自动调参和评估模式

- **固定参数**：使用用户给定或明确默认参数，适合快速基线和公平固定参数比较；
- **分类器/回归器内部调参**：只改变模型参数，使用验证集或训练集内部 CV；
- **validation-only**：只输出 train/validation 指标，不访问 test，适合 optimizer trial；
- **最终锁定测试**：模型和参数锁定后评估一次 test；
- **confirmatory test**：测试集已被查看时，明确标注为确认性结果而非完全盲测；
- **重复划分/CV**：按 fold/repeat 重拟合并汇总，不把多次结果误称为单次最终测试。

自动调参不会替用户自动改变上游预处理或特征方法；跨阶段调优属于 `spectral-optimizer`。实验模型需要显式确认训练预算和关键参数。

**主要输出**

- `modeling_contract.json`、`modeling_summary.json`、`metrics.json`；
- `predictions.csv`、`model_artifact.pkl`；
- 多分类器验证比较表 `classifier_validation_summary.csv`，逐行记录 train/validation accuracy、balanced accuracy、Macro-F1、AUC、参数和 `test_accessed`；
- 分类可选 `confusion_matrix.csv`；
- 不确定性模型可选 `prediction_std.csv`、`uncertainty_summary.json`；
- 多分类器验证比较输出每个模型的 train/validation 指标、参数和 `test_accessed` 状态。

### 7. `spectral-optimizer`：受预算约束的自动优化

optimizer 负责“规划候选、调用官方子 skill、比较验证/CV 结果、选择最佳管线”，不自行重写预处理、特征或模型算法。

**四种模式**

| 模式 | 功能 |
| --- | --- |
| `recommend_from_profile` | 只根据样本量、特征量、任务和类别结构推荐候选，不运行模型 |
| `tune_method` | 固定其他阶段，仅调一个方法的参数 |
| `compare_step` | 固定上下游，比较预处理、特征或模型中的一个阶段 |
| `optimize_pipeline` | 在确认的方法空间和 trial 预算内联合搜索多阶段管线 |

**三层调参强度**

- **Level 1：仅模型参数调优**。固定预处理和特征，只调分类器/回归器；
- **Level 2：传统特征 + 模型调优**。比较 PCA、PLS-LV、VIP、KBest、SPA 等参数与下游模型；
- **Level 3：深度嵌入 + 模型调优**。比较嵌入维度、训练参数和下游模型，计算预算大，必须先预览、裁剪并确认。

**安全规则**

- 必须明确候选方法、参数网格、选择指标和 `max_trials`；
- 所有 trial 使用 validation 或 train-only CV，禁止按 test 指标选方法；
- 平分时优先更简单预处理、更少输出特征、更少调参和更低计算成本；
- `best_pipeline.json` 保存 preprocess + feature + model 的完整 lineage；
- 最佳管线锁定后才可单独确认最终测试；
- 小 holdout 的高分建议用 5/10 次 repeated holdout 复核稳定性。

输出：`optimizer_contract.json`、`optimization_plan.json`、`candidate_space.json`、`trial_manifest.csv`、执行后可选 `trial_results.csv`、`best_pipeline.json` 和 `recommendation_report.md`。

### 8. `spectral-report`：论文级图表与可复现报告

从 reader、check、splitter、preprocess、feature、modeling 或 optimizer 的现有产物生成图表；不重新训练模型，不编造重复实验、误差条、显著性或单位。

**支持的图表类型**

- 原始/预处理光谱、均值与真实离散带；
- 波段选择、VIP、载荷、系数和潜变量解释图；
- PCA、UMAP、t-SNE、深度嵌入散点图；
- 分类器指标比较、重复实验箱线图/点范围图；
- 混淆矩阵、ROC、PR、校准曲线；
- 回归预测值–实测值、1:1 线、残差图和指标面板；
- optimizer 候选排名、trial landscape 和锁定管线性能；
- 现有图片的审计与重绘。

**默认论文风格**

- 白底、无网格、完整黑色全边框；
- Times New Roman；
- 低饱和、高区分度论文配色，不默认高饱和纯色或 hatch；
- 多面板统一使用外置小写标签 `(a)`, `(b)`, …；
- 柱状图从零开始，单次 holdout 不伪造误差条；
- 重复结果展示真实 folds/repeats 的分布、mean ± SD 或合适的不确定性；
- caption 说明数据分区、统计单位、重复定义和图形编码；
- embedding 坐标不能跨方法直接比较，视觉分离不能作为分类性能证据。

**图件包输出**

```text
report_contract.json
figures/*.svg
figures/*.pdf
figures/*.png
source_data/*.csv
code/*.py
captions/*.md
qa/*.md
```

每张图必须经过尺寸、裁切、字体、图例、全边框、网格、数据追溯和最终渲染 QA。

### 9. `spectral-workflow`：多阶段路由与合同编排

根据用户目标选择最小的 child skill 链，保存决策并传递合同。它负责流程，不重复实现各阶段算法。

**支持的典型路线**

- 只读取和质量检查；
- 推荐分类/回归基线；
- 手动逐阶段选择；
- 预处理、特征或分类器单阶段比较；
- 传统多阶段优化；
- 深度嵌入 + 传统模型；
- 实验性端到端小样本模型；
- 已有中间 contract 续跑；
- 从结果合同路由到论文绘图。

当用户只说“处理这个光谱数据”时，workflow 先只读检查数据结构，再给出路线选择，不会立即替用户运行任意模型。进入手动流程后，每个阶段应显示“推荐项 + 完整支持菜单 + 可执行选择示例”。

**合同链**

| 阶段 | 主要输入 | 主要合同/输出 |
| --- | --- | --- |
| Reader | 原始文件/文件夹 | `data_contract.json` |
| Check | 标准数据包 | `qc_result.json`，可选清理包 |
| Splitter | 标准数据包 | `split_contract.json` |
| Preprocess | 数据合同 + 划分合同 | `preprocess_contract.json` |
| Feature | 数据/预处理合同 + 划分合同 | `feature_contract.json` |
| Modeling | 数据/预处理/特征合同 + 划分合同 | `modeling_contract.json` |
| Optimizer | 已确认候选空间 + 上游合同 | `optimizer_contract.json`, `best_pipeline.json` |
| Report | 任一已存在结果合同 | `report_contract.json` + 图件包 |
| Workflow | 用户目标 + 各阶段合同 | `workflow_plan.json`, `workflow_result.json` |

## 安装

### Codex 插件市场

Before importing or enabling the plugin, validate the user-level Codex config:

```bash
python install/check_codex_config.py --json
```

If Codex reports `Could not load config.toml` or `unclosed table, expected ]`,
the failing file is `~/.codex/config.toml`, not the Spectral Skills runtime.
Plugin import makes Codex reload global config, so an old malformed
`[projects.'...']` entry can surface at that moment. Fix or remove the malformed
table, rerun the preflight, and then retry plugin import.

```bash
codex plugin marketplace add https://github.com/XY041216/spectral-skills.git --ref main
codex plugin add spectral-skills@spectral-skills-local-marketplace
```

If the Codex CLI cannot run on Windows, or if the marketplace/plugin entries
exist in `config.toml` but no `spectral-skills` folder appears under
`%USERPROFILE%\.codex\plugins\cache`, run the local installer from the clone
root:

```bash
python install/install_codex_plugin.py --json
```

It validates `config.toml`, writes a backup before changing it, enables the
local marketplace, and materializes the release image into Codex's plugin cache:

```text
%USERPROFILE%\.codex\plugins\cache\spectral-skills-local-marketplace\spectral-skills\<version>\
```

Restart Codex in a new thread after the installer reports success.

Codex Desktop 也可添加自定义插件市场：

- Marketplace source：`https://github.com/XY041216/spectral-skills.git`
- Branch/ref：`main`
- Plugin：`spectral-skills`

仓库根目录同时提供 `.codex-plugin/plugin.json` 和 `.mcp.json`，作为 GitHub
插件导入入口；该入口会把 Codex skills 指向已构建的
`plugins/spectral-skills/skills` 发布镜像，而不是根目录下的开发源 `skills/`。

不要使用 Codex 的 **GitHub skill installer** 直接从仓库导入子 skill。Spectral
Skills 是一个插件发布单元，不是一组可以独立安装的 skill 文件夹；如果导入日志里出现
`skill-installer` 或 `install-skill-from-github.py --repo XY041216/spectral-skills`，
应停止该流程并改用上面的插件市场安装命令。直接 skill 安装会丢失插件元数据、MCP
配置、共享运行时和发布检查，不是受支持的发行形态。

安装后开启新会话，例如：

```text
使用 $spectral-skills:spectral-workflow 处理 Tablet_ext_0-3.csv：
读取、质量检查、分层 6:2:2 划分、SNV、feature=none，并比较 SVM 与 PLS-DA。
```

### Claude-compatible agents

仓库在 `.claude-plugin/` 中提供 Claude-compatible plugin metadata：

```bash
claude plugin marketplace add XY041216/spectral-skills
claude plugin install spectral-skills@spectral-skills
```

不支持插件市场的智能体可以 clone 仓库，并将 `plugins/spectral-skills/` 与相应 `SKILL.md` 作为技能入口。`shared/`、`spectral_core/` 和 `scripts/` 必须与 skills 一同保留。

### 本地脚本运行

```bash
git clone https://github.com/XY041216/spectral-skills.git
cd spectral-skills
pip install -r requirements.txt
```

示例：

```bash
python plugins/spectral-skills/skills/spectral-workflow/scripts/run_spectral_workflow.py \
  --input path/to/data.csv \
  --output-dir outputs/workflow_demo \
  --task-goal classification \
  --split-ratio 6:2:2 \
  --preprocess-methods snv \
  --feature-method none \
  --models svm \
  --json
```

本地 marketplace 配置见 [`install.md`](install.md)。

## 可选依赖

基础依赖见 `requirements.txt`。以下路径需要对应本地依赖：

- UMAP：`umap-learn`；
- 深度嵌入和实验深度模型：PyTorch；
- XGBoost：`xgboost`；
- LightGBM：`lightgbm`；
- CatBoost：`catboost`。

缺少可选依赖时，相关方法会阻止执行并说明缺失项，不会静默替换成其他算法。

## 使用示例

```text
用 spectral-reader 把这个 Excel 光谱文件转成标准数据包，样本在行，标签在 Sheet2。
```

```text
对标准光谱包做非破坏性质量检查，标记尖峰、基线漂移、重复光谱和 PCA T²/Q 异常候选，不删除样本。
```

```text
使用 stratified_monte_carlo_cv 做 10 次 7:3 重复分类器比较，固定 SNV + PCA(10)，比较 regular-fast 模型并绘图。
```

```text
比较 PCA(10)、PLS-LV(3)、VIP(100)、KBest(80)、SPA(80)，使用验证集 Macro-F1 选择，不访问测试集。
```

```text
训练 8/16/32 维 contrastive spectral embedding，并在训练/验证内部调 SVM；锁定组合后再确认最终测试。
```

```text
根据 modeling_contract 生成分类比较图、混淆矩阵、source data、caption 和 figure QA。
```

## 仓库结构

```text
.agents/                         Codex marketplace metadata
.codex-plugin/                   GitHub plugin-import entrypoint for Codex
.mcp.json                        Root MCP entrypoint delegated to the plugin image
.claude-plugin/                  Claude-compatible metadata
install/                         plugin 构建与发布检查
plugins/spectral-skills/         发布用插件镜像（由构建脚本生成）
skills/                          开发源；不是 Codex 发布安装入口
spectral_core/                   共享运行时实现
scripts/                         统一 CLI/runtime 脚本
README.md                        面向用户的完整能力说明
install.md                       安装细节
```

开发源位于 `skills/` 和 `spectral_core/`；`plugins/spectral-skills/` 是生成的
Codex/Claude 发布镜像，不应手工编辑。发布前运行 `install/build_codex_plugin.py`
重新生成根插件入口和发布镜像，并用 `install/check_codex_plugin.py` 验证。

## 能力边界与科学解释

- 不静默删除样本、波段或标签；
- 不使用测试集选择预处理、特征、模型或超参数；
- 不启动未确认、无预算上限的 AutoML；
- 不把二维 embedding 的视觉分离当作分类性能；
- 不把单次 holdout 结果描述成 repeated holdout 或稳定性结论；
- 不编造误差条、显著性、置信区间、重复次数或波段单位；
- 不直接读取未导出的专有仪器二进制格式；应先导出为受支持的表格或容器格式。

## 开发与发行检查

在仓库根目录运行：

```powershell
python -m pytest -q
python install/build_codex_plugin.py --clean --verify --json
python install/check_codex_plugin.py --json
```

发布前不要提交缓存目录、临时输出、本地虚拟环境、调试档案或临时 QA 运行。修改源 skill 后，应重新构建插件镜像，确保开发源与公开插件一致。

## License

本项目使用 [MIT License](LICENSE)。

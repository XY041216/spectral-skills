# Method Selection

Supported methods:

When asking the user to confirm a split, show the recommended split first but
also show the full supported split-method menu. Do not hide advanced choices
behind the recommendation. Use these visible headings:

- `Recommended split`
- `Supported split methods`
- `When to choose another split`
- `You may choose`

- Every user-facing entry must be bilingual and include the executable method
  code, using `中文名称（method_code / English name）`. English-only split menus
  are invalid.
English-only split menus are invalid.
- 随机留出（`random` / random holdout）: shuffle samples with a fixed random seed and assign train/val/test.
- 分层留出（`stratified` / stratified holdout）: classification-only split that preserves class representation as much as possible.
- 预定义划分（`predefined_split` / predefined split）: use a user-specified split column, metadata split column,
  or external `split_indices.csv`.
- K 折交叉验证（`kfold` / K-fold cross-validation）: default `n_splits=5`, `shuffle=true`,
  `random_seed=42`.
- 分层 K 折交叉验证（`stratified_kfold` / stratified K-fold cross-validation）: classification K-fold preserving class representation.
- 留一法（`leave_one_out` / leave-one-out cross-validation）: LOOCV for small sample sets or explicit requests.
- 蒙特卡洛重复划分（`monte_carlo_cv` / Monte Carlo repeated holdout） / 重复随机划分（`repeated_random_split` / repeated random split）: repeated holdout, default
  `n_repeats=100`, `train_ratio=0.7`, `test_ratio=0.3`.
- 分层蒙特卡洛重复划分（`stratified_monte_carlo_cv` / stratified Monte Carlo repeated holdout）: classification repeated holdout with class
  stratification.
- Kennard-Stone 代表性划分（`kennard_stone` / Kennard-Stone split）: representative X-space holdout split.
- SPXY 代表性划分（`spxy` / SPXY split）: regression-only representative split using X and y distances.
- Duplex 代表性划分（`duplex` / Duplex split）: representative train/test construction.
- 回归分箱分层划分（`regression_stratified` / regression-binned stratified split） / y 分箱分层划分（`y_binned_stratified` / y-binned stratified split）: quantile-bin continuous y
  then split by bins.
- 分组划分（`group` / group split） / 分组防泄漏划分（`group_aware` / group-aware split）: group-aware splitting that keeps each metadata group in one split.
- 分层分组划分（`stratified_group` / stratified group split）: classification split that balances classes while keeping
  groups together.

If classification labels exist and no method is specified, recommend
`stratified` and request confirmation. If the user explicitly asks for `random`,
run random split and warn that stratification is usually safer for
classification.

Default recommendations:

- Classification holdout: `stratified`, ratio `6:2:2` or `8:2`, seed 42.
- Small classification data: `stratified_kfold` or repeated stratified split.
- Regression holdout: `kennard_stone` `8:2`; use `spxy` when y coverage is
  important and numeric y is present.
- Small regression data: `kfold`, `monte_carlo_cv`, or `leave_one_out`.
- Existing external validation: `predefined_split`.
- Group, batch, replicate, or QC replicate risk: `group_aware`, or
  `stratified_group` for classification.

Do not pretend to support time-series splitting, chronological splitting,
nested CV, or optimizer-specific validation schemes unless they are added later.

For KS/SPXY/Duplex, record distance metadata in `split_contract.json`: X metric,
X scaling, numeric dtype, tie-breaking seed/policy, and for SPXY also y metric,
y min-max scaling, and normalized-sum combination.

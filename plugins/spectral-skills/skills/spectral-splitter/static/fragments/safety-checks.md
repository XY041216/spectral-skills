# Safety Checks

Return `blocked` for invalid or incomplete standard packages.

Return `needs_confirmation` when the user must choose a ratio, confirm
classification stratification, change unsafe ratios, or requested an unsupported
method that should not be silently substituted.

Confirmation gates should remain simpler than QC, but ask before the agent
chooses an experimental-design-sensitive default:

- classification task with no method: recommend stratified and ask;
- classification task with user-requested random: warn about class imbalance;
- KS, SPXY, or Duplex: explain representative, non-random selection;
- SPXY or regression stratification: explain that y participates in splitting;
- group-aware or stratified-group: confirm group column or QC replicate groups;
- CV/LOOCV/MCCV instead of holdout: confirm folds/repeats and seed;
- small samples with 6:2:2 that make classes too small: block or recommend CV.

Do not write split outputs when any sample would be duplicated, omitted, or when
a requested non-empty split would contain zero samples.

Block when X contains NaN or infinite values.

For stratified splitting, every class must contain at least as many samples as
the number of requested non-empty splits.

For `predefined_split`, train and test must be non-empty and all samples must
have exactly one split assignment. Block duplicate assignment, unknown sample
ID, unknown split label, omitted sample, and out-of-range index.

For `group_aware` and `stratified_group`, no group may appear in more than one
split. Group integrity has priority over class balance; never split one group
to improve stratification.

For `spxy`, require regression task semantics and numeric y values; do not run
SPXY for classification.

For `regression_stratified`, use quantile bins and automatically reduce
effective bin count when bins would be too small for requested non-empty splits.

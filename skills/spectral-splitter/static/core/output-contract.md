# Output Contract

Write split results without copying spectral matrices.

Required output:

- `split_indices.csv`
- `split_contract.json`

Optional human-facing output:

- `split_summary.json`

For every split type, `split_indices.csv` must use one long-table schema:

`split_type,method,fold_id,repeat_id,role,sample_index,sample_id,label,group_id`

Leave `fold_id`, `repeat_id`, `label`, or `group_id` blank when they do not
apply.
`split_contract.json` must be compact, machine-readable, and sufficient for
downstream skills to combine with the original `data_contract.json`.

Use `split_type` to distinguish:

- `holdout`: write train/val/test `indices` and `sample_ids`.
- `cross_validation`: write `folds` with train/val indices.
- `repeated_holdout`: write `repeats` with train/val/test indices.

Include diagnostics when relevant: class distribution, fold size summary,
repeat size summary, regression target summary, group leakage check, and
X-space coverage. For KS/SPXY/Duplex, record `distance` metadata including
metric, scaling, combine rule when applicable, numeric dtype, random seed, and
tie-breaking policy.

Do not write `train_X.csv`, `test_X.csv`, or duplicate standard packages by
default.

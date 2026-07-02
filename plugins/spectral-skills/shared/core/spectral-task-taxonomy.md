# Spectral Task Taxonomy

Use this shared taxonomy unless a concrete skill provides a stricter extension.

## Classification

Classification predicts a discrete class label. Common label columns include
`label`, `class`, `category`, `type`, `group`, `origin`, `variety`, `species`,
`grade`, `类别`, `分类`, `等级`, `品种`, and `产地`.

Classification requires class labels, sample IDs, and a validation plan that
keeps leakage risks visible.

## Regression

Regression predicts a continuous target. Common target columns include
`target`, `y`, `value`, `concentration`, `content`, `amount`, `index`, `含量`,
`浓度`, `指标`, and analyte names.

Regression requires a numeric target, target units when available, and a
validation plan appropriate for small-sample/high-dimensional data.

## Unsupervised

Unsupervised tasks include exploration, clustering, anomaly screening, PCA
visualization, and quality inspection without labels. They may use metadata for
coloring or grouping, but metadata must not be silently treated as a supervised
target.

## Unknown

Use `unknown` when task evidence is incomplete or multiple interpretations are
possible. Preserve `unknown` until the user confirms the task. Do not infer a
supervised workflow only because a column has a label-like name.

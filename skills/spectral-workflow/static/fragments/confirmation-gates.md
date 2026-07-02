# Confirmation Gates

Ask only for the smallest missing decision. Do not execute a data-changing or
model-fitting step while its gate is open.

## Generic route gate

For “处理这个光谱数据”, inspect read-only metadata and show seven compact routes:
recommended baseline; read/QC only; manual conventional workflow; conventional
optimizer comparison; deep-learning model experiment; deep embedding plus
traditional classifier comparison; visualization-only exploration. Do not show
full stage menus until the user enters a stage.

## Stage-card contract

Use these headings: `推荐方案`, `为什么推荐`, `本轮默认纳入`,
`skills 还支持但本轮默认不纳入`, `需要额外确认才能执行`, `你可以选择`.
For modeling also show `自动调参能力`.

Every item must be bilingual with executable code:
`中文名称（method_code / English name）：说明`. Never use a code-only list,
“简版” or “等” in place of supported methods.

## Split gate

Confirm split type, design-specific ratios/folds/repeats/groups, shuffle, and
seed. Show the complete bilingual splitter menu. Do not ask for a holdout ratio
for K-fold/LOOCV.

## Preprocess gate

Confirm an explicit method, including `none`. Show the complete bilingual
preprocess menu and all method-shaping parameters. Train-fitted transforms use
train only and refit per fold/repeat.

## Feature gate

Confirm an explicit feature method, including `none`. Group the complete menu
as traditional/chemometric, projection/signal/manifold, and deep embeddings.
Deep methods require data-aware confirmation of dimensions, epochs, early
stopping status, batch, learning rate, weight decay, seed, device, and
method-specific parameters. Use 2D only for visualization unless the user
explicitly confirms a modeling experiment.

## Modeling gate

Group the complete menu as traditional/chemometric, optional boosting, and
small-sample deep/experimental models. List each deep model with Chinese name,
code, English name, and practical risk. A classifier set such as `regular-fast`
must show both included and excluded supported models.

Expose fixed defaults versus classifier-only tuning. State the exact grid,
selection metric, validation design, and final-test policy. Default
classification selection is Macro-F1. Test is never used for selection.

## Modeling completion gate

Before sending the final answer after a modeling stage, read the modeling
outputs. For multi-model comparisons, the final answer must include a Markdown
comparison table with every evaluated classifier, not just the selected model.
Minimum columns are Model, Train Macro-F1, Validation Macro-F1, Validation
Accuracy, selected parameters, and Test accessed. Explain why the winner was
selected and whether the test set remains isolated.

For `regular-fast`, explicitly list Logistic Regression, Linear SVM, RBF-SVM,
LDA, KNN, Random Forest, and Extra Trees in the final result summary.

## Optimizer gate

Show Level 1 classifier tuning, Level 2 traditional pipeline tuning, and Level
3 deep embedding/classifier tuning. Offer `quick`, `regular`, `extended`, and
`deep` budgets. Confirm candidates, grids, expanded trials, metric, repeats,
seed, device, and included/excluded methods. `deep` is opt-in only.

Before confirmation, use only `--preview-only`; write no candidate-space or
trial files. After selection, keep parameters locked for final test.

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

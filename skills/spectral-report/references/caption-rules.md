# Caption Rules

Write one self-contained caption per figure. Include:

1. the figure's evidence role and panel-by-panel content;
2. input layer and evaluated split;
3. sample count and class/target support where relevant;
4. statistical unit and number of folds/repeats/experiments;
5. uncertainty or interval definition;
6. metric scale and physical units;
7. statistical test, pairing, multiplicity correction, and effect definition when used;
8. optimizer selection rule and statement that test metrics were excluded;
9. confirmatory/previously accessed test status when applicable.

Use unambiguous result-scope names:

- `single validation split result`: one validation split; no repeated
  statistical units or error bars.
- `final locked test result`: final or confirmatory test evaluation after all
  parameters and pipeline choices are locked.
- `10 repeated held-out result`: ten repeated held-out evaluations; report the
  repeat as the statistical unit and use mean +/- SD when summarizing.

Avoid generic "test accuracy" or "held-out accuracy" when it could refer to
single validation, final test, or repeated held-out evaluation. Captions must
state the statistical unit and metric unit, for example percent, ratio, fold,
repeat, or sample.

Do not turn the panel title into a long conclusion. Put interpretation in the caption and manuscript text.

## Final Chat Summary

Every completed report should end with a concise interface summary, not only file paths. Include:

- one-sentence conclusion tied to the stated core claim;
- actual data/pipeline lineage, especially preprocessing and feature extraction;
- a paper-ready Markdown results table for model comparisons;
- an experiment-setting table when the design is not obvious;
- warnings that affect interpretation, such as QC warning samples marked but not removed, local supplemental scripts, convergence warnings, prior test access, slow confirmed model sets such as regular-full with Gradient Boosting, or single-split instability.

For repeated model comparisons, report values as `mean ± SD` and identify the statistical unit, for example `10 repeated holdouts`. If metrics were converted to percent, state that conversion. Include all compared classifiers in the ranked table unless the user requested a shortened table. A suitable interface table is:

| Rank | Classifier | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) | Notes |
|---:|---|---:|---:|---:|---|
| 1 | LDA | mean ± SD | mean ± SD | mean ± SD | Highest Macro-F1 |

Also include an experiment-setting table with dataset, sample count, class
count, repeated split design, train/test ratio, exact preprocessing/feature
pipeline, classifier set, primary metric, and result form. This table is part of
the final answer, not only a file artifact.

For boxplot captions, define the visual grammar: boxes are IQR, center lines
are medians, whiskers follow the chosen rule, raw points are individual
folds/repeats, and mean markers are explicitly named. If a nonzero metric axis
is used, state the plotted range and why it was chosen.
If the figure omits a legend for the boxplot layers, the caption must still
explain every layer, including orange raw repeat points and red/dark mean
markers when those encodings are used.

For bar chart captions, define that bars show the mean and error bars show SD
unless another interval was confirmed. If bar labels are displayed, state
whether labels show mean values or `mean ± SD`. For grouped classifier bars,
prefer concise mean labels and keep full `mean ± SD` values in the final
Markdown three-line table.
For single-holdout or single final-test bar charts, replace that default with:
bars show direct plotted metric values, no error bars are shown because there
are no repeated statistical units, and labels above bars show bar heights. Do
not imply uncertainty when none exists.
Bar value labels should normally be horizontal. If any label must be rotated,
the caption or QA must explain why a horizontal label was impossible and what
alternative was rejected; do not rotate short numeric labels merely to fit a
dense grouped bar chart.
If model or method names were abbreviated on the axis or in panel titles, the
caption or table note must expand them, for example `DAE = Denoising
autoencoder` or `CLS-F = CLS-former`.
If low-saturation fills, no hatches, or grayscale/hatch exceptions are used,
describe the visual grammar in the caption or QA so the figure does not imply a
different metric encoding.

For classifier line-chart requests, prefer a sorted point-range plot in the
caption and output unless the user explicitly reconfirms a line. If a line is
used, state: "The connecting line is a visual guide across classifiers ranked
by mean Macro-F1 and does not imply a continuous classifier variable."

For multi-method embedding scatter captions, state that embedding coordinates
are method-specific, unitless, and not directly comparable across methods. Also
state that visual separation is exploratory and is not classification
performance evidence. For deep embeddings, include or reference a `training audit table`
with method, epochs, batch size, learning rate, objective,
`supervised_y_used`, train-only status, fixed-epoch status, random seed, and
device. The final response must explicitly mention fixed-epoch training, no
claim of convergence, and small-sample risk when the training split is small.
If compact panel titles are used, expand abbreviated method names in the
caption or in the final result table note.

When upstream QC status is `warning`, include a plain-language note in the
final chat summary: "QC warning only marked candidate abnormal samples; no
samples were deleted or modified, and the results use the full dataset. A
sensitivity analysis excluding confirmed abnormal samples is recommended before
making a final manuscript claim."

After the result table, add one interpretive sentence tied to the actual
pipeline and uncertainty, for example: "Under SNV + PCA(95%), Linear SVM had
the highest mean Macro-F1, but the overlap of repeated-holdout distributions
and the SD values indicate noticeable small-sample split sensitivity." Replace
the pipeline and model names with contract-derived values.

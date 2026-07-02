# Classification and Model Comparison

Read `references/chart-decision-matrix.md`, `references/statistical-reporting.md`, `references/visual-system.md`, and `references/caption-rules.md`.

## Model Comparison

- One holdout value per model: use a horizontal Cleveland dot plot. A bar chart is allowed only for very few models and must start at zero.
- A small single-holdout or final-test summary with a few models and explicit manuscript panels may also use a zero-based bar chart when the user explicitly asks for bars. In that case, treat the bars as direct metric values, not distribution evidence, and do not invent error bars.
- Fold/repeat results: show all points, retain fold/repeat IDs, and add paired lines or repeat-level intervals when appropriate. For manuscript-style repeated-holdout classifier comparison with n >= 5 repeats, prefer boxplot or violin plus raw repeat points; state that the statistical unit is the repeat.
- Many datasets by many models: use a heatmap plus rank/difference view.
- Use a line only for ordered x variables such as training ratio, component count, top-k, or a hyperparameter.
- Default primary metric to the upstream selection metric, usually Macro-F1. Add accuracy, balanced accuracy, and AUC as secondary evidence when available.
- Heatmaps of mean metrics are summary panels only. If used, annotate mean +/- SD or provide a companion source table with SD; do not let a mean-only heatmap replace repeat-level distribution evidence.
- Use either ratio (0-1) or percent (0-100%) throughout one figure. For publication-style classifier comparison, prefer percent axes such as `Macro-F1 (%)` and record the conversion in source data and `report_contract.json`.
- For repeated performance boxplots, avoid both extremes: do not use a 0-100% axis when it makes the data unreadably compressed, and do not overzoom to the exact data range when it visually exaggerates differences. Use an automatic rounded-margin strategy and record the chosen range.

## Classifier Set Gate

If the user asks to "compare classifiers" without naming the classifiers, stop before modeling or plotting and ask for the candidate set. Offer:

- compact: `svm`, `linear_svm`, `pls_da`;
- regular-fast: `logistic_regression`, `linear_svm`, `svm`, `lda`, `knn_classifier`, `random_forest_classifier`, `extra_trees_classifier`;
- regular-full: `logistic_regression`, `linear_svm`, `svm`, `lda`, `knn_classifier`, `random_forest_classifier`, `extra_trees_classifier`, `gradient_boosting_classifier`;
- spectral modeling: `svm`, `linear_svm`, `pls_da`, `lda`, `qda`, `knn_classifier`, `random_forest_classifier`;
- user-specified: any explicit list supported by spectral-modeling.

Record the confirmed set in `classifier_set_source` and `candidate_classifiers`. If only three classifiers were already generated upstream, label the figure as a compact comparison rather than a complete classifier benchmark. If a local supplemental script is needed because official outputs contain only per-repeat winners, ask for permission before running it and mark `official_skill_used: false` for that supplemental result.
Warn that `gradient_boosting_classifier` can dominate runtime on wide spectra.
Do not include it in a generic "regular" comparison unless the user confirms
`regular-full`; recommend `regular-fast` for interactive runs.
Existing local scripts, cached outputs, or previously generated folders are not
confirmation of a classifier set. They may be mentioned as available evidence,
but the Agent must still ask the user to choose compact, regular-fast,
regular-full, spectral
modeling, or custom classifiers before any new repeated training or report
claim.

## Pipeline Semantics Gate

For classifier comparison reports, confirm and display the exact preprocessing
and feature pipeline before plotting or summarizing. The report contract must
include `feature_pipeline_statement`, for example:

- `preprocess=none; feature=PCA(explained_variance=0.95; fit=train_only)`;
- `preprocess=none; feature=PLS-LV(n_components=10; supervised; fit=train_only)`.

Do not let a result produced with `PCA(0.95 explained variance)` be described as
`none + PLS-LV(10)`. If the upstream contracts and the user request disagree,
block for clarification or state that the figure represents the contract
pipeline, not the earlier requested pipeline.

## Repeated Classifier Figure Layouts

For fixed preprocessing/feature pipelines such as `none + PLS-LV(10)` with repeated-holdout classifier outputs, use a confirmation card before plotting:

- fixed pipeline and repeated split strategy;
- candidate classifier set and locked parameters;
- primary and secondary metrics;
- plotting backend/tool: Python/Matplotlib-Seaborn supported default, or R/ggplot2 requested but unsupported in v1 unless the user confirms Python fallback;
- figure grammar: analysis dot plot, bar chart, boxplot + raw points, violin + raw points, mean +/- SD dot-range, heatmap summary, three-line table, or multipanel manuscript figure;
- language mode: English, Chinese, or bilingual;
- palette mode: reference-style pastel, cool-warm, grayscale, or custom;
- final size and ratio/percent metric scale.

The modeling/run confirmation is not a figure confirmation. If the user
confirmed `regular-full` modeling but did not explicitly choose a chart type,
plotting backend, language/font mode, palette, or panel layout, stop before
plotting and show a report confirmation card.

Recommended publication layout: panels for Accuracy, Macro-F1, and ROC-AUC, each showing classifiers on the x axis with boxplot + raw repeat points. Use paired repeat lines only if the same repeat IDs are present for every classifier and the lines remain readable. Put mean-only heatmaps in Extended Data or a summary panel.

If the user asks for a paper-style classifier comparison and repeated metrics
exist, present chart choices before plotting:

- A. manuscript boxplot + raw repeat points for the primary metric;
- B. three-panel boxplot layout for Accuracy, Balanced accuracy, and Macro-F1;
- C. boxplot plus paper-ready three-line table;
- D. boxplot + heatmap + confusion matrix multipanel;
- E. bar chart with real error bars, only when the user explicitly prefers it.

Default recommendation is option B or C for 10 repeated splits. Do not decide
the final chart grammar after seeing the metrics without recording
`chart_design_confirmation`.

## Bar Chart Rules

Use bar charts only when the user explicitly chooses them or when a small
single-metric summary is scientifically clearer than a distribution plot. For
classifier comparisons with repeated units, bars must encode `mean ± SD` and
must keep a zero baseline; if the performance range is narrow, recommend a
dot-range or boxplot instead of a bar chart.

Distinguish bar semantics explicitly:

- single holdout / single final-test snapshot: bars show direct metric values,
  with no error bars because there are no repeated statistical units;
- repeated holdout / CV / replicate summaries: bars show `mean ± SD` or the
  explicitly confirmed interval from real repeats.

If the user only says "draw a bar chart" after a classifier comparison, pause
for a compact report confirmation unless the metric and label style are already
explicit in the same request. Confirm:

- metric scope: Macro-F1, Accuracy, Balanced accuracy, or grouped metrics;
- label style: mean only, mean ± SD, or no value labels;
- plotting backend: Python/Matplotlib-Seaborn v1 default, or user-requested
  R/ggplot2 with a clear unsupported/fallback warning;
- palette mode: recommended ordinary paper high-distinction, soft paper,
  high-contrast, grayscale, colorblind-friendly, or custom. Do not present
  colorblind-friendly as the default;
- whether the final answer should include the full three-line result table.

Bar charts should support value labels:

- label the mean above each bar by default, for example `85.8`;
- label `mean ± SD` only when there is enough horizontal space; grouped bars
  usually label the mean only;
- keep value labels horizontal (`bar_label_rotation=0`) by default. Numeric
  labels such as `61.1` or `85.8` should be upright, not vertical or rotated;
- use label font size at least 90% of the tick-label font size, normally
  9.5-10 pt when tick labels are 10-11 pt;
- set the label offset 2-3 pt above the error-bar cap, or the equivalent in
  data coordinates after checking the final render;
- keep a visible gap between the label and the error-bar cap;
- if grouped bars are too dense, do not solve it by rotating value labels.
  Instead increase figure width, label only the primary metric (usually
  Macro-F1), switch to horizontal bars, or leave full `mean ± SD` in the
  three-line result table;
- if labels overlap or more than about 12 bars make the plot crowded, switch
  to horizontal bars, abbreviate labels, or label only the primary metric;
- record `bar_value_labels`, label format, font size, and collision handling in
  `report_contract.json` and QA.

Bar charts must also follow manuscript-style panel polish:

- use white panels with no background gridlines and full black frames on every
  subplot; record `all_spines_visible=true` and `gridlines_present=false` in QA;
- use low-saturation high-distinction fills with black bar edges by default;
  avoid saturated pure blue/orange/purple and do not use hatch patterns unless
  grayscale or black-and-white printing is confirmed;
- place panel labels outside the plotting region at the upper-left
  (`panel_label_position=outside_upper_left`) rather than inside the data area;
- keep panel titles concise and centered; if the y axis or caption already
  defines percent units, titles may omit repeated `(%)` text to reduce crowding;
- when the same long model labels repeat across panels, abbreviate labels
  before increasing rotation. Record the mapping in `display_name_map` and
  define the full names in the caption or result table note. Typical compact
  forms include `DAE`, `CNN1D`, `ResNet1D`, `CLS-F`, `Masked AE`, and
  `Contrastive`;
- prefer about 25-35 degrees of tick rotation after abbreviation. Do not solve
  dense repeated labels by rotation alone; widen the figure first, then switch
  to grouped bars or horizontal bars only if abbreviation plus width still fail.

Bar captions must say either:

- single-holdout/final-test case: bars show direct plotted metric values, no
  error bars are shown because there are no repeated units, and labels above
  bars show bar heights; or
- repeated case: bars show means across repeats, error bars show SD (or the
  confirmed interval), and labels above bars show the mean unless mean ± SD
  was explicitly confirmed.

## Line and Ranked Dot Rules

Classifier names are nominal, not continuous. Do not make a line chart the
default for classifier comparison. If the user asks for a line chart, first
explain that a sorted point-range plot is the recommended manuscript-safe
alternative. If the user still wants a line:

- order x by a recorded ranking such as mean Macro-F1;
- use a thin light gray connector or low-emphasis guide, never a strong trend
  line;
- keep SD/CI error bars on the points;
- write in the caption that the connector is only a visual guide across ranked
  classifiers and does not imply a continuous classifier variable.

If the user chooses a two-panel layout with a primary-metric boxplot plus a
mean-metric heatmap, the caption must state the asymmetry: panel A shows the
repeat-level distribution of the primary metric, while panel B is a compact
summary of mean +/- SD for secondary metrics. Prefer a three-line result table
over a heatmap for manuscript main text when the heatmap mainly repeats table
values.

Heatmaps are optional summary panels. If values occupy a narrow range such as
0.55-0.70, use a data-aware color scale or annotate `mean ± SD`; do not use a
0-1 colorbar that visually flattens differences unless the contract explicitly
requires a common ratio scale.

## Repeated Classifier Plot Polish

### Distinct method colors and ordering

- When x-axis categories are classifiers, feature methods, preprocess methods,
  or pipelines, assign one distinct low-saturation color to each method by
  default. Do not fill every method with the same blue unless the user confirms
  monochrome styling.
- Keep each method's color stable across Accuracy, Balanced accuracy, and
  Macro-F1 panels. This lets color encode method while panel encodes metric.
- Use black box/bar edges, black median/whisker lines, and either a shared pale
  orange raw-point layer or a darker shade of the method color. Use a dark red
  diamond for means only when that marker role is explained in the caption.
- When bars encode metrics within each method, use stable metric colors instead:
  Accuracy = low-saturation blue, Balanced accuracy = low-saturation orange,
  Macro-F1 = low-saturation green or purple.
- Do not default to a colorblind palette. Offer it as an explicit option and
  still run color/gray-scale QA.
- Sort methods by the confirmed primary statistic, normally mean Macro-F1 for
  classification. Keep the same order in every panel and record
  `method_order`, `sort_metric`, `sort_statistic`, and `sort_direction` in
  `report_contract.json` and the caption.
- Use stable short axis labels such as `LR`, `RBF-SVM`, `LDA`, `Linear SVM`,
  `ET`, `RF`, and `KNN`; define every abbreviation in the caption.

- For performance boxplots in percent units, compute y limits by
  `auto_y_limit=data_range_with_rounded_margin`, with `minimum_margin=5
  percentage points`, `round_to=5 or 10`, and `avoid_overzoom=true`. For
  example, data spanning about 45-75% should usually plot around 40-80% or
  45-85%, not 45-75% and not always 0-100%. State the nonzero axis range in
  the caption.
- Long classifier labels should be abbreviated on the axis, for example `LR`,
  `LDA`, `Linear SVM`, `RBF-SVM`, `ET`, `GB`, `RF`, and `KNN`. Define every
  abbreviation in the caption or table note.
- For three-panel classifier boxplots, keep x-axis labels around 30 degrees
  when possible; do not exceed 35 degrees unless the labels would otherwise
  collide. Prefer stable abbreviations over steeper rotation.
- Do not draw a lone panel letter `a` or `A` for a single-panel figure. Use
  panel letters only when at least two panels are present. For embedding and
  classification metric multipanels, use lowercase outside labels `(a)`,
  `(b)`, ... by default. Use one panel-letter style consistently across a
  report; do not mix `(a)(b)(c)` with `A/B/C`, and record the chosen style in
  `report_contract.json`.
- Boxplots must use full black frames on every axis and no background
  gridlines. Record `all_spines_visible=true` and `gridlines_present=false` in
  QA; a boxplot with only left and bottom axes visible must fail QA.
- Move long titles such as "Classifier accuracy across 10 stratified repeated
  holdouts" to the caption. The panel itself should keep only axis labels,
  panel letters, and concise condition labels.
- Explain boxplot elements in the caption: boxes indicate IQR, center lines
  medians, whiskers the chosen rule, orange points individual repeat results,
  and red/dark diamonds or short bars means.
- If no legend is shown for boxplot elements, the caption must explicitly
  define every visible layer: box/IQR, median line, whiskers, raw repeat points,
  and mean marker. Do not deliver an unlabeled orange-point/red-diamond
  boxplot without that caption text.
- Use the proper `±` symbol in figure annotations and captions. Avoid `+/-` in
  plotted cells or final manuscript text except when the output format cannot
  encode Unicode; if fallback is needed, record it in QA.
- Preferred reference-style palette: distinct low-saturation method fills,
  pale orange raw repeat points with thin dark edges (or darker same-method
  points), black median lines, dark red mean markers, black whiskers, full black
  frames, and no background gridlines. Record `method_palette`, `palette_roles`,
  and method-color mapping; run color/gray-scale QA.
- Increase font sizes by 1-2 pt when rendered previews show cramped or
  undersized axis tick labels. At final manuscript size, keep tick labels
  readable (usually at least 7 pt), axis labels one level larger, and panel
  letters clearly above the data layer. Record the final physical size and
  font QA result.
- For manuscript-working PNG previews and double-column classifier figures,
  target tick labels at 10-11 pt, axis titles at 12-13 pt, and panel titles or
  panel letters at 13-14 pt before downscaling. If the exported SVG/PDF is
  intended for final journal dimensions, also verify the scaled final-size
  minimums from `static/core/typography-export.md`.

## Paper-Ready Result Summary

After any classifier comparison, include a ranked Markdown table directly in the final response. Do not make the user open CSVs for the main result. Use a three-line-table style in Markdown:

| Rank | Classifier | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) | Notes |
|---:|---|---:|---:|---:|---|
| 1 | LDA | mean +/- SD | mean +/- SD | mean +/- SD | Highest Macro-F1 |

Use `mean ± SD` for repeated holdout/CV and state `n` and statistical unit. Also include a compact experiment-setting table:

| Item | Setting |
|---|---|
| Dataset | input file or contract dataset name |
| Samples/classes | n samples, n classes |
| Split/repetition | e.g. 10 stratified repeated holdouts |
| Pipeline | actual preprocess + feature pipeline from contracts |
| Classifier set | confirmed candidate set |
| Primary metric | Macro-F1 |

If the workflow used `standardization + PCA(0.95 explained variance)`, do not summarize it as `none + PLS-LV(10)`. Resolve pipeline lineage from contracts before writing the table.

For interface summaries, include all classifiers in the ranked table, not only
the top five, unless the user asked for a shortened abstract. Use percent units
for publication-style tables and include mean +/- SD for Accuracy, Balanced
accuracy, and Macro-F1. Add a short result interpretation after the table, for
example: "Under SNV + PCA(95%), Linear SVM had the highest mean Macro-F1, but
the overlap of repeat-level distributions and the SD values indicate noticeable
small-sample split sensitivity." Adapt the model names and pipeline to the
actual contracts.

Always include QC interpretation when upstream QC status is warning: "QC
warning only marked candidate abnormal samples; no samples were deleted or
modified, and the results use the full dataset. A sensitivity analysis excluding
confirmed abnormal samples is recommended before making a final manuscript
claim."

If the confirmed classifier set is `regular-full`, include an interpretation
note that it contains slower models such as Gradient Boosting. Preserve this in
`report_contract.json`, `figure_qa.md`, and the final chat summary even when
the current run completed quickly.

## Diagnostics

- Label confusion matrices as count, row-normalized percent, or column-normalized percent and show class support.
- For repeated-split aggregated confusion matrices, state that counts aggregate
  repeated test appearances and the same sample may appear in more than one
  split; they are not independent external-sample counts.
- Draw ROC/PR only from stored probability or decision scores. For multiclass, state one-vs-rest plus macro/micro averaging.
- Prefer PR when class imbalance is material and show prevalence/chance baseline.
- For calibration, report bin count and sample count with the reliability curve and ideal line.
- Keep final-test diagnostics separate from validation/CV candidate comparison.

---
name: spectral-report
description: >-
  Use when Codex needs to create, revise, audit, or package publication-grade figures and figure-centered reports from standard spectral-reader, QC, splitter, preprocess, feature, modeling, or optimizer outputs. Create task-specific Python plotting code, editable SVG/PDF, PNG previews, source data, captions, report contracts, and figure QA for spectra, preprocessing, selected bands, latent variables, classification, regression, model comparisons, and optimizer results. Trigger for paper figures, confusion matrices, ROC/PR/calibration, predicted-vs-measured plots, VIP/loadings, 光谱报告、论文绘图、科研绘图、模型精度对比、分类结果图、回归结果图、光谱图、预处理对比图、特征波段图. Do not use it to read raw vendor files, perform QC, split or transform data, fit features, train or optimize models, select by test metrics, or invent repetitions, units, statistics, or source data.
---

# Spectral Report

Create evidence-led spectral figures from existing spectral workflow artifacts. Version 1 is Python-first and supports Python/Matplotlib-Seaborn plotting. If the user asks for R/ggplot2, return unsupported in v1 or ask whether to use Python instead; never silently redraw an R-requested figure with Python.

## Required Loading

Read these files before designing any figure:

1. `static/core/boundary.md`
2. `static/core/figure-contract.md`
3. `static/core/integrity-rules.md`
4. `static/core/typography-export.md`
5. `static/core/qa-loop.md`
6. `references/input-artifact-map.md`

Then read only the relevant task fragment:

- raw/QC/preprocess spectra: `static/fragments/spectra-preprocess.md`
- feature selection or latent variables: `static/fragments/feature-interpretation.md`
- classification/model comparison: `static/fragments/classification.md`
- regression: `static/fragments/regression.md`
- optimizer trials: `static/fragments/optimizer.md`
- audit/redraw of an existing figure: `static/fragments/revise-existing-figure.md`

Read only the references named by the chosen fragment. After establishing the figure contract, consult `references/style-reference-index.md` and inspect at most three relevant files in `assets/style-references/`.

## Workflow

1. Identify the highest relevant input layer and verify its upstream lineage.
2. Determine `task_type`, `evaluation_scope`, test isolation, statistical unit, available repetition, band-axis semantics, and target units.
3. If the user asks to compare classifiers, features, preprocessing, or pipelines but the candidate set is not explicit, stop and show a confirmation card before any upstream run or plot design.
4. If the user asks for "repeated training/test plus a figure", route the upstream computation to workflow/modeling/optimizer first; spectral-report itself must not create model configs, call modeling scripts, or decide candidates.
5. Block only when missing information would change scientific meaning. Use defaults only for decorative choices; chart grammar, language/font mode, palette mode, metric scale, and panel layout are scientific presentation choices for publication figures, not mere decoration.
6. Write or preview the figure contract before plotting. When plot backend, chart type, target layout, language/font mode, palette, panel-letter style, metric scale, or y-axis strategy is ambiguous, ask the user to confirm the figure grammar and role. A user confirmation for modeling or workflow execution is not confirmation of the final figure design.
7. Extract only plotted values into `source_data/`; never hard-code final scientific values in plotting code.
8. Write task-specific Python in `code/` and render editable SVG, PDF, and PNG preview into `figures/`. Add TIFF only when requested.
9. Write a self-contained caption in `captions/`.
10. Run data/statistical QA and final-size visual/export QA. Revise and rerender until the last review passes.
11. In the final chat response, include a concise paper-ready results summary. For model comparisons, include a Markdown three-line-table-style result table and an experiment-setting table; do not require a separate file unless the user asks.

## Input Gates

Confirm or derive from contracts:

- classification, regression, or multi-target regression;
- train, validation/CV, repeated holdout, or final test scope;
- whether model parameters are locked and whether test was accessed previously;
- whether real fold/repeat/bootstrap/independent replication supports uncertainty or distributions;
- band type, direction, and unit;
- target or signal meaning and unit.
- candidate classifier/model set when a comparison request says only "compare classifiers" or similar;
- intended figure grammar/layout when repeated results could be shown as dot plot, box/violin, heatmap, or manuscript multipanel figure.
- plotting backend/tool: Python/Matplotlib-Seaborn is the supported v1 default; R/ggplot2 is unsupported in v1 unless a future R fragment is added. If the user asks for R style, confirm Python fallback before plotting.
- language mode: English, Chinese, or bilingual. English/numeric text uses Times New Roman; Chinese uses SimSun/Songti and English/numeric fallback remains Times New Roman.
- palette mode: recommended ordinary paper high-distinction colors, soft paper
  colors, high-contrast colors, grayscale submission, colorblind-friendly, or
  user-specified colors. Do not recommend colorblind-friendly as the default
  unless the user, journal, or QA explicitly requires it.
- y-axis/range strategy for performance figures when not using a zero baseline; record rounded margins and avoid overzoom.

Do not silently infer physical units, uncertainty, significance, or repetitions. Do not use candidate test metrics in optimizer figures. Do not turn a single holdout value into a boxplot or error bar.

## Output Package

Write one reproducible package:

```text
spectral-report-output/
  report_contract.json
  figures/fig_01_<slug>.svg
  figures/fig_01_<slug>.pdf
  figures/fig_01_<slug>.png
  source_data/fig_01_<slug>.csv
  code/fig_01_<slug>.py
  captions/fig_01_<slug>.md
  qa/figure_qa.md
```

Use SVG as the primary editable figure, PDF for submission/layout, and PNG for review. Preserve editable text in SVG/PDF. Record every export in the report contract.

## Non-Negotiable Rules

- Never retrain, tune, preprocess, select features, or alter locked models.
- Never choose a classifier/model collection for the user. Ask for compact, full, or user-specified candidates when the set is missing.
- Never write model-config JSON, candidate-space JSON, or trial-input JSON; those belong to spectral-modeling or spectral-optimizer.
- Never choose a method from final-test performance.
- Never fabricate values, units, intervals, replicates, sample sizes, or significance.
- Never connect nominal model names with a line.
- Never use a truncated bar axis; use a dot-range plot for narrow performance differences.
- Never treat t-SNE/UMAP separation as model-performance evidence.
- Never finish without rendering at final physical size and recording QA evidence.
- Never finish a model-comparison report with only prose. Provide a ranked results table with mean +/- SD where repeated units exist, and clearly state the actual preprocessing/feature/modeling pipeline used.
- Never silently choose a plotting backend when the user asks generically for a figure. Record `plot_backend`, backend support status, and backend confirmation in `report_contract.json` and QA.
- Never enable background gridlines by default for scatter, boxplot, barplot,
  or lineplot. Record `report_style.grid=false`; if gridlines are visible,
  figure QA must fail unless the user explicitly confirmed a grid exception.
- Never deliver ordinary scatter, barplot, boxplot, lineplot, or multipanel
  model-comparison figures with only left/bottom axes visible. Use full black
  frames by default and record `all_spines_visible=true` plus
  `gridlines_present=false` in QA.
- Never use saturated pure color bars or hatch patterns as the default paper
  bar-chart style. Use low-saturation high-distinction fills with black edges;
  use hatches only for confirmed grayscale or black-and-white printing.
- Never mix panel-label styles within one report. Use lowercase outside labels
  for embedding and classification metric multipanels; omit panel labels for
  single-panel figures unless explicitly requested.
- Never deliver a publication figure whose default style omits any required
  element: white background, no grid, full black frames, low-saturation colors,
  outside panel labels for multipanels, Times New Roman for English/numeric
  text, and captions that state the statistical unit and metric unit.
- Never blur result scopes. Use `single validation split result`,
  `final locked test result`, and `10 repeated held-out result` when those
  designs apply, instead of generic "test accuracy" or "held-out accuracy".
- Never deliver a deep-embedding report without a training audit summary table
  and a caption note that embedding coordinates are method-specific,
  unitless, and not directly comparable across methods.
- Never render `+/-` in publication figures when `±` can be encoded.
- Never omit QC warning interpretation from the final summary when upstream QC marked risk samples.

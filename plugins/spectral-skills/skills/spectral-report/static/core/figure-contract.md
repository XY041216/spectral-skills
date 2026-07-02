# Figure Contract

Create `report_contract.json` before writing plotting code. Include:

- `core_claim`: one sentence the figure supports.
- `figure_role`: `discovery`, `comparison`, `validation`, `robustness`, or `interpretation`.
- `input_lineage`: every contract/result path and identifier used.
- `task_type`: classification, regression, or multi-target regression.
- `evaluation_scope`: train, validation, cross_validation, repeated_holdout, or final_test.
- `test_isolation_status`: untouched, confirmatory, previously_accessed, or unknown.
- `figure_archetype`: single_panel, quantitative_grid, spectral_plus_quant, or asymmetric_hero.
- `panel_map`: panel letter, scientific question, source data, chart grammar, and statistical unit.
- `plot_backend`: plotting backend and support status, e.g. `python_matplotlib_supported_v1`.
- `statistical_unit`: sample, fold, repeat, dataset, or independent_experiment.
- `uncertainty_definition`: none, SD, SE, 95% CI, IQR, or prediction_interval.
- `axis_semantics`: variable, transformation, unit, and ratio/percent scale.
- `language_mode`: English, Chinese, or bilingual, with font assignments.
- `palette_mode`: ordinary-paper-high-distinction, soft-paper, high-contrast,
  grayscale, colorblind-friendly, or custom. Do not use colorblind-friendly as
  the default recommendation.
- `palette_roles`: stable category/method-to-color and non-color encodings,
  with explicit saturation policy and hatch policy.
- `method_palette`: for categorical method comparisons, record one distinct
  low-saturation color per classifier/feature/preprocess/pipeline method. Keep
  the mapping stable across metric panels; monochrome requires confirmation.
- `method_order`: exact displayed order plus `sort_metric`, `sort_statistic`,
  and `sort_direction`. State the same sorting rule in the caption.
- `display_name_map`: concise panel-title, tick-label, and table labels plus
  their full expansions, for example `DAE -> Denoising autoencoder` or
  `CLS-F -> CLS-former`.
- `report_style`: include `grid=false` by default, full black frame/spine policy, panel
  label style, panel-label position, title-shortening/abbreviation policy,
  shared-axis/legend placement, whether a grid exception was explicitly
  confirmed, and whether panel-letter case is consistent across the report.
  The default figure style is fixed as white background, no grid, full black
  frames, low-saturation colors, outside panel labels, Times New Roman for
  English/numeric text, and captions that explain the statistical unit and
  metric unit.
- `chart_design_confirmation`: user-confirmed chart grammar, layout, metric scale, language, and palette.
- `bar_value_labels`: for bar charts, whether labels are shown, label format
  (`mean` or `mean ± SD`), font size rule, and collision handling.
- `feature_pipeline_statement`: exact preprocessing and feature pipeline shown
  in the figure/table, resolved from upstream contracts.
- `summary_table_plan`: whether the final chat response includes a full ranked
  three-line-table-style results table and an experiment-setting table.
- `training_audit_summary`: required for deep embedding reports; include
  method, epochs, batch size, learning rate, objective, supervised-y use,
  train-only status, fixed-epoch status, random seed, and device.
- `final_size_mm`: width and height.
- `exports`: SVG, PDF, PNG, and optional TIFF.
- `reviewer_risks`: leakage, pseudoreplication, class imbalance, axis truncation, calibration, and overfit risks.

Use precise result-scope wording in the contract and captions:
`single validation split result` for one validation split,
`final locked test result` for the locked model's final/confirmatory test
evaluation, and `10 repeated held-out result` for ten repeated held-out
evaluations. Do not use generic "test accuracy" or "held-out accuracy" unless
the split scope is explicitly named in the same sentence.

Every panel must answer one independent question. Remove or merge a panel if hiding it would not weaken the core claim.

## Confirmation Card Before Plotting

Show a concise design confirmation card before plotting when any item changes scientific interpretation:

- candidate classifier/model set is missing or only implied;
- plotting backend is not explicit in a generic figure request, or the user
  asked for R/ggplot2 while v1 supports only Python/Matplotlib-Seaborn;
- chart grammar is ambiguous between dot plot, box/violin, heatmap, or multipanel manuscript figure;
- repeated results are available and the user has not chosen analysis-summary versus publication-style layout;
- metric scale could be ratio or percent;
- language/font mode is not explicit for a publication figure;
- palette or reference-style preference is not explicit and will affect interpretation or print legibility;
- final-test or confirmatory-test status affects the figure claim.
- a multi-method embedding scatter plot is requested; confirm 3-column
  multipanel layout, or a 2 x 3 layout for six methods, plus full frames, no
  gridlines, outside panel labels, shared axis labels, legend placement, and
  whether concise titles/abbreviations will be used.
- a multi-panel bar chart repeats the same long model labels across panels;
  confirm whether labels will be abbreviated, whether titles will omit repeated
  unit text such as `(%)`, and whether widening the figure is preferred over a
  steeper tick rotation.
- palette is unspecified; recommend ordinary paper high-distinction colors and
  offer soft paper colors, high-contrast colors, grayscale, colorblind-friendly,
  and custom colors as explicit alternatives. Do not silently recommend
  colorblind-friendly as the default.
- categorical methods are compared; confirm distinct low-saturation method
  colors (recommended), monochrome, grayscale, or custom. Do not silently use
  one fill for every method.

For classifier comparisons, include `classifier_set_source` in the contract: `user_specified`, `user_confirmed_compact`, `user_confirmed_full`, `from_upstream_contract`, or `partial_existing_outputs`. Also include `candidate_classifiers`, locked parameter source, selected chart grammar, and whether heatmaps are summary-only.

The confirmation card should offer the plotting backend (`Python/Matplotlib-Seaborn`, supported v1 default; `R/ggplot2`, unsupported in v1 unless the user confirms a Python fallback), chart choices such as boxplot + raw points, mean +/- SD dot-range, heatmap summary, three-line table only, bar chart with value labels, or multipanel manuscript figure. It should also offer language choices (English, Chinese, bilingual) and palette choices (recommended ordinary paper high-distinction, soft paper, high-contrast, grayscale, colorblind-friendly, custom).

For repeated classifier comparison, the card must also show the confirmed or
pending classifier set and feature pipeline. Existing helper scripts and cached
outputs may be listed as available inputs, but cannot fill
`classifier_set_source` or `feature_pipeline_statement` without user
confirmation.

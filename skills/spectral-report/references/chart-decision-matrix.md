# Chart Decision Matrix

| Data structure | Preferred grammar | Reject or downgrade |
|---|---|---|
| one holdout metric per nominal model | horizontal Cleveland dots; zero-based bars only for very few models or an explicitly requested small manuscript-style summary | box/violin, invented error, connected model line |
| same folds/repeats across models | box/violin + all raw points when n >= 5; paired lines or repeat-level intervals when readable | hiding points, mean-only heatmap as main evidence, unpaired treatment |
| multiple datasets and models | heatmap plus rank/difference dot plot | dozens of bars |
| ordered ratio/components/top-k/window/trial | line with real interval/points | unordered categories connected by line |
| optimizer candidates | validation/CV dot plot or structured landscape | test-based winner selection |
| many spectra | low-alpha individual curves + dark mean and justified band | rainbow spaghetti |
| selected bands | translucent spans/rug/top ticks over spectrum | opaque thick vertical lines |
| confusion | count or explicitly normalized matrix with support | unlabeled normalization |
| classification scores | ROC/PR/calibration only from stored scores | reconstructed probabilities |
| regression | predicted vs measured + 1:1; residual companion | metric-only decorative bars |
| embedding | scatter labeled discovery/interpretation | performance claim |

## Bars, Dots, and Axes

- Bars encoding magnitude start at zero.
- For explicit single-holdout bar summaries with a few models and two to three
  metrics, keep the zero baseline, omit fake uncertainty, and state in the
  caption that bars are direct metric values.
- For repeated metric bars, encode `mean ± SD` from real folds/repeats and
  show value labels by default. The default label is the mean, such as `85.8`;
  use `mean ± SD` labels only when there is enough space.
- Bar value labels must be visually subordinate but readable: use a font size
  at least 85% of axis tick labels, keep a minimum gap from the error-bar cap,
  and rerender if labels collide with error bars, legends, or panel borders.
- For crowded grouped bar charts, prefer horizontal bars, direct-label only the
  primary metric, or switch to dot-ranges/boxplots. Do not force long
  `mean ± SD` labels onto every bar when that harms readability.
- For metrics concentrated in a narrow range, use dots or dot-ranges and state the axis range.
- For repeated performance boxplots, use a rounded-margin y-axis strategy:
  compute the data min/max, add at least 5 percentage points of margin, round
  limits outward to 5 or 10 percentage points, and reject ranges that are too
  tight to the observations. A Macro-F1 spread near 45-75% should normally use
  about 40-80% or 45-85%, not 45-75% and not automatically 0-100%.
- Keep one 0-1 or 0-100% convention throughout a figure and record conversions.
- Use confidence bands with alpha below 0.20 and only from real uncertainty.
- Use heatmaps for compact summaries, not as the only evidence for repeated model comparisons. If a heatmap is retained, show mean ± SD in text or provide paired source data for the underlying repeats. Prefer a paper table when the heatmap mainly repeats numeric means.
## Gridlines

Default `report_style.grid=false` for scatter, boxplot, barplot, and lineplot.
Use `ax.grid(False)` in task-specific plotting code. A grid may be added only
when the user explicitly confirms a grid-assisted quantitative reading style,
and QA must record that exception. Multi-method embedding scatter plots must
never use background gridlines.

For manuscript multipanel embedding scatters, bar charts, boxplots, lineplots,
and ordinary model-comparison panels, also use full black frames on every panel
and place panel labels outside the data region at the upper-left. Boxplots with
only left/bottom axes visible must fail QA.

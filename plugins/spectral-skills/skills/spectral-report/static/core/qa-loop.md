# QA Loop

Run both QA layers after every render.

## Data and Statistical QA

- Match every plotted value to source data.
- Verify split, metric, class order, units, scale, sample count, and uncertainty definition.
- Verify result-scope terminology: use `single validation split result`,
  `final locked test result`, or `10 repeated held-out result` as appropriate;
  do not pass QA if a single split, final test, and repeated holdout are all
  described with the same generic phrase.
- Verify captions state the statistical unit and metric unit.
- Verify optimizer plots exclude test metrics from selection.
- Verify distributions and error bars have real repeat sources.
- Verify paired data retain pairing identifiers.
- Verify colors, markers, line styles, and ordering remain stable across panels.

## Visual and Export QA

- Render SVG/PDF at final width/height and inspect cropping, overlap, legends, panel letters, minus signs, superscripts, and Chinese glyphs.
- Verify axis labels, units, tick precision, significant digits, line widths, markers, and error caps.
- Verify the default figure style unless an explicit exception is recorded:
  white background, no grid, full black frames, low-saturation colors, outside
  panel labels for multipanels, and Times New Roman for English/numeric text.
- Verify every plotted axis uses full black frames unless the contract records
  a specific frameless exception. Record `all_spines_visible=true`,
  `spine_color=black`, and `spine_linewidth=1.0` in QA.
- For bar charts, verify the zero baseline, value-label format, label font
  size, spacing from error bars, absence of label collisions, full black
  frames, outside panel labels, low-saturation default fills without hatches,
  and readable abbreviated x labels.
- For boxplots, verify full black frames, no background gridlines, readable raw
  repeat points, and low-saturation fills. Do not pass QA with only left/bottom
  axes visible.
- Same-fill multi-method plots: for categorical method comparisons, verify
  `method_palette` assigns distinct
  low-saturation fills to methods unless monochrome was confirmed. Verify the
  same method keeps the same color across metric panels, and that
  `method_order` matches the recorded `sort_metric` and caption. Same-fill
  multi-method plots without a monochrome exception must fail QA.
- Verify `visible_gridlines=false` for scatter, boxplot, barplot, and lineplot
  unless the report contract records an explicit grid exception. If background
  gridlines are visible without that exception, mark QA FAIL and rerender.
- Verify panel-label style is consistent across the report. Use lowercase
  outside labels `(a)`, `(b)`, ... for embedding multipanels and classification
  metric multipanels; omit panel labels for single-panel figures unless the
  user explicitly requests one.
- For multi-method embedding scatter figures, verify full black frames on every
  panel, 2 x 3 layout for six methods or 3-column compact layout when seven or
  more panels are present, shared x/y labels, lowercase outside panel labels,
  concise centered titles, and a legend outside the data panels, below the
  panel grid, or in an empty panel.
- Inspect color, color-vision-deficiency simulation, and grayscale; add marker
  or line-style encoding if identity is lost. Add hatches only for confirmed
  grayscale or black-and-white printing.
- Verify SVG text remains text and PDF has no unintended rasterization or font substitution.

Write `qa/figure_qa.md` with concise PASS/FAIL, evidence, correction, and final rerender status. Deliver only after the last render has no unresolved defects.

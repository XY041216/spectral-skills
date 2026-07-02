# QA Checklist

Record each item as PASS/FAIL with brief evidence and correction.

## Data and Statistics

- Source-data row/value match.
- Contract lineage and split match.
- Metric name and 0-1/percent scale match.
- Band and target units match.
- Statistical unit and n are explicit.
- Error/distribution has a real repetition source.
- Paired structure is retained.
- Optimizer selection excludes test.
- Best highlight matches `best_pipeline.json`.
- Class/model ordering and visual mapping are stable.

## Visual and Export

- Final-size SVG/PDF render has no clipping or overlap.
- Labels, units, panel letters, legends, minus signs, and superscripts are correct.
- Chinese and Latin fonts render without substitution.
- Lines, markers, caps, ticks, and significant digits are readable.
- `visible_gridlines=false` for scatter, boxplot, barplot, and lineplot unless
  an explicit grid exception is recorded.
- Every plotted axis records `all_spines_visible=true`, black spine color, and
  1.0 pt spine width unless a specific frameless exception is recorded.
- Boxplots use full black frames and no background gridlines; left/bottom-only
  axes fail QA.
- Multipanel bar charts record `all_spines_visible=true`,
  `gridlines_present=false`, `panel_label_position=outside_upper_left`,
  low-saturation fills without default hatches, readable label
  abbreviations/rotation, and horizontal numeric value labels.
- Multi-method embedding scatter panels use full black frames, shared axis
  labels, lowercase panel labels outside the data region, concise centered
  titles, and a separate legend/note panel or bottom legend when space allows.
- Color, CVD simulation, and grayscale preserve identity.
- Colorblind-friendly is not recorded as the default palette unless the user,
  journal, or QA explicitly required it.
- SVG text remains editable.
- PDF has no unexpected rasterization.
- PNG preview matches vector output.

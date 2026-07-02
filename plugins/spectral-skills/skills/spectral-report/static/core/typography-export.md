# Typography and Export

- Use Times New Roman for English, numbers, axis ticks, and mathematical text.
- Use SimSun for Chinese; allow Songti SC or STSong only as an explicitly recorded platform substitute.
- Ask for language mode before publication plotting when not obvious: English, Chinese, or bilingual.
- Use the exact font name `Times New Roman`; never write or configure `Times News Roman`.
- Detect fonts before plotting. If neither required font nor an accepted substitute exists, block and report the missing font.
- Keep SVG/PDF text editable. Do not convert text to paths.
- Default figure widths: 89 mm single column or 183 mm double column; default maximum height 170 mm.
- At final size use 6-7 pt labels, 8-9 pt bold lowercase panel letters, and never less than 5 pt.
- For manuscript-working PNG previews and double-column classifier comparison
  figures, use larger readable working sizes before downscaling: tick labels 10-11 pt,
  axis labels 12-13 pt, and panel titles/letters 13-14 pt. Then verify the
  final physical-size minimums above after export.
- For classifier boxplots, prefer x-axis label rotation around 30 degrees and
  avoid rotations above 35 degrees by abbreviating labels first.
- For bar charts, keep numeric value labels horizontal (`rotation=0`) whenever
  possible and at least 90% of tick-label size; do not use vertical labels for
  short values such as `61.1`.
- Use 0.5-0.8 pt main lines (allowed 0.25-1 pt) and 2.5-4.5 pt markers.
- Use white backgrounds. For scatter, boxplot, barplot, and lineplot, set
  `ax.grid(False)` by default and record `report_style.grid=false`. A visible
  grid is allowed only after explicit user confirmation for numeric reading.
- For general quantitative charts, remove top/right spines unless a fragment
  requires a full frame. For multi-method embedding scatter panels, keep full
  black frames on every subplot.
- Export SVG, PDF, and PNG. Add TIFF only for a stated journal requirement.
- Render vector outputs back to pixels at final physical size and inspect the rendered result.
- In `figure_qa.md`, record font language, English/numeric font, Chinese font or `not used`, SVG editable text status, PDF font embedding status, and whether DejaVu or another fallback appeared.
- For classifier-comparison reports, include explicit QA lines:
  `Font check: PASS`, `English/numeric font: Times New Roman`,
  `Chinese font: SimSun/not used`, `SVG text editable: PASS`,
  `PDF text not outlined: PASS`, and `No DejaVu fallback: PASS` when verified.

# Scenario Tests

Use these as forward-test acceptance cases.

1. Single holdout, five classifiers: choose a dot plot; create no box, fake error, or significance.
2. Ten-fold CV, five classifiers: show all fold values and pairing; use Macro-F1.
3. Repeated holdout: align box/violin and summary to repeat-level units.
4. Optimizer trials: plot validation/CV only and highlight the existing best pipeline.
5. Locked final classifier: label confusion, ROC/PR, and calibration as final test.
6. Regression: predicted/measured with 1:1, residuals, and unit-bearing RMSE.
7. Raw versus SNV/MSC/SG/derivative spectra: preserve band axis and signal semantics.
8. VIP/SPA/selected bands: map positions to wavelength/wavenumber.
9. Chinese panels: SimSun/Songti plus Times New Roman; no glyph loss in SVG/PDF.
10. Missing band unit, repetition source, or required font: block rather than guess.
11. Export at 89 mm and 183 mm: final-size rendering remains legible.
12. CVD/grayscale: retain category identity through non-color encodings.
13. Request says `none + PLS-LV(10), ten repeated runs, compare classifiers` without classifier names: block for classifier-set confirmation before any upstream run or report plot.
14. Repeated-holdout classifier comparison with ten repeats: offer manuscript layout with boxplot plus raw repeat points; keep mean-only heatmap as summary or Extended Data.
15. Mixed request asks to run repeated modeling and draw: route modeling to spectral-workflow/modeling first; spectral-report must not write model-config JSON or decide candidate classifiers.
16. Publication classifier figure request without chart style, language, or palette: ask for chart grammar, language/font mode, and palette before drawing.
17. Completed classifier comparison: final response includes a ranked Markdown results table and an experiment-setting table, including the actual pipeline from contracts.
18. Generic publication figure request without a backend: confirmation card states Python/Matplotlib-Seaborn is the supported v1 default and asks for confirmation before assuming it; R/ggplot2 requests are blocked or converted only after explicit Python fallback confirmation.
19. Bar chart request from repeated classifier metrics: bars start at zero, show mean ± SD error bars, include readable mean labels above bars, keep labels separated from error caps, and downgrade to horizontal bars or primary-metric labels only when crowded.

20. Seven deep embedding test-set scatters: use a 3 x 3 layout with seven data
panels, one legend panel, and one note/blank panel; no background gridlines;
full black frames; lowercase panel labels; shared `Embedding 1` / `Embedding
2` labels; caption states coordinates are unitless, method-specific, and not
directly comparable.
21. Deep embedding report: final response includes a training audit table
covering epochs, batch size, learning rate, objective, supervised-y use,
train-only status, fixed-epoch status, seed, and device; it states fixed-epoch
training, no convergence claim, and small-sample risk.
22. Six deep embedding test-set scatters: use a 2 x 3 layout with six data
panels, lowercase panel labels outside the data region, concise centered
titles, a shared legend below or outside the grid, no background gridlines,
full black frames, and a caption note that visual separation is exploratory
only.
23. Six-model three-panel test-set metric bars: use white panels with full
black frames, no background gridlines, outside panel labels, concise panel
titles, abbreviated x labels such as `DAE` / `CLS-F` / `Masked AE` when
needed, horizontal numeric value labels, and a caption that states bars are
direct test-set metric values with no error bars because there are no repeated units.

For representative forward tests, require the full package: contract, SVG/PDF/PNG, source CSV, Python code, caption, and QA record.

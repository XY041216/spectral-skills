# Spectra and Preprocessing

Read `references/chart-decision-matrix.md`, `references/axis-and-unit-rules.md`, `references/visual-system.md`, and `references/layout-recipes.md`.

## Raw or QC Spectra

- Plot many individual spectra as thin low-alpha lines; overlay class means with a darker line.
- Add SD or 95% CI bands only when their definition and statistical unit are recorded.
- Prefer a separate QC overview for flagged samples; do not label QC candidates as confirmed errors.
- Show sample/class counts from the contract rather than estimating from the plot.

## Preprocessing Comparison

- Align raw and transformed spectra on the same physical band axis.
- Use paired panels or restrained overlays for raw versus SNV/MSC/SG/detrend results.
- Label derivative order and resulting signal semantics. Do not call detector intensity absorbance without upstream evidence.
- Mark retained/removed band regions with transparent spans and document boundaries in source data.

## Axis Direction

- Use increasing wavelength by default only when the contract states wavelength.
- For wavenumber, follow the contract or user-confirmed convention; record any reversed axis.
- Block if band type or unit is missing or contradictory.


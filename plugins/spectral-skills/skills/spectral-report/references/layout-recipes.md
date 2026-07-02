# Layout Recipes

- `single_panel`: one claim and one chart; use for a focused comparison or diagnostic.
- `quantitative_grid`: aligned small multiples with shared axes and one legend strip.
- `spectral_plus_quant`: a dominant spectrum panel plus two or three compact quantitative panels.
- `asymmetric_hero`: one large validation/interpretation panel with smaller supporting evidence.

Use 89 mm width for single-column figures and 183 mm for double-column figures. Keep height at or below 170 mm unless the target journal states otherwise.

Give the core claim the largest panel. Do not force all panels to equal size.
Align panel letters at the upper-left outside the data region. Use lowercase
outside labels `(a)`, `(b)`, ... for embedding and classification metric
multipanels unless the user explicitly requests another style. Do not mix
uppercase and lowercase panel-label styles within one report. Omit panel labels
for single-panel figures unless explicitly requested. Use horizontal model
labels/dot plots when names are long. Share legends and axes where possible.
Evaluate spacing only after rendering at final size.

For multi-method 2D embedding scatter figures, prefer three columns for seven
or more panels. A seven-method deep-embedding figure should normally use a 3 x
3 layout: `(a)` Autoencoder, `(b)` Denoising AE, `(c)` CNN1D, `(d)` ResNet1D,
`(e)` CLS-former, `(f)` Masked spectral AE, `(g)` Contrastive spectral, plus
one legend cell and one note or blank cell. Use full black frames on every data
panel, no background gridlines, shared x/y labels, and a legend outside data
panels or in an empty panel.

For six-method deep-embedding figures, prefer a 2 x 3 layout with six data
panels, shared x/y labels, lowercase outside panel labels, concise centered
titles, and a shared legend below the grid or centered outside the panel area.

For three-panel metric bar figures that repeat the same model labels in each
panel, abbreviate labels before increasing rotation, keep the rotation near
30 degrees when possible, and widen the figure before accepting crowded text.

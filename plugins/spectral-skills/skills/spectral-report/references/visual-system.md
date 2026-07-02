# Visual System

## Publication Pairing

- `control_blue`: `#5B9AE9`
- `control_blue_soft`: `#97B3D8`
- `treatment_peach`: `#E5B58E`
- `treatment_soft`: `#E8D1BA`
- `support_pink`: `#DBB4C6`
- `support_sage`: `#A9B98A`
- `neutral_dark`: `#333333`
- `neutral_mid`: `#777777`
- `neutral_light`: `#D9D9D9`

Use for control/treatment, baseline/method, or raw/processed comparisons. Pair low-saturation fills with darker lines or points.

## Default Palette Policy

Do not make colorblind-friendly palettes the default recommendation. The
default confirmation-card wording should be:

- recommended: ordinary paper high-distinction colors;
- optional: soft paper colors, high-contrast colors, grayscale, colorblind-friendly, or user-defined colors.

Use colorblind-friendly palettes only when the user requests them, when the
target journal or accessibility requirement demands them, or when QA shows the
ordinary palette fails identity checks. Still run CVD/grayscale QA for every
publication figure.

For paper bar charts, prefer low-saturation high-distinction fills with black
edges. Do not use saturated pure blue/orange/purple as the default. Do not use hatch patterns by default; reserve hatches for confirmed grayscale or black-and-white printing.

Recommended low-saturation bar fills:

- soft blue `#AFC6E8`
- soft orange `#E8C09A`
- soft sage `#B8C59B`
- soft purple `#D9BFD0`
- soft cyan `#B9D8DC`
- soft sand `#E3D2B8`

## High-Identity Methods or Classes

- blue `#2454E6`
- orange `#FF7A00`
- green `#137F22`
- purple `#A514C6`
- ochre `#A8780A`
- cyan `#13A8D3`
- rose `#D85B8A`
- olive `#789A3D`

Use no more than five strong colors in one main view. For additional groups, use facets, lightness, marker, line style, hatch, or direct labels. Reserve red/green for decrease/improvement when that semantic distinction is important. Record mappings in `report_contract.json` and keep them stable across panels.
Use these strong colors for explicit high-contrast requests or dense class
identity, not for default paper bar fills.

Pass color-vision-deficiency and grayscale review. Color must not be the only identity channel.
For dense four-class embedding panels, a color-only encoding is acceptable only
when the user explicitly confirms it, the palette is high-contrast, and
CVD/grayscale QA still passes. Record that choice in `palette_roles` and QA.

## Palette Confirmation

Ask for palette mode before final publication plotting when the user has not specified it or provided a clear reference:

- ordinary-paper-high-distinction: low-saturation high-distinction fills or points with black outlines, recommended default;
- soft-paper: low-saturation blue/orange/sage/purple/cyan family, closest to reference-style paper figures;
- high-contrast: stronger class or method identity colors when separation is more important than softness;
- grayscale submission: white/gray fills plus black points, hatches, and line styles;
- colorblind-friendly: use only when requested or required, not as the default;
- custom: user-provided hex colors.

For repeated classifier boxplots, keep raw repeat points visually above the box fill. Prefer orange points (`#FF7A00` or `#E5B58E`), black median lines, and a distinct mean marker or short line. Do not use the same hue and similar lightness for both box fill and raw points.

Record `palette_roles` in `report_contract.json`, including box fill, raw points, median, mean, classifier/model identity, and non-color encodings. QA must include color-vision-deficiency and grayscale checks; if they fail, add marker, hatch, line style, or direct labels.

For reference-style classifier boxplots, the preferred publication mapping is:
low-saturation blue/cyan/purple box fills, orange raw repeat points, black
median line, and orange diamond or short red line for the mean. Avoid green
triangles for means unless the user explicitly selects that style, because green
may imply improvement/acceptance rather than a statistic. If eight classifiers
share one panel, use either one neutral box family with orange points or subtle
family grouping plus non-color encodings; do not rely on default Matplotlib
blue boxes, blue points, orange medians, and green means without confirmation.
In short: default Matplotlib styling and green triangles are not acceptable
publication defaults unless the user confirms that style.

## Grid and Frame Defaults

Use `report_style.grid=false` as the default publication style. For scatter,
boxplot, barplot, lineplot, confusion-style metric panels, and small multiples,
call `ax.grid(False)` on every axis. Do not leave Matplotlib/Seaborn default
background gridlines in final figures.

Use white panels with full black axis frames on every subplot by default. This
applies to embedding scatters, bar charts, boxplots, line plots, and ordinary
model-comparison panels. In QA language, record `all_spines_visible=true` and
`gridlines_present=false`:

```python
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color("black")
ax.grid(False)
```

Use faint grids only when the user explicitly chooses a grid-assisted
quantitative style, and record that exception in `report_contract.json` and QA.

Keep panel labels outside the data region at the upper-left and keep titles
centered and concise. Single-panel figures normally have no panel label.

For repeated classifier comparisons with many categories, keep model
identity legible through abbreviations plus table notes rather than strong
color alone. A restrained publication default is:

- box fills: low-saturation blue/cyan/purple/sage family at 0.55-0.70 alpha;
- raw repeat points: orange with a thin dark edge, alpha around 0.75;
- means: dark red diamond or short horizontal marker, visually distinct from
  raw repeat points;
- medians and whiskers: neutral dark/black;
- grid: none by default; use `#E6E6E6` only for an explicitly confirmed
  grid-assisted exception.

Do not use green mean triangles by default; green has semantic baggage
(`improvement`, `pass`, `safe`) and should not encode a summary statistic
unless the user explicitly selects it.

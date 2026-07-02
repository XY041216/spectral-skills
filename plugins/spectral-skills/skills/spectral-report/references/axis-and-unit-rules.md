# Axis and Unit Rules

Read band value/unit/type from `band_axis.csv` and `data_contract.json`. Do not infer physical units from numeric range alone.

| Meaning | Preferred English label |
|---|---|
| wavelength | `Wavelength (nm)` or the contract unit |
| wavenumber | `Wavenumber (cm^-1)` using mathematical superscript in rendered text |
| Raman shift | `Raman shift (cm^-1)` |
| absorbance | `Absorbance (a.u.)` unless declared dimensionless |
| reflectance fraction | `Reflectance` |
| reflectance percent | `Reflectance (%)` |
| detector signal | `Intensity (a.u.)` |
| classification metric | `Macro-F1 (%)`, `Accuracy (%)`, `Balanced accuracy (%)` |
| regression error | `RMSE (<target unit>)`, `MAE (<target unit>)` |
| PCA score | `PC1 (explained variance, %)` when available |

- Put a space between values and units.
- Keep units upright and variables italic where mathematical typography permits.
- Use proper superscript in rendered output.
- State derivative order and denominator axis unit for derivative spectra.
- Do not mix fraction and percent within a figure.
- Shared axes may suppress repeated labels only when the full figure remains unambiguous.
- Avoid vague labels such as `Mean (%)`; name the exact metric or quantity.


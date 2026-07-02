# Chemometrics Background

Spectral modeling often involves high-dimensional, correlated variables and
limited sample sizes. These conditions make leakage control and reproducible
validation central to the workflow.

## Common Concepts

- High dimensionality: many wavelengths or wavenumbers relative to samples.
- Collinearity: neighboring bands often carry overlapping information.
- Small samples: metrics may vary strongly with split choice.
- Scatter: physical scattering can change baseline and multiplicative effects.
- Baseline drift: instrument and sample effects can shift spectra.
- Preprocessing: transformations such as smoothing, derivatives, scatter
  correction, and normalization.
- Feature selection: selecting wavelengths, latent variables, or constructed
  features.
- Classification and regression: supervised tasks requiring confirmed labels or
  numeric targets.
- Model validation: split design, cross-validation, and external validation
  determine how much to trust reported metrics.

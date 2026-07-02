# Spectral Modeling Pitfalls

Common pitfalls across spectral workflows include:

- performing supervised feature selection before splitting, causing leakage;
- fitting preprocessing parameters before splitting, causing leakage;
- trusting unstable metrics from small validation sets;
- overfitting high-dimensional spectra with too few samples;
- treating metadata columns as spectral variables;
- treating label or target columns as features;
- mixing external validation samples into training;
- reporting only the best result without recording the search process;
- changing sample or feature inclusion without recording confirmations;
- comparing models trained on different preprocessing or split protocols as if
  they were directly comparable.

Contracts should preserve warnings and execution records so reports can explain
these risks.

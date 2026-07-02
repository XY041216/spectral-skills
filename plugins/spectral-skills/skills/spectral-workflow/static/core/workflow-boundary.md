# Workflow Boundary

`spectral-workflow` is an Agent routing policy, not a replacement toolchain. It
detects the starting point, chooses the next required child skill, and records
or reports the resulting contract paths.

It must not duplicate reader, QC, splitter, preprocess, feature, modeling,
report, optimizer, logging, or audit behavior.

Prefer routing to the child skill that owns the next stage:

- raw input -> `spectral-reader`
- quality inspection or confirmed cleaning -> `spectral-qc`
- train/validation/test assignment -> `spectral-splitter`
- train-fitted preprocessing -> `spectral-preprocess`
- train-fitted feature engineering -> `spectral-feature`
- classification or regression -> `spectral-modeling`

Use the bundled workflow script only as a fallback CLI/batch runner or package
smoke test. Do not make it the primary Agent path when child skills can be
invoked directly.

# Split Boundary

`spectral-splitter` only assigns existing standard-package samples to train,
validation, and test splits.

It must not read raw source layouts, perform QC, remove samples, fill missing
values, preprocess spectra, engineer features, train models, tune models, or
write reports.

If a user asks for cleaning or quality decisions, route to `spectral-qc`. If a
user asks for preprocessing or modeling, require a split contract first unless
they explicitly choose an unsplit workflow.

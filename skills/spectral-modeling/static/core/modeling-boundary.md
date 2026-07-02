# Modeling Boundary

`spectral-modeling` starts only after a standard spectral package and
`split_contract.json` already exist.

It may consume reader/qc output directly after splitting, preprocess output, or
feature output. It must not read raw source files, repair datasets, remove
samples, split data, preprocess spectra, or select/derive features.

It writes model results and a modeling contract. It does not create a new
standard spectral package.

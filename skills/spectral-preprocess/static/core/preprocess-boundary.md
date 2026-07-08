# Preprocess Boundary

`spectral-preprocess` transforms spectra in an already split standard package.

It must not read raw source layouts, perform QC, delete samples, split data,
select features, train models, tune methods from validation/test performance, or
write reports/logs/debug archives.

Route cleaning requests to `spectral-check`, splitting requests to
`spectral-splitter`, feature selection or PCA requests to `spectral-feature`,
and modeling requests to `spectral-modeling`.

# Feature Boundary

`spectral-feature` transforms a standard spectral matrix into a feature matrix
after splitting and optional preprocessing.

It must not read raw source layouts, perform QC, delete samples, split data, run
spectral preprocessing, train models, evaluate models, optimize feature methods
from validation/test performance, or write reports/log/debug archives.

Route preprocessing requests such as SNV, MSC, SG smoothing, and derivatives to
`spectral-preprocess`. Route modeling requests to `spectral-modeling`.

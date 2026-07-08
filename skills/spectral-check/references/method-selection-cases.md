# Method Selection Cases

## Small Unlabeled Dataset

Prefer robust Z-score, IQR, MAD, missing-rate, constant-band, and low-variance
checks. Avoid covariance-heavy methods unless the user asks.

## Larger Unlabeled Dataset

Recommend MD or PCA_DISTANCE for sample candidate detection after missingness is
handled or imputed. PCA Hotelling T2 and Q residual are later extensions.

## Classification Dataset

Use class counts and class-aware outlier checks. Warn when classes are too small
for stable class-wise estimates.

## Regression Dataset

Use target IQR or MAD for target candidates. PLS residual and MCCV are later
extensions because they require modeling and repeated validation.

## User Specifies Method

Run the specified method when it fits the data. If the method is unsafe for the
data shape, explain why and suggest a safer non-destructive alternative. If the
method is not implemented, say so directly and do not automatically switch to a
different method.

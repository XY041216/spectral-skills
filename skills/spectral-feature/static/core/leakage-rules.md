# Leakage Rules

Fit feature extraction or feature selection rules only on train samples from
`split_contract.json`.

Validation and test spectra must not influence PCA components, variance
thresholds, selected feature counts, selected variables, projection matrices, or
threshold decisions.

Do not use validation/test labels for supervised selection. Fit
`pls_latent_variables`, `vip`, `correlation_filter`, `select_k_best`,
`interval_pls`, `cars`, `uve`, and `mcuve` from train X/y only. Refit them
independently for every fold or repeat.

If no split contract is available, train-fit feature methods require explicit
confirmation for exploratory non-modeling use.

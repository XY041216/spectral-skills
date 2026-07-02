# Leakage Rules

Fit preprocessing parameters only on train samples from `split_contract.json`.
Transform train, validation, and test samples with the same fitted parameters.

Train-fit methods include mean centering, standardization, and MSC reference
spectrum fitting.

Do not use validation or test spectra to estimate means, standard deviations,
MSC references, scaling ranges, or imputation values. Do not use test results to
choose preprocessing methods or parameters.

If no split contract is available, train-fit methods require explicit
confirmation for an unsupervised whole-dataset transform.

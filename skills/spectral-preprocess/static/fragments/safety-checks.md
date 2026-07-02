# Safety Checks

Return `blocked` for invalid standard packages, non-numeric X values, missing
required files, mismatched row counts, duplicate sample IDs, invalid split
indices, empty train splits, or transforms that would change sample count.

Return `needs_confirmation` when methods are missing, SG parameters are missing,
train-fit methods are requested without `split_contract.json`, baseline
correction is not confirmed, absorbance conversion is not confirmed, or
band-axis-changing methods are not confirmed.

Do not write outputs if any split sample is duplicated, omitted, or references a
sample ID/index outside the package.

Block absorbance and log transforms if any X value is non-positive. Block band
range selection/removal if every band would be removed.

The preprocess skill does not support sample removal or model-driven feature
selection. Physical band range selection/removal is allowed as preprocessing,
but it must update `X.csv`, `band_axis.csv`, `data_contract.json`, and any
fold/repeat role matrices consistently.

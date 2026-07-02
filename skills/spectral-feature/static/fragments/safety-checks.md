# Safety Checks

Return `blocked` for invalid standard packages, non-numeric X values, missing
required files, mismatched row counts, duplicate sample IDs, invalid split
indices, empty train splits, or transforms that would change sample count.

Return `needs_confirmation` when method is missing, task type is unknown for a
supervised method, band range/indices are missing, a score threshold selects no
features, critical method parameters are omitted, or train-fit methods are
requested without `split_contract.json`.

Do not describe silent defaults as user decisions. Record `parameter_sources`,
`defaulted_params`, `user_specified_params`, and `defaults_confirmed`.
Compute the complete missing-parameter set before execution. Never start a
workflow after confirming only part of a method's critical parameter bundle.

Return `blocked` when y is missing for a supervised method, `top_k` exceeds the
input feature count, task and method are incompatible, or split assignments
duplicate, omit, or reference unknown samples. Clip excessive component counts
to the train-set legal maximum and record a warning.

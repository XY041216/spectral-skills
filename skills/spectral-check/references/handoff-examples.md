# Handoff Examples

## Reader Output Used Directly

If the user skips spectral-check, downstream skills read:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- optional `y.csv`
- optional `metadata.csv`

## Spectral-Check Output After Cleaning

If spectral-check removes confirmed samples or bands, downstream skills still read the same
filenames from the spectral-check output directory.

Do not require downstream skills to know whether the package came from reader or spectral-check.

The spectral-check output `data_contract.json` may include a compact `qc_summary` with
methods used, removed sample count, removed band count, imputation choice, and
handoff readiness. Keep the package filenames unchanged.

## Candidate Summary Only

If spectral-check only reports candidates and does not change data, no new downstream
package is required unless the user asks for one.

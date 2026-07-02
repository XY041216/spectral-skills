# Handoff Examples

## Reader Output Used Directly

If the user skips QC, downstream skills read:

- `data_contract.json`
- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- optional `y.csv`
- optional `metadata.csv`

## QC Output After Cleaning

If QC removes confirmed samples or bands, downstream skills still read the same
filenames from the QC output directory.

Do not require downstream skills to know whether the package came from reader or
QC.

The QC output `data_contract.json` may include a compact `qc_summary` with
methods used, removed sample count, removed band count, imputation choice, and
handoff readiness. Keep the package filenames unchanged.

## Candidate Summary Only

If QC only reports candidates and does not change data, no new downstream
package is required unless the user asks for one.

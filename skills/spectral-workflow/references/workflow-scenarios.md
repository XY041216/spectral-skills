# Workflow Scenarios

- Read only: raw input -> use `spectral-reader` -> `reader_output/data_contract.json`.
- Quality check: raw input or package -> use `spectral-reader` if needed -> use `spectral-check` ->
  `qc_output/qc_result.json`.
- Split and model: raw input or package -> use `spectral-splitter` -> use `spectral-modeling`.
- Full modeling chain: raw input or package -> use `spectral-splitter` -> use
  `spectral-preprocess` -> use `spectral-feature` -> use `spectral-modeling`.
- Include check: use `spectral-check` before splitter. Data-changing check remains a
  confirmation-gated `spectral-check` action.
- Continue from feature output: package starts at feature stage; workflow can
  reuse the provided `split_contract.json` and route directly to
  `spectral-modeling`.

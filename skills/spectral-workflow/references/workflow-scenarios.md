# Workflow Scenarios

- Read only: raw input -> use `spectral-reader` -> `reader_output/data_contract.json`.
- Quality check: raw input or package -> use `spectral-reader` if needed -> use `spectral-qc` ->
  `qc_output/qc_result.json`.
- Split and model: raw input or package -> use `spectral-splitter` -> use `spectral-modeling`.
- Full modeling chain: raw input or package -> use `spectral-splitter` -> use
  `spectral-preprocess` -> use `spectral-feature` -> use `spectral-modeling`.
- Include QC: use `spectral-qc` before splitter. Data-changing QC remains a
  confirmation-gated `spectral-qc` action.
- Continue from feature output: package starts at feature stage; workflow can
  reuse the provided `split_contract.json` and route directly to
  `spectral-modeling`.

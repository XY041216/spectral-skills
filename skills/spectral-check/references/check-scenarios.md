# Quality-Check Scenarios

These cases define expected quality-check behavior.

## Quality Check Only

User asks to inspect quality.

Expected behavior:

- load the standard package;
- check structure, missing values, duplicate spectra candidates, band quality,
  outlier candidates, and task-specific risks;
- explain risks in conversation;
- do not modify data.

## Mark Candidates

User asks to mark outliers or bad bands.

Expected behavior:

- run the selected or recommended method;
- use implemented first-stage sample methods: NOE, MD, PCA_DISTANCE,
  ROBUST_ZSCORE, IQR, and MAD;
- return candidate IDs or band positions;
- do not remove data unless the user confirms removal.

If the user asks to remove outliers without naming a method, ask for method
selection or recommend a first-stage method. Do not default to MCCV.

If the user requests MCCV, HR, PLS residual, PCA T2, Q residual, or class-aware
model-based outlier detection before implementation, say it is unavailable and
ask whether to use an implemented method instead.

## Confirmed Cleaning

User asks to remove or fill and confirms the proposed operation.

Expected behavior:

- apply only the confirmed action;
- write the standard package to a new output directory;
- keep filenames identical to reader output;
- keep `data_contract.json` compact.

## Package Integrity Failure

If `X.csv`, `sample_ids.csv`, `band_axis.csv`, `y.csv`, or `metadata.csv` do not
align, block spectral-check execution. The user should repair the reader package first.

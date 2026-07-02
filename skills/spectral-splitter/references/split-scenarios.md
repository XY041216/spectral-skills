# Split Scenarios

Common requests and behavior:

- "Split 8:2": use train/test, random unless classification stratification is confirmed.
- "Split 7:2:1": use train/validation/test.
- "Use seed 42": pass `--random-seed 42`.
- "Use stratified split": require classification `y.csv`.
- "Use my existing split": use `predefined_split` from metadata `split` or an external `split_indices.csv`.
- "Use KFold": use `kfold`; for classification recommend `stratified_kfold`.
- "Use LOOCV": use `leave_one_out` only when explicitly requested or for small-sample workflows.
- "Use MCCV": use `monte_carlo_cv`, or `stratified_monte_carlo_cv` for classification.
- "Use Kennard-Stone": use `kennard_stone`; explain that it is representative X-space selection, not random.
- "Use SPXY": require regression and numeric y; explain that y participates in splitting.
- "Use group-aware split": require a metadata group column such as `group_id`, `batch`, `year`, `origin`, or `replicate_id`.
- "Split QC output and avoid replicate leakage": prefer `group_aware` when a group column or QC replicate grouping is available.
- "Split after QC and hand off to modeling": read the QC output package, write `split_contract.json`, and keep the original spectral matrices unchanged.

Unsupported requests:

- time-series or chronological splitting;
- nested CV;
- optimizer-specific validation schemes.

# Input Artifact Map

Identify the highest available layer, then follow its lineage to lower layers only as needed.

| Layer | Primary inputs | Valid figure evidence |
|---|---|---|
| reader/QC | `data_contract.json`, `X.csv`, `y.csv`, `sample_ids.csv`, `band_axis.csv`, `qc_result.json` | spectra, class means, sample/class counts, QC candidate overview |
| split | `split_contract.json`, `split_indices.csv`, split summary | experiment-design counts and class proportions only |
| preprocess | `preprocess_contract.json`, output `X.csv`, upstream package | raw/transformed spectra, baseline/scatter/derivative comparison, retained bands |
| feature | `feature_contract.json`, `feature_state.json`, scores/loadings/selected files | PCA/PLS scores/loadings, VIP, SPA/CARS/UVE, selected bands/intervals |
| modeling | `modeling_contract.json`, `metrics.json`, `predictions.csv`, `confusion_matrix.csv`, fold/repeat results, uncertainty outputs | model validation, confusion, ROC/PR/calibration, predicted/measured, residuals, uncertainty |
| optimizer | `optimizer_contract.json`, `trial_results.csv`, `best_pipeline.json`, plan/manifest | validation/CV candidate comparison, parameter response, trial landscape, selected-pipeline evidence |

## Required Checks

- Resolve relative paths against the contract that contains them.
- Verify sample IDs, class labels, band count, and axis length align across files.
- Read test access/evaluation context from workflow, modeling, or optimizer contracts.
- Use stored predictions/scores for diagnostics; do not rerun a model.
- Use stored fold/repeat identifiers for distributions and pairing.
- Treat missing units, ambiguous target meaning, or contradictory split labels as blocking semantic gaps.


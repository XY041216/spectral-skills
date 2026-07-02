# Route Index

Use this compact owner map before reading any child skill details.

| Intent / input state | Owner skill | Required contract or artifact |
| --- | --- | --- |
| Raw spectral file or folder | `spectral-reader` | writes `data_contract.json` |
| Existing standard package | `spectral-qc` or next requested stage | requires `data_contract.json` |
| Data quality inspection or confirmed cleaning | `spectral-qc` | reads standard package; writes `qc_result.json` |
| Train/validation/test assignment | `spectral-splitter` | reads standard package; writes `split_contract.json` |
| Spectral preprocessing | `spectral-preprocess` | requires `split_contract.json`; writes `preprocess_contract.json` |
| Feature extraction, variable selection, embeddings | `spectral-feature` | requires `split_contract.json`; writes `feature_contract.json` |
| Classification or regression | `spectral-modeling` | requires package/preprocess/feature contract plus split |
| Candidate recommendation, tuning, or pipeline search | `spectral-optimizer` | requires reader/QC/split context; never selects by test |
| Publication figure or figure-centered report | `spectral-report` | reads completed contracts and source data only |

Routing rule: load only the active owner skill and its method-selection
fragment. Do not load full menus for stages that are not yet being decided.

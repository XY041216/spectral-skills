# Contract Passing

Pass only standard contracts and package directories between stages:

- reader/qc/preprocess/feature outputs: `data_contract.json`
- splitter output: `split_contract.json`
- modeling output: `modeling_contract.json`

Use the same `split_contract.json` for preprocess, feature, and modeling. Do
not require downstream skills to parse logs, debug reports, inventories, or
workflow internals.

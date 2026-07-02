# Preprocess Scenarios

Common requests and behavior:

- "Do SNV on the split data": run `snv` with the provided split contract.
- "SG smoothing then first derivative": require `window_length` and `polyorder`, then apply methods in order.
- "Use MSC before modeling": fit MSC reference on train samples only, transform all splits, and write a new standard package.
- "Standardize train, val, and test": fit mean/std on train only, then transform all samples.
- "Standardize all data together": warn about leakage and require confirmation if no split contract is used.
- "Use PCA/CARS/SPA": route to later feature-selection scope; do not run it in preprocess MVP.

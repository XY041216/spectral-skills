# Class And Target Risks

Use `y.csv` and task hints from `data_contract.json` when available.

## Classification

Check:

- class counts;
- rare classes;
- class imbalance ratios;
- missing class labels;
- class-aware sample outlier candidates when labels are reliable.

Do not delete rare classes or rebalance data. Rebalancing belongs to later
modeling or splitting strategy.

## Regression

Check:

- missing target values;
- target range;
- IQR or MAD target outlier candidates;
- inconsistent target scale hints from metadata when obvious.

Do not cap, transform, or delete targets without confirmation. Target
transformation belongs to modeling or preprocessing decisions.

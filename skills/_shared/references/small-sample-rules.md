# Small Sample Rules

Small-sample settings require conservative interpretation and explicit risk
warnings.

## Risk Scenarios

- Total sample count is low.
- One or more classes have very few samples.
- The p/n ratio is high: features greatly outnumber samples.
- Train/test splitting leaves too few samples per class or target range.
- Cross-validation folds become unstable.
- Multiple preprocessing, feature selection, or model choices are searched
  without enough validation evidence.

## Shared Guidance

Do not treat a high metric from a tiny validation set as definitive. Preserve
small-sample warnings in Contracts and reports. Prefer leakage-aware validation
and avoid optimistic claims.

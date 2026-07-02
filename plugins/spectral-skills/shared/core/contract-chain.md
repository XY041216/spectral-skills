# Contract Chain

Contracts are the handoff objects between Spectral Skills. Downstream skills
must read upstream Contracts rather than silently re-guessing missing fields.

## Order

1. Spectral Data Contract
2. QC Contract
3. Split Contract
4. Preprocess Contract
5. Feature Contract
6. Model Contract
7. Optimization Contract
8. Report Contract

## Shared Requirements

Every Contract must record:

- its own `contract_id` and `contract_type`;
- `contract_status`;
- input Contract IDs;
- output paths or artifact references;
- execution record;
- warnings, errors, and confirmation log;
- schema version and tool or script chain.

## Blocking Rule

A `blocked` Contract prevents unsafe downstream execution. A downstream skill
may create a diagnostic report about the block, but it must not manufacture
missing X, y, sample IDs, split indices, preprocessing fits, features, model
metrics, or reports.

## Provisional Rule

A `provisional` Contract may be passed downstream only when the next skill can
preserve the unresolved confirmations and avoid making irreversible claims.
Modeling and optimization generally require confirmed upstream data and split
contracts.

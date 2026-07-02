# Missing Values

Missing values are expected QC inputs when reader preserved a valid sample by
band matrix.

## Check

Inspect missing values in:

- `X.csv`
- `y.csv`
- `metadata.csv`

Summarize missingness by sample and by band. A conversational summary is enough
unless the user asks for a candidate table.

## Actions

Default action is to preserve missing values and explain the risk.

Filling missing values requires confirmation. The user should know the method
and scope before QC writes a modified package.

Do not delete samples or bands solely because missing values exist unless the
user explicitly confirms a threshold and action.

## Handoff

If no filling or deletion is confirmed, the package can still go downstream with
missing-value warnings. Downstream modeling skills may later require either QC
cleaning or model-specific missing-value support.

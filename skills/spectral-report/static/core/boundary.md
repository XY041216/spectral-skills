# Boundary

## Do

- Read standard contracts and result files from spectral-reader through spectral-optimizer.
- Create or revise single figures, multipanel main figures, Extended Data figures, or a coherent figure set.
- Extract figure source data, write task-specific Python, export figures, draft captions, and record QA.
- Audit existing plots for lineage, statistics, units, axes, typography, palette, and export quality.

## Do Not

- Parse or repair vendor raw files; route them to spectral-reader.
- Execute QC actions, splitting, preprocessing, feature fitting, model training, or optimization.
- Choose candidate classifiers, feature methods, preprocessing methods, or model parameters for an upstream comparison.
- Write `model-config.json`, `candidate_space.json`, `trial_inputs.json`, or any other upstream execution input.
- Call spectral-modeling, spectral-feature, spectral-preprocess, or spectral-optimizer scripts as part of the report step.
- Recompute a missing metric in a way that changes a locked result or pipeline.
- Use test metrics to choose methods or parameters.
- Create illustrative scientific values that cannot be traced to an input artifact.
- Use R in v1. If the user explicitly requires R, return unsupported and keep the current run unchanged.

Treat upstream contracts as read-only evidence. If a required value is absent, request the minimum clarification or route to the owning upstream skill.

## Mixed Compute And Report Requests

When the user asks for a new experiment and a figure in the same sentence, split the work:

1. Use spectral-workflow/modeling/optimizer to produce standard artifacts after confirming missing candidate sets, budgets, and locked parameters.
2. Use spectral-report only after those artifacts exist.

Do not let the report phase conceal upstream decisions. If "compare classifiers" is underspecified, ask whether to use a compact set, a full traditional set, or user-specified classifiers before any modeling run.

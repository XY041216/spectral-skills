# Revise an Existing Figure

Read `references/qa-checklist.md`, `references/chart-decision-matrix.md`, `references/axis-and-unit-rules.md`, and `references/visual-system.md`.

1. Locate the figure's source data, code, contract, and upstream result files. Block data-changing redraws when lineage is unavailable.
2. Audit scientific claim, statistical unit, split, uncertainty, metric, unit, and test isolation before visual styling.
3. List high-impact issues first: leakage, invented distribution, misleading chart grammar, truncated bars, missing units, or unreadable export.
4. Preserve valid data and semantics while rewriting task-specific plotting code.
5. Re-export all requested formats and rerun both QA layers.
6. Record what changed in `qa/figure_qa.md`; do not preserve verbose debugging history.

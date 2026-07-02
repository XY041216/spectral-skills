# Spectral Skills Shared Layer

`skills/_shared/` is the shared resource layer for all Spectral Skills.

It is not a user-facing Codex skill and must not contain `SKILL.md`. Concrete
skills may read shared schemas, core rules, method registries, and references
from this directory, but each skill owns its own workflow, tool choices,
confirmation behavior, and domain-specific references.

## Contents

- `schemas/`: base Contract and execution record schemas shared by skills.
- `core/`: shared confirmation, execution, reproducibility, boundary, and
  contract-chain rules.
- `references/`: cross-skill background knowledge.

## Boundary

Keep cross-skill rules here. Put reader-specific file quirks, CSV or Excel
cases, parsing plans, preview behavior, and `read_plan` examples in
`skills/spectral-reader/references/` when the reader skill is implemented.

Plugin release mirrors should include this directory as shared resources but
should avoid exposing `_shared` as a directly callable skill.

Executable method IDs and aliases are canonical in `spectral_core`. Each
user-facing skill keeps only its routing/confirmation menu and manifest scope;
do not create a second shared method registry that can drift from runtime code.

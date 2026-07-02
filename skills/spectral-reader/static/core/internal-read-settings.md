# Internal Read Settings

The reader may build internal read settings so `read_spectral_dataset` can
execute deterministically. These settings are not a user-facing workflow,
development guide, persisted artifact, or required concept for downstream
skills.

## Required Semantics

Before reading, the entrypoint must know or confirm:

- input path and file/folder type;
- encoding, delimiter, sheet, variable, or dataset path when relevant;
- sample orientation;
- sample ID source;
- label or target source when requested;
- metadata columns when requested;
- spectral columns, variable, or dataset path;
- band axis source and unit when known;
- external label alignment key or explicit row-order alignment;
- any required user confirmation.

Internal preview evidence may be used to choose or ask for these semantics, but
it must not be written as `preview_report.json` or exposed as a separate user
workflow.

## Reader Status

Return `ready` only after standard files are written and assertions pass. Return
`needs_confirmation` when a minimal user choice is required. Return `blocked`
when the input or parameters cannot produce reliable standard output.

## Hard Assertions

Python code performs deterministic reading and hard assertions. It must not ask
users to understand internal settings, run staged validators, or inspect
intermediate reports.

Assertions cover numeric X, row counts, band axis length, label alignment,
metadata alignment, and explicit blocked reasons.

Confidence scores are not part of the downstream Data Contract.

## Execution Requirements

For execution, internal read settings must have no unresolved required
confirmations and must use a supported read mode. CSV/TSV/TXT matrix-file
settings with samples as rows must identify:

- source path;
- encoding and delimiter;
- skipped rows and header row when needed;
- sample orientation;
- spectral columns or a spectral start/end region;
- optional sample ID, label, target, and metadata columns;
- band axis strategy and unit.

The reader must not invent missing semantic fields when doing so would change
the meaning of user data.

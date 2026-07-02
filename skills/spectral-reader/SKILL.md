---
name: spectral-reader
description: >-
  Use only when Codex needs to read, parse, or convert spectral datasets from
  CSV/TSV/TXT, Excel/ODS, NPY/NPZ/MAT, HDF5/NetCDF, external labels, or
  one-file-per-sample folders into X.csv, optional y.csv, sample_ids.csv,
  band_axis.csv, optional metadata.csv, and data_contract.json. Do not use this
  skill to develop, modify, refactor, test, package, install, document, or
  review the spectral-reader skill itself; treat those as software development.
---

# Spectral Reader

`spectral-reader` has one job: read user spectral data into standard data files
that downstream skills can use directly.

It is the entry point before `spectral-qc`, `spectral-splitter`,
`spectral-preprocess`, `spectral-feature`, and `spectral-modeling`.

## Activation Boundary

Use this skill only when the user wants to read, parse, or convert spectral
datasets into the standard spectral package:

- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- `data_contract.json`
- optional `y.csv`
- optional `metadata.csv`

This skill is only for reading spectral datasets.

Do not use this skill to develop, modify, refactor, test, package, install,
document, or review the spectral-reader skill itself. When the user asks to
build, edit, audit, clean, package, or improve this skill, treat the task as
software development, not as spectral data reading.

Do not use this skill for QC, splitting, preprocessing, feature engineering,
modeling, optimization, fixed reporting, debugging archives, log systems, or
audit packages.

## Core Flow

Follow this flow:

`input data -> determine or confirm read semantics -> read deterministically -> assert structure -> write standard package`

After deterministic reading, run a post-read audit before marking the package
ready:

- compare original table width and numeric band-like headers against X feature
  count;
- verify band_axis length equals X feature count;
- block non-monotonic numeric band axes;
- block declared sample, label, target, or metadata columns if they remain in X;
- return `blocked` or `needs_confirmation` when the source has many continuous
  numeric band headers but X contains only a small subset.

Use `read_spectral_dataset` for user work. Do not route normal reading through a
chain of separate helper scripts.

For CSV/TSV/TXT tables whose first column is a sample index with an empty
header, use the formal CLI parameter `--sample-id-column-index 0` or
`--sample-id-column-position 0`. Do not work around empty headers by wrapping
the CLI with `runpy` or temporary Python argument injection. The resulting
`data_contract.json` must record `sample_id_source`, for example
`source_first_column_empty_header`.

When band-axis semantics are known, pass both type and unit explicitly. Keep
them separate:

- wavenumber axis: `--band-type wavenumber --band-unit cm-1`;
- wavelength axis: `--band-type wavelength --band-unit nm`;
- unknown/generated axis: `--band-unit unknown` or `--band-unit index`.

Do not pass `--band-unit wavenumber`; `wavenumber` is an axis type, not a unit.
Wide CSV/TSV/TXT tables are first-class spectral inputs. The reader scans the
full header by default up to `--max-auto-columns 10000`, detects long
contiguous monotonic numeric header blocks as spectral columns, and supports
up to `--max-spectral-columns 20000` before asking for confirmation. Tablet
style tables with an empty first header, numeric headers such as `3600..200`,
and final `class`/`label`/`target` columns should be auto-recognized. Use
explicit `--spectral-start-column`, `--spectral-end-column`,
`--sample-id-column-index`, and `--confirm-read-plan` only as a reproducibility
or fallback path when automatic inference is not sufficient.

Internal read settings may exist while reading, but they are not a user-facing
workflow and must not be treated as a development process.

Semantic rules belong in the static knowledge fragments and references. Python
code should only perform deterministic reading, path resolution, conversion,
standardization, export, and hard assertions.

## Standard Handoff

Write one flat standard package for downstream skills:

- `X.csv`
- `sample_ids.csv`
- `band_axis.csv`
- `data_contract.json`
- `y.csv` when labels or targets are available
- `metadata.csv` when metadata is available

Downstream skills should only need `data_contract.json` and the standard CSV
files. They should not need read settings, inventories, validation reports,
summaries, logs, or package manifests.

## Confirmation Rules

Ask for the smallest necessary confirmation when read semantics are ambiguous.
Do not guess when more than one interpretation is plausible.

Confirm before:

- reading samples as columns;
- choosing among multiple spectral files, sheets, variables, or dataset paths;
- choosing among multiple label, metadata, or band-axis candidates;
- using row-order label alignment;
- generating replacements for partially missing sample IDs;
- treating a user-specific token as a missing value when it is not already a
  known missing token.

Block instead of writing a pseudo-ready package when structural alignment fails.
Block instead of writing a ready package when post-read audit shows suspicious
feature loss, for example a CSV with thousands of numeric wavenumber columns
but an X matrix with only tens of features. Recommend explicit
`spectral_start_column` and `spectral_end_column` rather than continuing.

## Execution Boundary

This skill is a reader skill, not a script collection, workflow system,
evaluation platform, log system, preflight platform, debugging platform, or
audit system. All user-facing reading capability is reached through
`read_spectral_dataset`.

Before returning ready output, assert:

- `X.csv` exists, is numeric-or-missing, rectangular, and has nonzero rows and
  columns.
- `sample_ids.csv`, when present, has the same row count as `X.csv`.
- `y.csv`, when present, has the same row count as `X.csv`.
- `metadata.csv`, when present, has the same row count as `X.csv`.
- `band_axis.csv` has the same row count as the feature count of `X.csv`.
- external labels align by the confirmed sample key.
- post-read audit did not find suspicious truncation, label/sample columns in X,
  band_axis length mismatch, or invalid numeric band-axis order.

Missing spectral values may remain in `X.csv`. Imputation, deletion, duplicate
spectra decisions, and outlier handling belong to `spectral-qc`.

## Output Boundary

Keep `data_contract.json` minimal and downstream-useful. It should describe
file refs, shape, source, layout, task hints, band-axis status, missing-value
status, sample-ID status, and compact warnings only as needed.

Common downstream fields include:

- `status`
- `X`
- `y`
- `sample_ids`
- `band_axis`
- `metadata`
- `n_samples`
- `n_features`
- `sample_orientation`
- `label_status`
- `task_hint`
- `band_unit`
- `source_type`
- `missing_value_status`
- `sample_id_status`
- `sample_id_source`
- `warnings`

Do not include confidence scores, full read plans, preview evidence, validation
details, stage results, internal refs, debug refs, logs, or package manifests.

Do not create package manifests, summaries, `_internal/`, persisted preview
reports, validation reports, profile reports, logs, decision traces, confidence
drafts, inventory outputs, or multi-view outputs.

## Current Scope

Supported execution scope:

- CSV/TSV/TXT spectral matrix files with samples as rows.
- CSV/TSV/TXT spectral matrix files with samples as columns when confirmed.
- Excel/ODS workbook tables with one clear spectral sheet.
- Excel workbooks with a confirmed spectral sheet and optional label sheet.
- NPY single 2D numeric matrices.
- NPZ containers with selected X/y/sample_ids/band_axis variables.
- Non-HDF5 MAT files with selected X/y/sample_ids/band_axis variables.
- External label files aligned by `sample_id`.
- One-file-per-sample folders using confirmed CSV/TSV/TXT band/value columns.
- Mixed folders with confirmed or uniquely detected spectra, labels, metadata,
  and band-axis files.
- Folder-name or file-name labels when explicitly requested.
- Row-order label alignment when explicitly allowed.
- Missing values in X preserved as missing values when non-missing cells remain
  numeric.

Not in scope:

- Automatic merging of multiple workbook spectral sheets.
- MAT v7.3 HDF5-style MATLAB execution.
- Deep MATLAB struct, complex cell arrays, sparse arrays, or 3D cube unfolding.
- QC, split, preprocess, feature engineering, modeling, optimization.
- Legacy parser/inference main flows.
- Debug/user/standard output modes.
- Package manifests or audit archives.
- Instrument-specific proprietary format support.

## Read As Needed

- Use `static/core/workflow.md` for the one-shot reader flow.
- Use `static/core/internal-read-settings.md` for internal read settings that
  must not become user-facing outputs.
- Use `static/core/confirmation-gates.md` when read semantics are ambiguous.
- Use `static/core/execution-boundary.md` for reader scope and blocked cases.
- Use `static/core/output-contract.md` for the minimal standard package.
- Use `static/core/handoff-rules.md` for downstream handoff rules.
- Use `static/fragments/csv-with-preamble.md` for commented or preamble CSV/TXT files.
- Use `static/fragments/metadata-before-spectra.md` when metadata columns appear before spectra.
- Use `static/fragments/samples-as-columns.md` when samples may be columns.
- Use `static/fragments/external-label-file.md` for external label alignment.
- Use `static/fragments/sample-files-folder.md` for one-file-per-sample folders.
- Use `static/fragments/folder-name-as-label.md` for folder or file name labels.
- Use `static/fragments/excel-multi-sheet.md` and `static/fragments/excel-layout-cases.md` for workbook layouts.
- Use `static/fragments/mat-npz-variable-selection.md` for NPZ and MAT variable mapping.
- Use `static/fragments/hdf5-netcdf-dataset-path.md` for HDF5 or NetCDF dataset paths.
- Use `static/fragments/complex-table-layout.md` for multirow headers or partitioned tables.
- Use `static/fragments/numeric-band-columns.md`, `wavelength-columns.md`, and `wavenumber-columns.md` for band-axis recognition.
- Use `static/fragments/chinese-column-names.md` for Chinese sample, label, target, metadata, and band column names.
- Use `static/fragments/missing-values.md`, `static/fragments/sample-id-rules.md`, and `static/fragments/reader-qc-boundary.md` for missing values, sample IDs, and reader/QC boundaries.
- Use `references/*-cases.md` only for concrete scenario examples after selecting the relevant fragment.

# Reader Handoff Rules

Downstream skills use `data_contract.json` and the standard CSV files in the
same output directory.

They should not require package manifests, summaries, `_internal/`, logs,
preview evidence, read plans, validation reports, or confidence scores.

## Ready Handoff

`status: ready` means:

- `X.csv` exists, is rectangular, numeric-or-missing, and nonempty;
- `band_axis.csv` length equals the feature count of `X.csv`;
- `sample_ids.csv`, `y.csv`, and `metadata.csv`, when present, align to the
  sample count of `X.csv`;
- external labels, when used, have already been aligned to the spectra order.

For samples-as-columns reads, `X.csv` is already samples by features.

Missing spectral values may remain blank in `X.csv`. The reader must preserve
the sample by band structure; imputation, deletion, or retention decisions
belong to downstream QC.

## Blocked Handoff

If a hard reader assertion fails, return `blocked` with a short reason and do
not claim the output is downstream-ready.

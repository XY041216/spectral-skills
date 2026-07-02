# Spectral Skills Plugin

`spectral-skills` is a local agent plugin image for spectral data skills. It
is packaged for Codex through `.codex-plugin/plugin.json` and for
Claude-compatible agents through the repository-level `.claude-plugin/`
metadata. This beta ships the `spectral-reader`, `spectral-qc`, `spectral-splitter`,
`spectral-preprocess`, `spectral-feature`, `spectral-modeling`,
`spectral-optimizer`, `spectral-report`, and `spectral-workflow` skills.

## Included Skills

- `spectral-reader`: read user spectral data files or folders through the
  single `read_spectral_dataset` entrypoint and write `X.csv`, optional
  `y.csv`, `sample_ids.csv`, `band_axis.csv`, optional `metadata.csv`, and
  `data_contract.json`.
- `spectral-qc`: inspect and, after user confirmation, clean standard spectral
  packages before splitting, preprocessing, feature engineering, or modeling.
- `spectral-splitter`: create reproducible random or classification-stratified
  train/validation/test assignments and write `split_indices.csv` plus
  `split_contract.json` without copying spectral matrices.
- `spectral-preprocess`: apply leakage-safe spectral preprocessing after
  splitting, fitting global parameters on train samples only and writing a new
  standard package plus `preprocess_state.json`.
- `spectral-feature`: apply leakage-safe feature engineering after splitting
  or preprocessing, fitting PCA and low-variance selection on train samples only
  and writing a new standard package plus `feature_state.json`.
- `spectral-modeling`: train leakage-safe classification or regression models,
  select candidates on validation or train-only CV, evaluate the final model on
  test, and write `modeling_contract.json`, `metrics.json`, and
  `predictions.csv`.
- `spectral-optimizer`: recommend candidates, tune one method, compare one
  workflow stage, or plan a budgeted pipeline search using validation/CV
  metrics without selecting by test performance.
- `spectral-report`: create publication-grade spectral figures, source data,
  captions, task-specific plotting code, report contracts, and figure QA from
  completed workflow artifacts without rerunning analysis.
- `spectral-workflow`: orchestrate the minimal skill chain, pass only standard
  contracts between stages, and write `workflow_result.json`.

Shared resources live in `shared/`. They are not exposed as user-callable
skills.

## Reader Entry

`read_spectral_dataset` is the user-facing reader entrypoint. It determines or
asks for the minimum required read semantics, reads deterministically, runs hard
assertions, and writes only the standard output files. It does not expose staged
workflow modes, debug archives, reports, inventories, or persisted internal
read settings.

## Fallback Scripts

Run from the plugin root:

```bash
python skills/spectral-reader/scripts/server_health.py --json
python skills/spectral-reader/scripts/check_consistency.py --json
python scripts/reader/check_consistency.py --json
python skills/spectral-reader/scripts/read_spectral_dataset.py --input path/to/data.csv --output-dir outputs/plugin_reader_basic --json
python skills/spectral-qc/scripts/qc_spectral_package.py --mode methods --json
python skills/spectral-splitter/scripts/split_spectral_package.py --package-dir outputs/plugin_reader_basic --output-dir outputs/plugin_split_basic --method random --ratio 8:2 --json
python skills/spectral-preprocess/scripts/preprocess_spectral_package.py --package-dir outputs/plugin_reader_basic --split-contract outputs/plugin_split_basic/split_contract.json --output-dir outputs/plugin_preprocess_basic --methods snv --json
python skills/spectral-feature/scripts/feature_spectral_package.py --package-dir outputs/plugin_preprocess_basic --split-contract outputs/plugin_split_basic/split_contract.json --output-dir outputs/plugin_feature_basic --method variance_threshold --json
python skills/spectral-modeling/scripts/model_spectral_package.py --package-dir outputs/plugin_feature_basic --split-contract outputs/plugin_split_basic/split_contract.json --output-dir outputs/plugin_model_basic --task-type classification --models random_forest_classifier --json
python skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py --mode recommend_from_profile --task-type classification --n-samples 120 --n-features 3401 --output-dir outputs/plugin_optimizer_basic --json
python skills/spectral-workflow/scripts/run_spectral_workflow.py --package-dir outputs/plugin_reader_basic --output-dir outputs/plugin_workflow_basic --task-goal classification --split-ratio 8:2 --preprocess-methods none --feature-method none --models random_forest_classifier --json
```

## MCP

`.mcp.json` starts the `spectral-reader` MCP server with the generic `python`
command and `PYTHONPATH=.`. It does not contain a machine-specific Python path.

## Local Marketplace

The repository-level `.agents/plugins/marketplace.json` points Codex to
`./plugins/spectral-skills`. The repository-level `.claude-plugin/` directory
contains Claude-compatible plugin and marketplace metadata that reuse the same
skill image.

## Codex Config Preflight

If Codex reports that it cannot load `config.toml` after importing or enabling
the plugin, the failing file is the user-level Codex configuration, not the
Spectral Skills runtime. Validate it before retrying plugin import:

```bash
python install/check_codex_config.py --json
```

An `unclosed table, expected ]` error usually means an older `[projects.'...']`
entry in `~/.codex/config.toml` was truncated or contains a malformed path.
Fix or remove the malformed table, then rerun the preflight and reopen Codex.

## Codex Desktop Cache Install

If Codex Desktop cannot execute the Codex CLI or the plugin appears in
`config.toml` but not under `~/.codex/plugins/cache`, run the repository
installer from the clone root:

```bash
python install/install_codex_plugin.py --json
```

The installer validates `config.toml`, writes a backup before changing it, adds
the local marketplace/plugin enablement entries, and materializes the built
plugin image at
`~/.codex/plugins/cache/spectral-skills-local-marketplace/spectral-skills/<version>/`.
Restart Codex in a new thread after it reports success.

## Scope

The plugin supports standard spectral reading and QC, holdout/CV/repeated and
representative splitting, leakage-safe preprocessing, traditional and confirmed
deep feature extraction, traditional/chemometric/optional boosting/experimental
small-sample modeling, bounded pipeline optimization, publication figures, and
multi-stage workflow routing.

Out of scope: proprietary instrument formats without an exported tabular or
container representation, silent sample deletion, test-set-driven selection,
unbounded AutoML, and unconfirmed deep training. Optional methods such as UMAP,
PyTorch models, XGBoost, LightGBM, and CatBoost require their local dependencies.

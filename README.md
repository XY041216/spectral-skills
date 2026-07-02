# Spectral Skills

Spectral Skills is a leakage-aware agent skill collection for end-to-end
spectral data analysis. It ships a Codex plugin image and Claude-compatible
plugin metadata from the same source tree. The repository contains nine
user-facing skills:

- `spectral-reader`: standardize supported spectral files and folders;
- `spectral-qc`: inspect and confirm non-destructive or destructive QC actions;
- `spectral-splitter`: create reproducible holdout, CV, repeated, representative,
  and group-aware splits;
- `spectral-preprocess`: apply train-safe spectral preprocessing;
- `spectral-feature`: run traditional, manifold, or confirmed deep embeddings;
- `spectral-modeling`: train, tune, compare, and evaluate models without using
  test data for selection;
- `spectral-optimizer`: run bounded, auditable validation/CV searches;
- `spectral-report`: create publication-grade figures and report artifacts;
- `spectral-workflow`: route multi-stage requests across the child skills.

## Installation

### Codex plugin marketplace

Codex users should install the complete plugin bundle rather than copying a
single skill folder. The bundle is generated at `plugins/spectral-skills/` and
is exposed through `.agents/plugins/marketplace.json`.

CLI installation:

```bash
codex plugin marketplace add https://github.com/XY041216/spectral-skills.git --ref main
codex plugin add spectral-skills@spectral-skills-local-marketplace
```

Codex Desktop users can add the same repository as a custom plugin marketplace:

- Marketplace source: `https://github.com/XY041216/spectral-skills.git`
- Branch/ref: `main`
- Plugin: `spectral-skills`

After installation, start a new Codex session and invoke the full workflow skill
naturally, for example:

```text
Use $spectral-skills:spectral-workflow to process Tablet_ext_0-3.csv:
read, QC, stratified 6:2:2 split, SNV preprocessing, feature=none, and SVM.
```

### Claude-compatible agents

This repository also includes Claude-compatible plugin metadata under
`.claude-plugin/`, following the same public plugin layout pattern used by
`nature-skills-reference`.

CLI installation in compatible Claude environments:

```bash
claude plugin marketplace add XY041216/spectral-skills
claude plugin install spectral-skills@spectral-skills
```

If your agent does not support plugin marketplaces, clone the repository and
point the agent at `plugins/spectral-skills/` plus the relevant `SKILL.md`
entrypoint. Keep `shared/`, `spectral_core/`, and `scripts/` together with the
skills because the workflow depends on those runtime resources.

### Manual local use

Clone the repository:

```bash
git clone https://github.com/XY041216/spectral-skills.git
cd spectral-skills
```

Install Python dependencies for direct script use:

```bash
pip install -r requirements.txt
```

Then run the workflow script from the plugin image:

```bash
python plugins/spectral-skills/skills/spectral-workflow/scripts/run_spectral_workflow.py \
  --input path/to/data.csv \
  --output-dir outputs/workflow_demo \
  --task-goal classification \
  --split-ratio 6:2:2 \
  --preprocess-methods snv \
  --feature-method none \
  --models random_forest_classifier \
  --json
```

See [`install.md`](install.md) for local marketplace configuration details.

## Repository Layout

Development source lives in `skills/` and `spectral_core/`. The distributable
agent plugin image is generated in `plugins/spectral-skills/`; do not hand-edit
that directory. Codex metadata lives in `.agents/plugins/marketplace.json` and
`plugins/spectral-skills/.codex-plugin/plugin.json`; Claude-compatible metadata
lives in `.claude-plugin/`. Shared schemas and cross-skill rules live under
`skills/_shared/`.

The standard reader handoff is a flat package containing `X.csv`, optional
`y.csv`, `sample_ids.csv`, `band_axis.csv`, optional `metadata.csv`, and
`data_contract.json`. Downstream stages add their own contracts without using
the final test split for method or parameter selection.

## Supported Scope

The reader supports CSV/TSV/TXT, Excel/ODS, NPY/NPZ/MAT, HDF5/NetCDF, external
labels, samples as rows or columns, one-file-per-sample folders, and mixed
folders when the required semantics are clear or confirmed. Downstream skills
support the methods listed in their `SKILL.md` and method-selection fragments.
Optional UMAP, PyTorch, XGBoost, LightGBM, and CatBoost paths require their local
dependencies and explicit confirmation where documented.

Out of scope are silent sample deletion, test-set-driven selection, unbounded
AutoML, unconfirmed deep training, and proprietary instrument formats that have
not been exported to a supported tabular or container representation.

## Skill Index

| Skill | Purpose |
| --- | --- |
| [`spectral-reader`](skills/spectral-reader/SKILL.md) | Read raw spectral files/folders into a standard package. |
| [`spectral-qc`](skills/spectral-qc/SKILL.md) | Inspect spectral data quality and record confirmed QC actions. |
| [`spectral-splitter`](skills/spectral-splitter/SKILL.md) | Create reproducible train/validation/test, CV, repeated, representative, and group-aware splits. |
| [`spectral-preprocess`](skills/spectral-preprocess/SKILL.md) | Apply leakage-safe spectral preprocessing after splitting. |
| [`spectral-feature`](skills/spectral-feature/SKILL.md) | Run feature selection, dimensionality reduction, signal features, manifold methods, and gated deep embeddings. |
| [`spectral-modeling`](skills/spectral-modeling/SKILL.md) | Train, tune, compare, lock, and evaluate classification/regression models. |
| [`spectral-optimizer`](skills/spectral-optimizer/SKILL.md) | Run bounded validation/CV-based preprocessing, feature, and model searches. |
| [`spectral-report`](skills/spectral-report/SKILL.md) | Produce publication-grade figures, source data, captions, and QA records. |
| [`spectral-workflow`](skills/spectral-workflow/SKILL.md) | Route multi-stage spectral analysis across the child skills. |

## Development and Release Checks

Run from the repository root:

```powershell
python -m pytest -q
python install\build_codex_plugin.py --clean --verify --json
python install\check_codex_plugin.py --json
```

The Codex local marketplace entry is `.agents/plugins/marketplace.json`; the
Claude-compatible local marketplace and plugin metadata are under
`.claude-plugin/`; the shared release artifact is `plugins/spectral-skills/`.

## Contribution Notes

Keep `SKILL.md` files concise and route detailed method menus through
`static/`, `references/`, or runtime manifests. Do not commit cache folders,
ad-hoc outputs, generated QA runs, or local virtual environments. Rebuild the
plugin image with `install/build_codex_plugin.py` before publishing so the
source tree and public plugin mirror stay aligned.

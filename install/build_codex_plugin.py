"""Build the local Codex plugin image for Spectral Skills."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.reader.response import error_response, ok_response


PLUGIN_NAME = "spectral-skills"
PLUGIN_VERSION = "0.1.0-beta.1"
PLUGIN_DESCRIPTION = (
    "Agent-oriented spectral data skills for leakage-safe spectral reading, QC, "
    "splitting, preprocessing, feature extraction, modeling, optimization, "
    "reporting, and workflow routing."
)
PLUGIN_AUTHOR = {
    "name": "Spectral Skills Contributors",
    "url": "https://github.com/XY041216/spectral-skills",
}
PLUGIN_KEYWORDS = [
    "spectral",
    "chemometrics",
    "classification",
    "regression",
    "workflow",
    "publication-figures",
]
PLUGIN_INTERFACE = {
    "displayName": "Spectral Skills",
    "shortDescription": "Leakage-safe spectral data processing workflows.",
    "longDescription": (
        "Spectral Skills packages nine coordinated Codex skills for spectral "
        "data reading, QC, splitting, preprocessing, feature extraction, "
        "modeling, optimization, reporting, and workflow routing. The bundle "
        "keeps shared runtime code and schemas inside one plugin image so the "
        "skills install and update together."
    ),
    "developerName": "Spectral Skills Contributors",
    "category": "Productivity",
    "capabilities": ["Write", "Data Analysis", "Research"],
    "websiteURL": "https://github.com/XY041216/spectral-skills",
    "defaultPrompt": [
        "Run a leakage-safe spectral workflow.",
        "Read spectral data into a standard package.",
        "Create publication-ready spectral figures.",
    ],
    "brandColor": "#2563EB",
}
PLUGIN_DIR = ROOT / "plugins" / PLUGIN_NAME
CLAUDE_PLUGIN_DIR = ROOT / ".claude-plugin"
EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "outputs",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "fixtures",
    "evals",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {"package_manifest.json", "standardized_package_report.json"}


def build_codex_plugin(*, clean: bool = False, verify: bool = False) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    copied: list[dict[str, str]] = []
    written: list[str] = []
    try:
        if clean and PLUGIN_DIR.exists():
            shutil.rmtree(PLUGIN_DIR)
        _ensure_dirs()
        copied.append(_copy_tree(ROOT / "spectral_core", ROOT / "skills" / "spectral-core" / "spectral_core"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-reader", PLUGIN_DIR / "skills" / "spectral-reader"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-qc", PLUGIN_DIR / "skills" / "spectral-qc"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-splitter", PLUGIN_DIR / "skills" / "spectral-splitter"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-preprocess", PLUGIN_DIR / "skills" / "spectral-preprocess"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-feature", PLUGIN_DIR / "skills" / "spectral-feature"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-modeling", PLUGIN_DIR / "skills" / "spectral-modeling"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-optimizer", PLUGIN_DIR / "skills" / "spectral-optimizer"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-report", PLUGIN_DIR / "skills" / "spectral-report"))
        copied.append(_copy_tree(ROOT / "skills" / "spectral-workflow", PLUGIN_DIR / "skills" / "spectral-workflow"))
        copied.append(_copy_tree(ROOT / "skills" / "_shared", PLUGIN_DIR / "shared"))
        copied.append(_copy_tree(ROOT / "spectral_core", PLUGIN_DIR / "spectral_core"))
        copied.append(_copy_tree(ROOT / "scripts" / "reader", PLUGIN_DIR / "scripts" / "reader"))
        copied.append(_copy_tree(ROOT / "scripts" / "qc", PLUGIN_DIR / "scripts" / "qc"))
        copied.append(_copy_tree(ROOT / "scripts" / "splitter", PLUGIN_DIR / "scripts" / "splitter"))
        copied.append(_copy_tree(ROOT / "scripts" / "preprocess", PLUGIN_DIR / "scripts" / "preprocess"))
        copied.append(_copy_tree(ROOT / "scripts" / "feature", PLUGIN_DIR / "scripts" / "feature"))
        copied.append(_copy_tree(ROOT / "scripts" / "modeling", PLUGIN_DIR / "scripts" / "modeling"))
        copied.append(_copy_tree(ROOT / "scripts" / "optimizer", PLUGIN_DIR / "scripts" / "optimizer"))
        copied.append(_copy_tree(ROOT / "scripts" / "workflow", PLUGIN_DIR / "scripts" / "workflow"))
        copied.append(_copy_tree(ROOT / "install", PLUGIN_DIR / "install", include_names={"build_codex_plugin.py", "check_codex_plugin.py"}))
        written.extend(_write_plugin_files())
        written.append(str(_write_marketplace()))
        written.extend(str(path) for path in _write_claude_plugin_files())
        result = {
            "plugin_dir": str(PLUGIN_DIR),
            "plugin_name": PLUGIN_NAME,
            "plugin_version": PLUGIN_VERSION,
            "copied": copied,
            "written_files": written,
            "clean": clean,
            "verify": verify,
            "excluded": {
                "directories": sorted(EXCLUDED_DIRS),
                "suffixes": sorted(EXCLUDED_SUFFIXES),
                "names": sorted(EXCLUDED_NAMES),
            },
        }
        if verify:
            from check_codex_plugin import check_codex_plugin

            verify_response = check_codex_plugin()
            result["verify_result"] = verify_response.get("result", {})
            if not verify_response.get("ok"):
                return error_response(
                    "build_codex_plugin",
                    "Plugin image was built but verification failed.",
                    backend="script",
                    code="PLUGIN_VERIFY_FAILED",
                    result=result,
                    warnings=warnings,
                    details={"errors": verify_response.get("errors", [])},
                )
        return ok_response("build_codex_plugin", result, backend="script", warnings=warnings)
    except Exception as exc:
        return error_response(
            "build_codex_plugin",
            "Failed to build Codex plugin image.",
            backend="script",
            code="PLUGIN_BUILD_FAILED",
            result={"plugin_dir": str(PLUGIN_DIR)},
            warnings=warnings,
            details={"error": str(exc)},
        )


def _ensure_dirs() -> None:
    for path in [
        PLUGIN_DIR / ".codex-plugin",
        PLUGIN_DIR / "skills",
        PLUGIN_DIR / "shared",
        PLUGIN_DIR / "spectral_core",
        PLUGIN_DIR / "scripts" / "reader",
        PLUGIN_DIR / "scripts" / "qc",
        PLUGIN_DIR / "scripts" / "splitter",
        PLUGIN_DIR / "scripts" / "preprocess",
        PLUGIN_DIR / "scripts" / "feature",
        PLUGIN_DIR / "scripts" / "modeling",
        PLUGIN_DIR / "scripts" / "optimizer",
        PLUGIN_DIR / "scripts" / "workflow",
        PLUGIN_DIR / "install",
        ROOT / ".codex-plugin",
        ROOT / ".agents" / "plugins",
        CLAUDE_PLUGIN_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _copy_tree(src: Path, dst: Path, *, include_names: set[str] | None = None) -> dict[str, str]:
    if not src.exists():
        raise FileNotFoundError(f"Required source path is missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            if name in EXCLUDED_DIRS or name in EXCLUDED_NAMES:
                ignored.add(name)
            elif path.suffix in EXCLUDED_SUFFIXES:
                ignored.add(name)
            elif include_names is not None and path.parent == src and name not in include_names:
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=ignore)
    return {"source": str(src), "destination": str(dst)}


def _write_plugin_files() -> list[str]:
    plugin_json = {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": PLUGIN_DESCRIPTION,
        "author": PLUGIN_AUTHOR,
        "homepage": "https://github.com/XY041216/spectral-skills",
        "repository": "https://github.com/XY041216/spectral-skills.git",
        "license": "MIT",
        "keywords": PLUGIN_KEYWORDS,
        "skills": "./skills",
        "mcpServers": "./.mcp.json",
        "interface": PLUGIN_INTERFACE,
    }
    mcp_json = {
        "mcpServers": {
            "spectral-reader": {
                "command": "python",
                "args": ["skills/spectral-reader/mcp-server/server.py"],
                "env": {"PYTHONPATH": "."},
            }
        }
    }
    root_plugin_json = {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": (
            f"{PLUGIN_DESCRIPTION} This repository root is a GitHub import "
            "entrypoint that delegates Codex components to the built "
            "plugins/spectral-skills plugin image."
        ),
        "author": PLUGIN_AUTHOR,
        "homepage": "https://github.com/XY041216/spectral-skills",
        "repository": "https://github.com/XY041216/spectral-skills.git",
        "license": "MIT",
        "keywords": PLUGIN_KEYWORDS,
        "skills": "./plugins/spectral-skills/skills",
        "mcpServers": "./.mcp.json",
        "interface": PLUGIN_INTERFACE,
    }
    root_mcp_json = {
        "mcpServers": {
            "spectral-reader": {
                "command": "python",
                "args": ["plugins/spectral-skills/skills/spectral-reader/mcp-server/server.py"],
                "env": {"PYTHONPATH": "plugins/spectral-skills"},
            }
        }
    }
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if not requirements.endswith("\n"):
        requirements += "\n"
    readme = """# Spectral Skills Plugin

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
"""
    files = {
        PLUGIN_DIR / ".codex-plugin" / "plugin.json": plugin_json,
        PLUGIN_DIR / ".mcp.json": mcp_json,
        ROOT / ".codex-plugin" / "plugin.json": root_plugin_json,
        ROOT / ".mcp.json": root_mcp_json,
    }
    written: list[str] = []
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_lf(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        written.append(str(path))
    _write_text_lf(PLUGIN_DIR / "requirements.txt", requirements)
    written.append(str(PLUGIN_DIR / "requirements.txt"))
    _write_text_lf(PLUGIN_DIR / "README.md", readme)
    written.append(str(PLUGIN_DIR / "README.md"))
    return written


def _write_marketplace() -> Path:
    path = ROOT / ".agents" / "plugins" / "marketplace.json"
    payload = {
        "name": "spectral-skills-local-marketplace",
        "interface": {"displayName": "Spectral Skills Local Marketplace"},
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "source": {"source": "local", "path": f"./plugins/{PLUGIN_NAME}"},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                "category": "Productivity",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_lf(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _write_claude_plugin_files() -> list[Path]:
    plugin_json = {
        "name": PLUGIN_NAME,
        "description": (
            "Claude-compatible and Codex-compatible spectral workflow bundles "
            "for reading raw spectral files, QC, splitting, preprocessing, "
            "feature extraction, modeling, optimizer searches, publication "
            "figures, and end-to-end workflow routing. Skills include "
            "spectral-reader, spectral-qc, spectral-splitter, "
            "spectral-preprocess, spectral-feature, spectral-modeling, "
            "spectral-optimizer, spectral-report, and spectral-workflow."
        ),
        "version": PLUGIN_VERSION,
        "author": PLUGIN_AUTHOR,
        "license": "MIT",
        "homepage": "https://github.com/XY041216/spectral-skills",
        "repository": "https://github.com/XY041216/spectral-skills.git",
        "keywords": PLUGIN_KEYWORDS,
    }
    marketplace_json = {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": "Claude-compatible spectral data processing skills for leakage-safe analysis workflows.",
        "owner": {"name": "Spectral Skills Contributors"},
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "version": PLUGIN_VERSION,
                "source": "./",
                "description": PLUGIN_DESCRIPTION,
                "author": {"name": "Spectral Skills Contributors"},
                "keywords": PLUGIN_KEYWORDS,
                "category": "research",
            }
        ],
    }
    files = {
        CLAUDE_PLUGIN_DIR / "plugin.json": plugin_json,
        CLAUDE_PLUGIN_DIR / "marketplace.json": marketplace_json,
    }
    written: list[Path] = []
    for path, payload in files.items():
        _write_text_lf(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        written.append(path)
    return written


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the spectral-skills Codex plugin image.")
    parser.add_argument("--clean", action="store_true", help="Remove the previous plugin image before rebuilding.")
    parser.add_argument("--verify", action="store_true", help="Run the plugin consistency checker after building.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON envelope.")
    args = parser.parse_args(argv)
    response = build_codex_plugin(clean=args.clean, verify=args.verify)
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

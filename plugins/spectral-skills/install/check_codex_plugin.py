"""Check the local Codex plugin image for Spectral Skills."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.reader.response import error_response, ok_response


PLUGIN_DIR = ROOT / "plugins" / "spectral-skills"
PLUGIN_NAME = "spectral-skills"
PLUGIN_VERSION = "0.1.0-beta.1"
CLAUDE_PLUGIN_DIR = ROOT / ".claude-plugin"
REPOSITORY_URL = "https://github.com/XY041216/spectral-skills.git"


def check_codex_plugin() -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    plugin_json = _load_json(PLUGIN_DIR / ".codex-plugin" / "plugin.json", checked, missing, mismatches, "plugin_json")
    mcp_json = _load_json(PLUGIN_DIR / ".mcp.json", checked, missing, mismatches, "mcp_json")
    root_plugin_json = _load_json(ROOT / ".codex-plugin" / "plugin.json", checked, missing, mismatches, "root_plugin_json")
    root_mcp_json = _load_json(ROOT / ".mcp.json", checked, missing, mismatches, "root_mcp_json")
    _load_json(ROOT / ".agents" / "plugins" / "marketplace.json", checked, missing, mismatches, "marketplace_json")
    claude_plugin_json = _load_json(CLAUDE_PLUGIN_DIR / "plugin.json", checked, missing, mismatches, "claude_plugin_json")
    claude_marketplace_json = _load_json(CLAUDE_PLUGIN_DIR / "marketplace.json", checked, missing, mismatches, "claude_marketplace_json")

    _require_dir(PLUGIN_DIR, checked, missing, "plugin_dir")
    _require_dir(PLUGIN_DIR / "skills", checked, missing, "plugin_skills_dir")
    _require_file(PLUGIN_DIR / "skills" / "spectral-reader" / "SKILL.md", checked, missing, "spectral_reader_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-reader" / "manifest.yaml", checked, missing, "spectral_reader_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-qc" / "SKILL.md", checked, missing, "spectral_qc_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-qc" / "manifest.yaml", checked, missing, "spectral_qc_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-splitter" / "SKILL.md", checked, missing, "spectral_splitter_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-splitter" / "manifest.yaml", checked, missing, "spectral_splitter_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-preprocess" / "SKILL.md", checked, missing, "spectral_preprocess_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-preprocess" / "manifest.yaml", checked, missing, "spectral_preprocess_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-feature" / "SKILL.md", checked, missing, "spectral_feature_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-feature" / "manifest.yaml", checked, missing, "spectral_feature_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-modeling" / "SKILL.md", checked, missing, "spectral_modeling_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-modeling" / "manifest.yaml", checked, missing, "spectral_modeling_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-optimizer" / "SKILL.md", checked, missing, "spectral_optimizer_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-optimizer" / "manifest.yaml", checked, missing, "spectral_optimizer_manifest")
    _require_file(
        PLUGIN_DIR / "skills" / "spectral-optimizer" / "scripts" / "optimize_spectral_pipeline.py",
        checked,
        missing,
        "spectral_optimizer_entry",
    )
    _require_file(PLUGIN_DIR / "skills" / "spectral-report" / "SKILL.md", checked, missing, "spectral_report_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-report" / "manifest.yaml", checked, missing, "spectral_report_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-report" / "references" / "style-reference-index.md", checked, missing, "spectral_report_style_index")
    _require_file(PLUGIN_DIR / "skills" / "spectral-report" / "assets" / "style-references" / "palette-reference.svg", checked, missing, "spectral_report_palette")
    _require_file(PLUGIN_DIR / "skills" / "spectral-workflow" / "SKILL.md", checked, missing, "spectral_workflow_skill")
    _require_file(PLUGIN_DIR / "skills" / "spectral-workflow" / "manifest.yaml", checked, missing, "spectral_workflow_manifest")
    _require_file(PLUGIN_DIR / "skills" / "spectral-workflow" / "static" / "core" / "route-index.md", checked, missing, "spectral_workflow_route_index")
    _require_dir(PLUGIN_DIR / "shared", checked, missing, "plugin_shared_dir")
    _require_dir(PLUGIN_DIR / "shared" / "schemas", checked, missing, "plugin_shared_schemas")
    _require_dir(PLUGIN_DIR / "spectral_core" / "reader", checked, missing, "plugin_spectral_core_reader")
    _require_dir(PLUGIN_DIR / "spectral_core" / "qc", checked, missing, "plugin_spectral_core_qc")
    _require_dir(PLUGIN_DIR / "spectral_core" / "splitter", checked, missing, "plugin_spectral_core_splitter")
    _require_dir(PLUGIN_DIR / "spectral_core" / "preprocess", checked, missing, "plugin_spectral_core_preprocess")
    _require_dir(PLUGIN_DIR / "spectral_core" / "feature", checked, missing, "plugin_spectral_core_feature")
    _require_dir(PLUGIN_DIR / "spectral_core" / "modeling", checked, missing, "plugin_spectral_core_modeling")
    _require_dir(PLUGIN_DIR / "spectral_core" / "optimizer", checked, missing, "plugin_spectral_core_optimizer")
    _require_dir(PLUGIN_DIR / "spectral_core" / "workflow", checked, missing, "plugin_spectral_core_workflow")
    _require_dir(PLUGIN_DIR / "scripts" / "reader", checked, missing, "plugin_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "qc", checked, missing, "plugin_qc_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "splitter", checked, missing, "plugin_splitter_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "preprocess", checked, missing, "plugin_preprocess_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "feature", checked, missing, "plugin_feature_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "modeling", checked, missing, "plugin_modeling_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "optimizer", checked, missing, "plugin_optimizer_fallback_scripts")
    _require_dir(PLUGIN_DIR / "scripts" / "workflow", checked, missing, "plugin_workflow_fallback_scripts")
    _require_file(PLUGIN_DIR / "install" / "check_codex_config.py", checked, missing, "plugin_codex_config_preflight")
    _require_file(PLUGIN_DIR / "install" / "install_codex_plugin.py", checked, missing, "plugin_codex_desktop_installer")
    _require_file(PLUGIN_DIR / "README.md", checked, missing, "plugin_readme")
    _require_file(PLUGIN_DIR / "requirements.txt", checked, missing, "plugin_requirements")

    if (PLUGIN_DIR / "skills" / "_shared").exists():
        mismatches.append(_issue("SHARED_EXPOSED_AS_SKILL", "plugins/spectral-skills/skills/_shared must not exist.", severity="error"))
    else:
        checked.append(_ok("shared_not_in_skills", str(PLUGIN_DIR / "skills" / "_shared")))
    if (PLUGIN_DIR / "shared" / "SKILL.md").exists():
        mismatches.append(_issue("SHARED_SKILL_MD_PRESENT", "plugin shared/ must not contain SKILL.md.", severity="error"))
    else:
        checked.append(_ok("shared_skill_absent", str(PLUGIN_DIR / "shared" / "SKILL.md")))

    _check_plugin_json(plugin_json, checked, mismatches)
    _check_root_plugin_json(root_plugin_json, checked, mismatches)
    _check_claude_plugin_json(claude_plugin_json, checked, mismatches)
    _check_claude_marketplace_json(claude_marketplace_json, checked, mismatches)
    _check_mcp_json(mcp_json, checked, mismatches)
    _check_root_mcp_json(root_mcp_json, checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-reader" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-qc" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-splitter" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-preprocess" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-feature" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-modeling" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-optimizer" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-report" / "manifest.yaml", checked, mismatches)
    _check_yaml(PLUGIN_DIR / "skills" / "spectral-workflow" / "manifest.yaml", checked, mismatches)
    _check_manifest_method_scopes(checked, mismatches)
    _check_schema_mirrors(checked, mismatches)
    _check_standalone_core_skill(checked, missing, mismatches)
    _check_source_mirrors(checked, mismatches)
    _check_no_excluded_artifacts(PLUGIN_DIR, checked, mismatches)
    _run_codex_config_preflight_selftest(checked, mismatches, warnings)
    _run_codex_desktop_install_selftest(checked, mismatches, warnings)
    _run_plugin_script(["skills/spectral-reader/scripts/server_health.py", "--json"], "plugin_server_health", checked, mismatches, warnings)
    _run_plugin_script(["skills/spectral-reader/scripts/check_consistency.py", "--json"], "plugin_skill_consistency", checked, mismatches, warnings)
    _run_plugin_script(["skills/spectral-qc/scripts/qc_spectral_package.py", "--mode", "methods", "--json"], "plugin_qc_methods", checked, mismatches, warnings)
    _run_plugin_splitter_smoke(checked, mismatches, warnings)
    _run_plugin_preprocess_smoke(checked, mismatches, warnings)
    _run_plugin_feature_smoke(checked, mismatches, warnings)
    _run_plugin_modeling_smoke(checked, mismatches, warnings)
    _run_plugin_optimizer_smoke(checked, mismatches, warnings)
    _run_plugin_workflow_smoke(checked, mismatches, warnings)
    _remove_empty_dir(PLUGIN_DIR / "assets")

    status = "failed" if missing or any(item.get("severity") == "error" for item in mismatches) else "degraded" if mismatches or warnings else "passed"
    result = {
        "plugin_consistency_status": status,
        "plugin_dir": str(PLUGIN_DIR),
        "checked_items": checked,
        "missing_items": missing,
        "mismatches": mismatches,
        "warnings": warnings,
        "recommendations": [],
    }
    if status == "failed":
        return error_response("check_codex_plugin", "Codex plugin image consistency failed.", backend="script", code="PLUGIN_CONSISTENCY_FAILED", result=result, warnings=warnings)
    return ok_response("check_codex_plugin", result, backend="script", warnings=warnings)


def _load_json(path: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], mismatches: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not path.exists():
        missing.append(_item(key, str(path), "Required JSON file is missing."))
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checked.append(_ok(key, str(path)))
        return payload
    except Exception as exc:
        mismatches.append(_issue("JSON_INVALID", "JSON file could not be parsed.", severity="error", path=str(path), error=str(exc)))
        return {}


def _check_plugin_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    if payload.get("name") != PLUGIN_NAME:
        mismatches.append(_issue("PLUGIN_NAME_MISMATCH", "plugin.json name must be spectral-skills.", severity="error", observed=payload.get("name")))
    else:
        checked.append(_ok("plugin_name", PLUGIN_NAME))
    if payload.get("version") != PLUGIN_VERSION:
        mismatches.append(
            _issue(
                "PLUGIN_VERSION_MISMATCH",
                "plugin.json version must match the current release version.",
                severity="error",
                expected=PLUGIN_VERSION,
                observed=payload.get("version"),
            )
        )
    else:
        checked.append(_ok("plugin_version", PLUGIN_VERSION))
    if payload.get("skills") not in {"./skills", "./skills/"}:
        mismatches.append(_issue("PLUGIN_SKILLS_PATH_MISMATCH", "plugin.json skills path must point to ./skills.", severity="error", observed=payload.get("skills")))
    else:
        checked.append(_ok("plugin_skills_path", str(payload.get("skills"))))
    if payload.get("mcpServers") != "./.mcp.json":
        mismatches.append(_issue("PLUGIN_MCP_DECLARATION_MISSING", "plugin.json must declare mcpServers as ./.mcp.json.", severity="error", observed=payload.get("mcpServers")))
    else:
        checked.append(_ok("plugin_mcp_servers_path", "./.mcp.json"))
    author = payload.get("author") or {}
    if author.get("name") != "Spectral Skills Contributors":
        mismatches.append(_issue("PLUGIN_AUTHOR_MISSING", "plugin.json must include the Spectral Skills author object.", severity="error", observed=author))
    else:
        checked.append(_ok("plugin_author", author["name"]))
    if payload.get("repository") != REPOSITORY_URL:
        mismatches.append(_issue("PLUGIN_REPOSITORY_MISMATCH", "plugin.json repository must point to the public GitHub repository.", severity="error", expected=REPOSITORY_URL, observed=payload.get("repository")))
    else:
        checked.append(_ok("plugin_repository", REPOSITORY_URL))
    interface = payload.get("interface") or {}
    required_interface = ["displayName", "shortDescription", "longDescription", "developerName", "category", "capabilities", "defaultPrompt"]
    missing_interface = [key for key in required_interface if not interface.get(key)]
    if missing_interface:
        mismatches.append(_issue("PLUGIN_INTERFACE_INCOMPLETE", "plugin.json interface metadata is incomplete.", severity="error", missing=missing_interface))
    else:
        checked.append(_ok("plugin_interface", interface.get("displayName", "")))


def _check_root_plugin_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    if payload.get("name") != PLUGIN_NAME:
        mismatches.append(_issue("ROOT_PLUGIN_NAME_MISMATCH", "Root .codex-plugin/plugin.json name must be spectral-skills.", severity="error", observed=payload.get("name")))
    else:
        checked.append(_ok("root_plugin_name", PLUGIN_NAME))
    if payload.get("version") != PLUGIN_VERSION:
        mismatches.append(_issue("ROOT_PLUGIN_VERSION_MISMATCH", "Root plugin version must match the current release version.", severity="error", expected=PLUGIN_VERSION, observed=payload.get("version")))
    else:
        checked.append(_ok("root_plugin_version", PLUGIN_VERSION))
    if payload.get("skills") != "./plugins/spectral-skills/skills":
        mismatches.append(
            _issue(
                "ROOT_PLUGIN_SKILLS_PATH_MISMATCH",
                "Root plugin entrypoint must delegate skills to the built plugin image.",
                severity="error",
                observed=payload.get("skills"),
            )
        )
    else:
        checked.append(_ok("root_plugin_skills_path", "./plugins/spectral-skills/skills"))
    if payload.get("mcpServers") != "./.mcp.json":
        mismatches.append(_issue("ROOT_PLUGIN_MCP_DECLARATION_MISSING", "Root plugin entrypoint must declare mcpServers as ./.mcp.json.", severity="error", observed=payload.get("mcpServers")))
    else:
        checked.append(_ok("root_plugin_mcp_servers_path", "./.mcp.json"))
    interface = payload.get("interface") or {}
    if interface.get("displayName") != "Spectral Skills":
        mismatches.append(_issue("ROOT_PLUGIN_INTERFACE_MISMATCH", "Root plugin entrypoint must expose the Spectral Skills interface metadata.", severity="error", observed=interface))
    else:
        checked.append(_ok("root_plugin_interface", interface["displayName"]))


def _check_claude_plugin_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    if payload.get("name") != PLUGIN_NAME:
        mismatches.append(_issue("CLAUDE_PLUGIN_NAME_MISMATCH", ".claude-plugin/plugin.json name must be spectral-skills.", severity="error", observed=payload.get("name")))
    else:
        checked.append(_ok("claude_plugin_name", PLUGIN_NAME))
    if payload.get("version") != PLUGIN_VERSION:
        mismatches.append(_issue("CLAUDE_PLUGIN_VERSION_MISMATCH", ".claude-plugin/plugin.json version must match the current release version.", severity="error", expected=PLUGIN_VERSION, observed=payload.get("version")))
    else:
        checked.append(_ok("claude_plugin_version", PLUGIN_VERSION))
    keywords = set(payload.get("keywords") or [])
    if not {"spectral", "chemometrics"} <= keywords:
        mismatches.append(_issue("CLAUDE_PLUGIN_KEYWORDS_INCOMPLETE", ".claude-plugin/plugin.json should expose spectral/chemometrics keywords.", severity="error", observed=sorted(keywords)))
    else:
        checked.append(_ok("claude_plugin_keywords", ",".join(sorted(keywords))))


def _check_claude_marketplace_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    plugins = payload.get("plugins") or []
    if not plugins:
        mismatches.append(_issue("CLAUDE_MARKETPLACE_EMPTY", ".claude-plugin/marketplace.json must list spectral-skills.", severity="error"))
        return
    entry = plugins[0]
    if entry.get("name") != PLUGIN_NAME:
        mismatches.append(_issue("CLAUDE_MARKETPLACE_PLUGIN_NAME_MISMATCH", "Claude marketplace plugin name must be spectral-skills.", severity="error", observed=entry.get("name")))
    else:
        checked.append(_ok("claude_marketplace_plugin_name", PLUGIN_NAME))
    if entry.get("version") != PLUGIN_VERSION:
        mismatches.append(_issue("CLAUDE_MARKETPLACE_VERSION_MISMATCH", "Claude marketplace plugin version must match the current release version.", severity="error", expected=PLUGIN_VERSION, observed=entry.get("version")))
    else:
        checked.append(_ok("claude_marketplace_version", PLUGIN_VERSION))
    if entry.get("source") != "./":
        mismatches.append(_issue("CLAUDE_MARKETPLACE_SOURCE_MISMATCH", "Claude marketplace source should point at the repository/plugin bundle root.", severity="error", expected="./", observed=entry.get("source")))
    else:
        checked.append(_ok("claude_marketplace_source", "./"))


def _check_mcp_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    server = ((payload.get("mcpServers") or {}).get("spectral-reader") or {})
    command = server.get("command")
    if command != "python":
        mismatches.append(_issue("MCP_COMMAND_MISMATCH", ".mcp.json must use generic python command.", severity="error", observed=command))
    else:
        checked.append(_ok("mcp_command", command))
    args = server.get("args") or []
    if args != ["skills/spectral-reader/mcp-server/server.py"]:
        mismatches.append(_issue("MCP_ARGS_MISMATCH", ".mcp.json args must point to plugin-local server.py.", severity="error", observed=args))
    else:
        checked.append(_ok("mcp_args", ",".join(args)))
    env = server.get("env") or {}
    if env.get("PYTHONPATH") != ".":
        mismatches.append(_issue("MCP_PYTHONPATH_MISMATCH", ".mcp.json must set PYTHONPATH to '.'.", severity="error", observed=env.get("PYTHONPATH")))
    else:
        checked.append(_ok("mcp_pythonpath", "."))
    if command and (":" in command or "\\" in command or "/" in command):
        mismatches.append(_issue("MCP_ABSOLUTE_PYTHON_PATH", ".mcp.json must not hard-code a machine Python path.", severity="error", command=command))


def _check_root_mcp_json(payload: dict[str, Any], checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    server = ((payload.get("mcpServers") or {}).get("spectral-reader") or {})
    command = server.get("command")
    if command != "python":
        mismatches.append(_issue("ROOT_MCP_COMMAND_MISMATCH", "Root .mcp.json must use generic python command.", severity="error", observed=command))
    else:
        checked.append(_ok("root_mcp_command", command))
    args = server.get("args") or []
    expected_args = ["plugins/spectral-skills/skills/spectral-reader/mcp-server/server.py"]
    if args != expected_args:
        mismatches.append(_issue("ROOT_MCP_ARGS_MISMATCH", "Root .mcp.json args must point to the built plugin image server.py.", severity="error", expected=expected_args, observed=args))
    else:
        checked.append(_ok("root_mcp_args", ",".join(args)))
    env = server.get("env") or {}
    if env.get("PYTHONPATH") != "plugins/spectral-skills":
        mismatches.append(_issue("ROOT_MCP_PYTHONPATH_MISMATCH", "Root .mcp.json must set PYTHONPATH to the built plugin image.", severity="error", observed=env.get("PYTHONPATH")))
    else:
        checked.append(_ok("root_mcp_pythonpath", "plugins/spectral-skills"))


def _check_yaml(path: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    try:
        import yaml

        yaml.safe_load(path.read_text(encoding="utf-8"))
        checked.append(_ok("manifest_yaml", str(path)))
    except Exception as exc:
        mismatches.append(_issue("MANIFEST_YAML_INVALID", "Plugin manifest.yaml could not be parsed.", severity="error", error=str(exc)))


def _check_no_excluded_artifacts(root: Path, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    hits = []
    content_roots = ["skills", "shared", "spectral_core", "scripts", "install", "assets"]
    for name in content_roots:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            parts = set(path.relative_to(root).parts)
            if {"outputs", "__pycache__", ".pytest_cache", "tests", "fixtures", "evals"} & parts:
                hits.append(str(path))
    if hits:
        mismatches.append(_issue("EXCLUDED_ARTIFACT_COPIED", "Plugin image contains excluded outputs/cache artifacts.", severity="error", hits=hits[:20], total=len(hits)))
    else:
        checked.append(_ok("excluded_artifacts_absent", str(root)))


def _check_manifest_method_scopes(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    """Ensure duplicated human-facing manifests cannot drift from runtime IDs."""

    try:
        import yaml

        from spectral_core.feature.methods import SUPPORTED_METHODS as FEATURE_METHODS
        from spectral_core.modeling.registry import CLASSIFICATION_MODELS, REGRESSION_MODELS
        from spectral_core.preprocess.methods import SUPPORTED_METHODS as PREPROCESS_METHODS
        from spectral_core.splitter.methods import SUPPORTED_METHODS as SPLIT_METHODS

        manifests = {
            name: yaml.safe_load((PLUGIN_DIR / "skills" / name / "manifest.yaml").read_text(encoding="utf-8"))
            for name in ["spectral-preprocess", "spectral-splitter", "spectral-feature", "spectral-modeling", "spectral-workflow"]
        }
        comparisons = {
            "preprocess_methods": (
                set(PREPROCESS_METHODS),
                set(manifests["spectral-preprocess"]["preprocess_scope"]["methods"]),
            ),
            "split_methods": (
                set(SPLIT_METHODS),
                set(manifests["spectral-splitter"]["split_scope"]["methods"]),
            ),
            "feature_methods": (
                set(FEATURE_METHODS),
                set(manifests["spectral-feature"]["feature_scope"]["methods"]),
            ),
        }

        model_scope = manifests["spectral-modeling"]["model_scope"]
        optional = set(model_scope.get("optional_dependency_models") or [])
        experimental = set(model_scope.get("experimental_small_sample_models") or [])
        manifest_classifiers = set(model_scope.get("classification_models") or []) | {
            name for name in optional | experimental if name.endswith("_classifier") or name.endswith("_embedding_svm")
        }
        manifest_regressors = set(model_scope.get("regression_models") or []) | {
            name for name in optional | experimental if name.endswith("_regressor")
        }
        comparisons["classification_models"] = (set(CLASSIFICATION_MODELS), manifest_classifiers)
        comparisons["regression_models"] = (set(REGRESSION_MODELS), manifest_regressors)

        for key, (runtime_values, manifest_values) in comparisons.items():
            if runtime_values != manifest_values:
                mismatches.append(
                    _issue(
                        "MANIFEST_RUNTIME_SCOPE_MISMATCH",
                        "A skill manifest method scope differs from runtime code.",
                        severity="error",
                        scope=key,
                        manifest_only=sorted(manifest_values - runtime_values),
                        runtime_only=sorted(runtime_values - manifest_values),
                    )
                )
            else:
                checked.append(_ok("manifest_runtime_scope", key))

        workflow_goals = set(manifests["spectral-workflow"]["workflow_scope"]["cli_supported_goals"])
        from spectral_core.workflow.workflow import SUPPORTED_GOALS

        if workflow_goals != set(SUPPORTED_GOALS):
            mismatches.append(
                _issue(
                    "WORKFLOW_GOAL_SCOPE_MISMATCH",
                    "Workflow manifest CLI goals differ from runtime code.",
                    severity="error",
                    manifest_only=sorted(workflow_goals - set(SUPPORTED_GOALS)),
                    runtime_only=sorted(set(SUPPORTED_GOALS) - workflow_goals),
                )
            )
        else:
            checked.append(_ok("workflow_cli_goal_scope", "workflow manifest"))
    except Exception as exc:
        mismatches.append(
            _issue(
                "MANIFEST_RUNTIME_SCOPE_CHECK_FAILED",
                "Could not compare manifest scopes with runtime registries.",
                severity="error",
                error=str(exc),
            )
        )


def _check_source_mirrors(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    """Block release when generated plugin content differs from source."""

    excluded_dirs = {"tests", "fixtures", "evals", "__pycache__", ".pytest_cache", "outputs", ".mypy_cache", ".ruff_cache"}
    mappings = [
        (ROOT / "skills" / name, PLUGIN_DIR / "skills" / name, f"skill:{name}")
        for name in [
            "spectral-reader",
            "spectral-qc",
            "spectral-splitter",
            "spectral-preprocess",
            "spectral-feature",
            "spectral-modeling",
            "spectral-optimizer",
            "spectral-report",
            "spectral-workflow",
        ]
    ]
    mappings.extend(
        [
            (ROOT / "skills" / "_shared", PLUGIN_DIR / "shared", "shared"),
            (ROOT / "spectral_core", PLUGIN_DIR / "spectral_core", "spectral_core"),
            (ROOT / "scripts", PLUGIN_DIR / "scripts", "scripts"),
            (ROOT / "install", PLUGIN_DIR / "install", "install"),
        ]
    )

    for source, mirror, key in mappings:
        source_files = _release_file_map(source, excluded_dirs)
        mirror_files = _release_file_map(mirror, excluded_dirs)
        if source_files != mirror_files:
            source_paths = set(source_files)
            mirror_paths = set(mirror_files)
            changed = sorted(path for path in source_paths & mirror_paths if source_files[path] != mirror_files[path])
            mismatches.append(
                _issue(
                    "PLUGIN_SOURCE_MIRROR_MISMATCH",
                    "Generated plugin content differs from development source.",
                    severity="error",
                    scope=key,
                    source_only=sorted(source_paths - mirror_paths)[:20],
                    mirror_only=sorted(mirror_paths - source_paths)[:20],
                    changed=changed[:20],
                )
            )
        else:
            checked.append(_ok("plugin_source_mirror", key))


def _check_schema_mirrors(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    """Treat shared schemas as checked mirrors of runtime schemas."""

    shared = ROOT / "skills" / "_shared" / "schemas"
    runtime = ROOT / "spectral_core" / "schemas"
    differences: list[str] = []
    missing: list[str] = []
    for shared_path in sorted(shared.glob("*.json")):
        runtime_path = runtime / shared_path.name
        if not runtime_path.exists():
            missing.append(shared_path.name)
        elif shared_path.read_bytes() != runtime_path.read_bytes():
            differences.append(shared_path.name)
    if missing or differences:
        mismatches.append(
            _issue(
                "SHARED_RUNTIME_SCHEMA_MISMATCH",
                "Shared contract schemas differ from their runtime mirrors.",
                severity="error",
                missing_runtime=missing,
                different=differences,
            )
        )
    else:
        checked.append(_ok("shared_runtime_schema_mirrors", str(shared)))


def _check_standalone_core_skill(
    checked: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
) -> None:
    """Ensure direct GitHub skill installs can get the shared runtime."""

    skill_dir = ROOT / "skills" / "spectral-core"
    runtime_mirror = skill_dir / "spectral_core"
    _require_file(skill_dir / "SKILL.md", checked, missing, "standalone_spectral_core_skill")
    _require_file(runtime_mirror / "__init__.py", checked, missing, "standalone_spectral_core_runtime")
    source_files = _release_file_map(ROOT / "spectral_core", {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
    mirror_files = _release_file_map(runtime_mirror, {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
    if source_files != mirror_files:
        source_paths = set(source_files)
        mirror_paths = set(mirror_files)
        changed = sorted(path for path in source_paths & mirror_paths if source_files[path] != mirror_files[path])
        mismatches.append(
            _issue(
                "STANDALONE_CORE_RUNTIME_MISMATCH",
                "skills/spectral-core/spectral_core must mirror the root spectral_core runtime for direct GitHub skill installs.",
                severity="error",
                source_only=sorted(source_paths - mirror_paths)[:20],
                mirror_only=sorted(mirror_paths - source_paths)[:20],
                changed=changed[:20],
            )
        )
    else:
        checked.append(_ok("standalone_core_runtime_mirror", str(runtime_mirror)))


def _release_file_map(root: Path, excluded_dirs: set[str]) -> dict[str, bytes]:
    if not root.exists():
        return {}
    result: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if excluded_dirs & set(relative.parts) or path.suffix in {".pyc", ".pyo", ".zip"}:
            continue
        result[relative.as_posix()] = path.read_bytes()
    return result


def _run_plugin_script(args: list[str], key: str, checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    script_args = [sys.executable, *args]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(script_args, cwd=PLUGIN_DIR, capture_output=True, text=True, timeout=60, env=env)
        payload = json.loads(completed.stdout)
        if completed.returncode != 0 or not payload.get("ok"):
            mismatches.append(_issue("PLUGIN_SCRIPT_FAILED", "Plugin script returned non-ok.", severity="error", key=key, returncode=completed.returncode, stderr=completed.stderr, errors=payload.get("errors", [])))
        else:
            checked.append(_ok(key, " ".join(args)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_SCRIPT_EXCEPTION", "Plugin script could not run.", severity="error", key=key, error=str(exc)))


def _run_codex_config_preflight_selftest(
    checked: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    test_dir = PLUGIN_DIR / "assets" / "_check_codex_config"
    valid_config = test_dir / "valid.toml"
    invalid_config = test_dir / "invalid.toml"
    script = PLUGIN_DIR / "install" / "check_codex_config.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        valid_config.write_text(
            "[marketplaces.spectral-skills-local-marketplace]\n"
            "source_type = \"local\"\n"
            "source = 'C:\\\\path\\\\to\\\\spectral-skills'\n",
            encoding="utf-8",
        )
        invalid_config.write_text(
            "[projects.'C:\\\\bad\\\\truncated]\n"
            "trust_level = \"trusted\"\n",
            encoding="utf-8",
        )
        valid = subprocess.run(
            [sys.executable, str(script), "--config", str(valid_config), "--json"],
            cwd=PLUGIN_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        invalid = subprocess.run(
            [sys.executable, str(script), "--config", str(invalid_config), "--json"],
            cwd=PLUGIN_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        valid_payload = json.loads(valid.stdout)
        invalid_payload = json.loads(invalid.stdout)
        invalid_codes = {item.get("code") for item in invalid_payload.get("errors", [])}
        if valid.returncode != 0 or not valid_payload.get("ok"):
            mismatches.append(_issue("CODEX_CONFIG_PREFLIGHT_VALID_FAILED", "Codex config preflight rejected valid TOML.", severity="error", stdout=valid.stdout, stderr=valid.stderr))
        elif invalid.returncode == 0 or invalid_payload.get("ok") or "CODEX_CONFIG_TOML_INVALID" not in invalid_codes:
            mismatches.append(_issue("CODEX_CONFIG_PREFLIGHT_INVALID_FAILED", "Codex config preflight did not detect malformed TOML.", severity="error", stdout=invalid.stdout, stderr=invalid.stderr))
        else:
            checked.append(_ok("codex_config_preflight_selftest", str(script)))
    except Exception as exc:
        mismatches.append(_issue("CODEX_CONFIG_PREFLIGHT_EXCEPTION", "Codex config preflight self-test could not run.", severity="error", error=str(exc)))
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def _run_codex_desktop_install_selftest(
    checked: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    script = ROOT / "install" / "install_codex_plugin.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="spectral-codex-home-") as temp_dir:
        test_home = Path(temp_dir)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(ROOT),
                    "--codex-home",
                    str(test_home),
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            payload = json.loads(completed.stdout)
            result = payload.get("result", {})
            cache_path = Path(result.get("cache", {}).get("path", ""))
            config_path = test_home / "config.toml"
            if completed.returncode != 0 or not payload.get("ok"):
                mismatches.append(_issue("CODEX_DESKTOP_INSTALL_FAILED", "Codex Desktop installer rejected a sandbox install.", severity="error", stdout=completed.stdout, stderr=completed.stderr))
            elif not config_path.exists():
                mismatches.append(_issue("CODEX_DESKTOP_INSTALL_CONFIG_MISSING", "Codex Desktop installer did not write config.toml in the sandbox.", severity="error", path=str(config_path)))
            elif not (cache_path / ".codex-plugin" / "plugin.json").exists():
                mismatches.append(_issue("CODEX_DESKTOP_INSTALL_CACHE_MISSING", "Codex Desktop installer did not materialize the plugin cache.", severity="error", path=str(cache_path)))
            elif not (cache_path / "skills" / "spectral-workflow" / "SKILL.md").exists():
                mismatches.append(_issue("CODEX_DESKTOP_INSTALL_SKILL_MISSING", "Codex Desktop installer cache is missing spectral-workflow.", severity="error", path=str(cache_path)))
            else:
                checked.append(_ok("codex_desktop_install_selftest", str(script)))
        except Exception as exc:
            mismatches.append(_issue("CODEX_DESKTOP_INSTALL_EXCEPTION", "Codex Desktop install self-test could not run.", severity="error", error=str(exc)))


def _remove_empty_dir(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _run_plugin_splitter_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    package_dir = PLUGIN_DIR / "assets" / "_check_splitter_package"
    output_dir = PLUGIN_DIR / "assets" / "_check_splitter_output"
    try:
        _write_splitter_smoke_package(package_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-splitter/scripts/split_spectral_package.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(output_dir),
                "--method",
                "random",
                "--ratio",
                "8:2",
                "--json",
            ],
            "plugin_splitter_random_split",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "split_contract.json").exists():
            mismatches.append(_issue("PLUGIN_SPLITTER_CONTRACT_MISSING", "Plugin splitter smoke did not write split_contract.json.", severity="error", output_dir=str(output_dir)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_SPLITTER_SMOKE_EXCEPTION", "Plugin splitter smoke could not run.", severity="error", error=str(exc)))
    finally:
        for path in [package_dir, output_dir]:
            if path.exists():
                shutil.rmtree(path)


def _run_plugin_preprocess_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    package_dir = PLUGIN_DIR / "assets" / "_check_preprocess_package"
    split_dir = PLUGIN_DIR / "assets" / "_check_preprocess_split"
    output_dir = PLUGIN_DIR / "assets" / "_check_preprocess_output"
    try:
        _write_preprocess_smoke_package(package_dir)
        _write_preprocess_smoke_split(split_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-preprocess/scripts/preprocess_spectral_package.py",
                "--package-dir",
                str(package_dir),
                "--split-contract",
                str(split_dir / "split_contract.json"),
                "--output-dir",
                str(output_dir),
                "--methods",
                "snv",
                "--json",
            ],
            "plugin_preprocess_snv",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "preprocess_state.json").exists():
            mismatches.append(_issue("PLUGIN_PREPROCESS_STATE_MISSING", "Plugin preprocess smoke did not write preprocess_state.json.", severity="error", output_dir=str(output_dir)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_PREPROCESS_SMOKE_EXCEPTION", "Plugin preprocess smoke could not run.", severity="error", error=str(exc)))
    finally:
        for path in [package_dir, split_dir, output_dir]:
            if path.exists():
                shutil.rmtree(path)


def _run_plugin_feature_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    package_dir = PLUGIN_DIR / "assets" / "_check_feature_package"
    split_dir = PLUGIN_DIR / "assets" / "_check_feature_split"
    output_dir = PLUGIN_DIR / "assets" / "_check_feature_output"
    try:
        _write_feature_smoke_package(package_dir)
        _write_preprocess_smoke_split(split_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-feature/scripts/feature_spectral_package.py",
                "--package-dir",
                str(package_dir),
                "--split-contract",
                str(split_dir / "split_contract.json"),
                "--output-dir",
                str(output_dir),
                "--method",
                "variance_threshold",
                "--json",
            ],
            "plugin_feature_variance_threshold",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "feature_state.json").exists():
            mismatches.append(_issue("PLUGIN_FEATURE_STATE_MISSING", "Plugin feature smoke did not write feature_state.json.", severity="error", output_dir=str(output_dir)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_FEATURE_SMOKE_EXCEPTION", "Plugin feature smoke could not run.", severity="error", error=str(exc)))
    finally:
        for path in [package_dir, split_dir, output_dir]:
            if path.exists():
                shutil.rmtree(path)


def _run_plugin_modeling_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    package_dir = PLUGIN_DIR / "assets" / "_check_modeling_package"
    split_dir = PLUGIN_DIR / "assets" / "_check_modeling_split"
    output_dir = PLUGIN_DIR / "assets" / "_check_modeling_output"
    try:
        _write_modeling_smoke_package(package_dir)
        _write_modeling_smoke_split(split_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-modeling/scripts/model_spectral_package.py",
                "--package-dir",
                str(package_dir),
                "--split-contract",
                str(split_dir / "split_contract.json"),
                "--output-dir",
                str(output_dir),
                "--task-type",
                "classification",
                "--models",
                "random_forest_classifier",
                "--json",
            ],
            "plugin_modeling_classification",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "modeling_contract.json").exists():
            mismatches.append(_issue("PLUGIN_MODELING_CONTRACT_MISSING", "Plugin modeling smoke did not write modeling_contract.json.", severity="error", output_dir=str(output_dir)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_MODELING_SMOKE_EXCEPTION", "Plugin modeling smoke could not run.", severity="error", error=str(exc)))
    finally:
        for path in [package_dir, split_dir, output_dir]:
            if path.exists():
                shutil.rmtree(path)


def _run_plugin_optimizer_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    output_dir = PLUGIN_DIR / "assets" / "_check_optimizer_output"
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-optimizer/scripts/optimize_spectral_pipeline.py",
                "--mode",
                "recommend_from_profile",
                "--task-type",
                "classification",
                "--n-samples",
                "120",
                "--n-features",
                "3401",
                "--output-dir",
                str(output_dir),
                "--json",
            ],
            "plugin_optimizer_recommendation",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "optimizer_contract.json").exists():
            mismatches.append(
                _issue(
                    "PLUGIN_OPTIMIZER_CONTRACT_MISSING",
                    "Plugin optimizer smoke did not write optimizer_contract.json.",
                    severity="error",
                    output_dir=str(output_dir),
                )
            )
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_OPTIMIZER_SMOKE_EXCEPTION", "Plugin optimizer smoke could not run.", severity="error", error=str(exc)))
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def _run_plugin_workflow_smoke(checked: list[dict[str, Any]], mismatches: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    package_dir = PLUGIN_DIR / "assets" / "_check_workflow_package"
    output_dir = PLUGIN_DIR / "assets" / "_check_workflow_output"
    try:
        _write_modeling_smoke_package(package_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_plugin_script(
            [
                "skills/spectral-workflow/scripts/run_spectral_workflow.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(output_dir),
                "--task-goal",
                "classification",
                "--split-ratio",
                "6:2:2",
                "--split-method",
                "random",
                "--preprocess-methods",
                "none",
                "--feature-method",
                "none",
                "--models",
                "random_forest_classifier",
                "--json",
            ],
            "plugin_workflow_classification",
            checked,
            mismatches,
            warnings,
        )
        if not (output_dir / "workflow_result.json").exists():
            mismatches.append(_issue("PLUGIN_WORKFLOW_RESULT_MISSING", "Plugin workflow smoke did not write workflow_result.json.", severity="error", output_dir=str(output_dir)))
    except Exception as exc:
        mismatches.append(_issue("PLUGIN_WORKFLOW_SMOKE_EXCEPTION", "Plugin workflow smoke could not run.", severity="error", error=str(exc)))
    finally:
        for path in [package_dir, output_dir]:
            if path.exists():
                shutil.rmtree(path)


def _write_splitter_smoke_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(package_dir / "X.csv", [["900", "1000"], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
    _write_csv(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"]])
    _write_csv(package_dir / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"]])
    contract = {
        "contract_id": "plugin-splitter-smoke",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": None, "metadata": None},
        "shape": {"n_samples": 5, "n_features": 2},
        "task_hint": "unsupervised",
    }
    (package_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_preprocess_smoke_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(package_dir / "X.csv", [["900", "1000", "1100"], [1, 2, 3], [2, 3, 4], [4, 5, 6], [5, 6, 7]])
    _write_csv(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"]])
    _write_csv(package_dir / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"]])
    contract = {
        "contract_id": "plugin-preprocess-smoke",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": None, "metadata": None},
        "shape": {"n_samples": 4, "n_features": 3},
        "task_hint": "unsupervised",
    }
    (package_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_feature_smoke_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(package_dir / "X.csv", [["900", "1000", "1100"], [1, 2, 5], [2, 2, 6], [4, 2, 8], [5, 2, 9]])
    _write_csv(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"]])
    _write_csv(package_dir / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"]])
    contract = {
        "contract_id": "plugin-feature-smoke",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": None, "metadata": None},
        "shape": {"n_samples": 4, "n_features": 3},
        "task_hint": "unsupervised",
    }
    (package_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_modeling_smoke_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(package_dir / "X.csv", [["900", "1000", "1100"], [1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 11, 12], [11, 12, 13], [12, 13, 14]])
    _write_csv(package_dir / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"]])
    _write_csv(package_dir / "band_axis.csv", [["index", "value", "unit"], [0, 900, "cm-1"], [1, 1000, "cm-1"], [2, 1100, "cm-1"]])
    _write_csv(package_dir / "y.csv", [["class"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"]])
    contract = {
        "contract_id": "plugin-modeling-smoke",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 6, "n_features": 3},
        "task_hint": "classification",
    }
    (package_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_modeling_smoke_split(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        split_dir / "split_indices.csv",
        [
            ["sample_id", "index", "split"],
            ["S001", 0, "train"],
            ["S002", 1, "train"],
            ["S004", 3, "train"],
            ["S005", 4, "train"],
            ["S003", 2, "val"],
            ["S006", 5, "test"],
        ],
    )
    contract = {
        "contract_type": "split_contract",
        "contract_id": "plugin-modeling-split",
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 6, "train": 4, "val": 1, "test": 1},
    }
    (split_dir / "split_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_preprocess_smoke_split(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        split_dir / "split_indices.csv",
        [
            ["sample_id", "index", "split"],
            ["S001", 0, "train"],
            ["S002", 1, "train"],
            ["S003", 2, "val"],
            ["S004", 3, "test"],
        ],
    )
    contract = {
        "contract_type": "split_contract",
        "contract_id": "plugin-preprocess-split",
        "split_files": {"split_indices": "split_indices.csv"},
        "n_samples": {"total": 4, "train": 2, "val": 1, "test": 1},
    }
    (split_dir / "split_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _require_file(path: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], key: str) -> None:
    if path.exists() and path.is_file():
        checked.append(_ok(key, str(path)))
    else:
        missing.append(_item(key, str(path), "Required file is missing."))


def _require_dir(path: Path, checked: list[dict[str, Any]], missing: list[dict[str, Any]], key: str) -> None:
    if path.exists() and path.is_dir():
        checked.append(_ok(key, str(path)))
    else:
        missing.append(_item(key, str(path), "Required directory is missing."))


def _ok(key: str, path: str) -> dict[str, Any]:
    return {"key": key, "status": "ok", "path": path}


def _item(key: str, path: str, message: str) -> dict[str, Any]:
    return {"key": key, "path": path, "message": message}


def _issue(code: str, message: str, *, severity: str = "warning", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "severity": severity, "details": details}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the spectral-skills Codex plugin image.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON envelope.")
    parser.parse_args(argv)
    response = check_codex_plugin()
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

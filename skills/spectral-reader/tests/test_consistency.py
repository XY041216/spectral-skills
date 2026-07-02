from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_core.reader.consistency import REQUIRED_SCHEMAS, REQUIRED_TOOLS, check_consistency


FALLBACK_TOOLS = [*REQUIRED_TOOLS, "check_consistency"]


def test_consistency_core_passes() -> None:
    response = check_consistency(repo_root=REPO_ROOT)
    assert response["ok"] is True
    assert response["result"]["consistency_status"] == "passed"


def test_manifest_yaml_loads_and_declares_required_tools() -> None:
    import yaml

    manifest = yaml.safe_load((REPO_ROOT / "skills" / "spectral-reader" / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["primary_tool"]["name"] == "read_spectral_dataset"
    assert sorted(manifest["mcp"]["tools"]) == sorted(f"reader.{tool}" for tool in REQUIRED_TOOLS)
    assert sorted(manifest["scripts_fallback"]["tools"].keys()) == sorted(FALLBACK_TOOLS)


def test_required_schemas_are_valid_json_and_synced() -> None:
    reader_dir = REPO_ROOT / "skills" / "spectral-reader" / "schemas"
    core_dir = REPO_ROOT / "spectral_core" / "schemas"
    for name in REQUIRED_SCHEMAS:
        reader_schema = json.loads((reader_dir / name).read_text(encoding="utf-8"))
        core_schema = json.loads((core_dir / name).read_text(encoding="utf-8"))
        assert reader_schema == core_schema


def test_scripts_and_fallbacks_exist() -> None:
    for tool in FALLBACK_TOOLS:
        assert (REPO_ROOT / "skills" / "spectral-reader" / "scripts" / f"{tool}.py").exists()
        assert (REPO_ROOT / "scripts" / "reader" / f"{tool}.py").exists()


def test_server_health_and_consistency_cli_output_json() -> None:
    for script in [
        REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "server_health.py",
        REPO_ROOT / "skills" / "spectral-reader" / "scripts" / "check_consistency.py",
        REPO_ROOT / "scripts" / "reader" / "check_consistency.py",
    ]:
        completed = subprocess.run(
            [sys.executable, str(script), "--json"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        assert {"ok", "tool", "backend", "schema_version", "result", "warnings", "errors"} <= set(payload)


def test_shared_skill_entry_absent_and_old_flow_terms_absent() -> None:
    assert not (REPO_ROOT / "skills" / "_shared" / "SKILL.md").exists()
    response = check_consistency(repo_root=REPO_ROOT)
    assert not response["result"]["mismatches"]


def test_new_runtime_knowledge_fragments_exist_without_development_flow() -> None:
    fragment_dir = REPO_ROOT / "skills" / "spectral-reader" / "static" / "fragments"
    forbidden = ["Step ", "Read Plan Mapping", "debug mode", "_internal", "package_manifest", "summary.json", "confidence_scores", "decision_trace", "validation_report", "preview_report", "skill development"]
    for name in ["numeric-band-columns.md", "folder-name-as-label.md", "excel-layout-cases.md"]:
        path = fragment_dir / name
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text


def test_skill_md_uses_standard_progressive_disclosure_sections() -> None:
    skill = (REPO_ROOT / "skills" / "spectral-reader" / "SKILL.md").read_text(encoding="utf-8")
    for heading in [
        "## Activation Boundary",
        "## Core Flow",
        "## Standard Handoff",
        "## Confirmation Rules",
        "## Execution Boundary",
        "## Output Boundary",
        "## Current Scope",
        "## Read As Needed",
    ]:
        assert heading in skill
    for path in [
        "static/core/workflow.md",
        "static/core/internal-read-settings.md",
        "static/core/confirmation-gates.md",
        "static/core/output-contract.md",
        "static/fragments/missing-values.md",
        "static/fragments/reader-qc-boundary.md",
    ]:
        assert path in skill

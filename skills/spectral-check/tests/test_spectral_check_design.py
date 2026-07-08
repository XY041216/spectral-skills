from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECK_DIR = REPO_ROOT / "skills" / "spectral-check"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_spectral_check_skill_files_exist() -> None:
    required = [
        "SKILL.md",
        "manifest.yaml",
        "static/core/check-boundary.md",
        "static/core/standard-package.md",
        "static/core/confirmation-gates.md",
        "static/fragments/method-pool.md",
        "static/fragments/missing-values.md",
        "static/fragments/outlier-candidates.md",
        "static/fragments/class-target-risks.md",
        "static/fragments/duplicate-spectra.md",
        "static/fragments/band-quality.md",
        "references/check-scenarios.md",
        "references/method-selection-cases.md",
        "references/handoff-examples.md",
        "schemas/qc_action_result.schema.json",
    ]
    for rel in required:
        assert (CHECK_DIR / rel).exists(), rel


def test_spectral_check_frontmatter_and_manifest_boundary() -> None:
    skill = _read(CHECK_DIR / "SKILL.md")
    assert "name: spectral-check" in skill
    assert "spectral-reader" in skill
    assert "data_contract.json" in skill
    assert "Do not use this skill to read raw source" in skill
    assert "Do not use this skill to read raw files" in skill
    assert "Do not use this skill to read raw source" in skill

    manifest = yaml.safe_load(_read(CHECK_DIR / "manifest.yaml"))
    assert manifest["name"] == "spectral-check"
    assert manifest["skill_type"] == "quality check"
    assert "reading raw spectral source files" in manifest["activation_boundary"]["not_for"]
    assert "spectral quality check" in manifest["activation_boundary"]["use_for"]


def test_standard_package_contract_is_same_handoff_shape() -> None:
    manifest = yaml.safe_load(_read(CHECK_DIR / "manifest.yaml"))
    package = manifest["standard_package"]
    assert package["required_files"] == [
        "data_contract.json",
        "X.csv",
        "sample_ids.csv",
        "band_axis.csv",
    ]
    assert package["optional_files"] == ["y.csv", "metadata.csv"]
    assert package["rewritten_output_files"] == [
        "data_contract.json",
        "X.csv",
        "sample_ids.csv",
        "band_axis.csv",
        "y.csv",
        "metadata.csv",
    ]
    assert "X_qc.csv" in package["forbidden_handoff_files"]
    assert "qc_contract.json" not in package["forbidden_handoff_files"]


def test_confirmation_gate_protects_destructive_actions() -> None:
    text = _read(CHECK_DIR / "static/core/confirmation-gates.md")
    for phrase in [
        "filling missing values",
        "deleting samples",
        "deleting bands",
        "deleting duplicate spectra candidates",
        "deleting target outlier candidates",
    ]:
        assert phrase in text
    assert "Candidate detection is not a destructive edit." in text


def test_method_pool_contains_required_families() -> None:
    text = _read(CHECK_DIR / "static/fragments/method-pool.md")
    for phrase in [
        "NOE",
        "MD",
        "PCA Hotelling T2",
        "Q residual",
        "Robust Z-score",
        "IQR",
        "MAD",
        "HR",
        "MCCV",
        "PLS residual",
        "Class-aware",
    ]:
        assert phrase.lower() in text.lower()


def test_outlier_check_mentions_standard_strategy_and_advanced_mark_options() -> None:
    skill = _read(CHECK_DIR / "SKILL.md")
    for phrase in [
        "These results come from the standard comprehensive",
        "HR/MCCV were not run by default",
        "rerun mark mode with MD",
        "If the user names MD",
        "run `mark` mode with the requested method",
    ]:
        assert phrase in skill


def test_qc_knowledge_does_not_create_report_or_debug_system() -> None:
    forbidden = [
        "write debug",
        "create logs",
        "write package_manifest.json",
        "write summary.json",
        "use X_qc.csv",
    ]
    for path in [*CHECK_DIR.glob("**/*.md"), *CHECK_DIR.glob("**/*.yaml")]:
        text = _read(path)
        for phrase in forbidden:
            assert phrase not in text


def test_qc_action_result_schema_is_valid_json() -> None:
    payload = json.loads(_read(CHECK_DIR / "schemas/qc_action_result.schema.json"))
    assert payload["title"] == "Spectral Check Action Result"
    assert payload["properties"]["status"]["enum"] == ["ready", "needs_confirmation", "blocked"]

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITTER_DIR = REPO_ROOT / "skills" / "spectral-splitter"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_spectral_splitter_skill_files_exist() -> None:
    required = [
        "SKILL.md",
        "manifest.yaml",
        "static/core/split-boundary.md",
        "static/core/standard-package.md",
        "static/core/output-contract.md",
        "static/fragments/method-selection.md",
        "static/fragments/safety-checks.md",
        "references/split-scenarios.md",
        "schemas/split_action_result.schema.json",
        "scripts/split_spectral_package.py",
    ]
    for rel in required:
        assert (SPLITTER_DIR / rel).exists(), rel


def test_frontmatter_and_manifest_boundary() -> None:
    skill = _read(SPLITTER_DIR / "SKILL.md")
    assert "name: spectral-splitter" in skill
    assert "spectral-reader" in skill
    assert "spectral-check" in skill
    assert "split_contract.json" in skill
    assert "Do not use this skill to read raw files" in skill
    assert "Do not rewrite `X.csv`" in skill

    manifest = yaml.safe_load(_read(SPLITTER_DIR / "manifest.yaml"))
    assert manifest["name"] == "spectral-splitter"
    assert manifest["skill_type"] == "data splitting"
    assert "quality control" in manifest["activation_boundary"]["not_for"]
    assert "classification stratified split" in manifest["activation_boundary"]["use_for"]
    assert "SPXY split" in manifest["activation_boundary"]["use_for"]
    assert "cross validation" not in manifest["activation_boundary"]["not_for"]


def test_output_boundary_forbids_copied_matrices() -> None:
    manifest = yaml.safe_load(_read(SPLITTER_DIR / "manifest.yaml"))
    package = manifest["standard_package"]
    assert package["required_files"] == ["data_contract.json", "X.csv", "sample_ids.csv", "band_axis.csv"]
    assert package["output_files"] == ["split_indices.csv", "split_contract.json", "split_summary.json"]
    assert "train_X.csv" in package["forbidden_outputs"]
    assert "test_X.csv" in package["forbidden_outputs"]
    assert "X.csv" in package["not_rewritten"]


def test_method_selection_documents_later_extensions() -> None:
    text = _read(SPLITTER_DIR / "static/fragments/method-selection.md")
    for phrase in [
        "random",
        "stratified",
        "predefined_split",
        "SPXY",
        "Kennard-Stone",
        "group-aware",
        "K-fold",
        "regression",
        "nested CV",
        "Recommended split",
        "Supported split methods",
        "When to choose another split",
        "You may choose",
        "full supported split-method menu",
        "method_code / English name",
        "English-only split menus are invalid",
        "stratified_monte_carlo_cv",
        "stratified_group",
    ]:
        assert phrase.lower() in text.lower()

    skill = _read(SPLITTER_DIR / "SKILL.md")
    for phrase in [
        "show the full supported split-method menu",
        "Supported split-method menu",
        "stratified",
        "stratified_group",
    ]:
        assert phrase.lower() in skill.lower()


def test_split_action_result_schema_is_valid_json() -> None:
    payload = json.loads(_read(SPLITTER_DIR / "schemas/split_action_result.schema.json"))
    assert payload["title"] == "Spectral Split Action Result"
    assert payload["properties"]["status"]["enum"] == ["ready", "needs_confirmation", "blocked"]

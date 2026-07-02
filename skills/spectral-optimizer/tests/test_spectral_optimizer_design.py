from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER_DIR = REPO_ROOT / "skills" / "spectral-optimizer"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_spectral_optimizer_skill_files_exist() -> None:
    required = [
        "SKILL.md",
        "manifest.yaml",
        "static/core/optimizer-boundary.md",
        "static/core/leakage-rules.md",
        "static/core/output-contract.md",
        "static/fragments/mode-selection.md",
        "static/fragments/search-spaces.md",
        "references/optimizer-scenarios.md",
        "schemas/optimizer_action_result.schema.json",
        "scripts/optimize_spectral_pipeline.py",
    ]
    for rel in required:
        assert (OPTIMIZER_DIR / rel).exists(), rel


def test_frontmatter_and_manifest_boundary() -> None:
    skill = _read(OPTIMIZER_DIR / "SKILL.md")
    assert "name: spectral-optimizer" in skill
    assert "recommend_from_profile" in skill
    assert "test_used_for_selection: false" in skill
    assert "When validation/CV metrics tie" in skill
    assert "Do not use this skill to read raw files" in skill

    manifest = yaml.safe_load(_read(OPTIMIZER_DIR / "manifest.yaml"))
    assert manifest["name"] == "spectral-optimizer"
    assert manifest["skill_type"] == "optimization"
    assert "model training execution" in manifest["activation_boundary"]["not_for"]
    assert manifest["safety"]["test_used_for_selection"] is False
    assert "fewer_output_features" in manifest["safety"]["tie_breaker_policy"]
    assert "optimize_pipeline" in manifest["modes"]


def test_best_pipeline_preserves_full_upstream_lineage() -> None:
    skill = _read(OPTIMIZER_DIR / "SKILL.md")
    for phrase in [
        "best_pipeline.json",
        "full trial lineage",
        "modeling_output",
        "preprocess_method",
        "feature_method",
        "feature_contract.json",
        "preprocess_contract.json",
        "silently drops the",
    ]:
        assert phrase in skill


def test_optimizer_action_result_schema_is_valid_json() -> None:
    payload = json.loads(_read(OPTIMIZER_DIR / "schemas/optimizer_action_result.schema.json"))
    assert payload["title"] == "Spectral Optimizer Action Result"
    assert payload["properties"]["status"]["enum"] == ["ready", "needs_confirmation", "blocked"]


def test_discovery_embeddings_are_excluded_from_default_search_spaces() -> None:
    text = _read(OPTIMIZER_DIR / "static/fragments/search-spaces.md")
    for phrase in [
        "Tuning Levels",
        "Level 1",
        "classifier parameter tuning only",
        "Level 2",
        "traditional feature plus classifier tuning",
        "PCA(10)",
        "PLS-LV(3)",
        "VIP(100)",
        "KBest(80)",
        "SPA(80)",
        "Level 3",
        "deep embedding plus classifier tuning",
        "extra budget confirmation",
        "Do not include discovery-only feature embeddings",
        "tsne_embedding",
        "umap_embedding",
        "isomap_embedding",
        "lle_embedding",
        "primarily for `spectral-report` visualization",
        "contrastive_spectral_embedding",
        "masked_spectral_autoencoder_embedding",
        "self_supervised_spectral_embedding",
        "excluded from compact/recommended optimizer spaces",
        "n_components=2",
        "visualization embeddings",
        "n_components=[8,16,32]",
        "Macro-F1",
        "classifier-only",
        "previously inspected test metrics",
    ]:
        assert phrase in text


def test_optimizer_documents_four_budget_profiles_and_deep_opt_in() -> None:
    mode = _read(OPTIMIZER_DIR / "static/fragments/mode-selection.md")
    spaces = _read(OPTIMIZER_DIR / "static/fragments/search-spaces.md")
    skill = _read(OPTIMIZER_DIR / "SKILL.md")
    combined = mode + spaces + skill
    for phrase in [
        "`quick`",
        "`regular`",
        "`extended`",
        "`deep`",
        "recommended` is accepted",
        "[8,16,32]",
        "[linear_svm, svm, lda]",
        "exceed 300",
        "never selected automatically",
    ]:
        assert phrase in combined

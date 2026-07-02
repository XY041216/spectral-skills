from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESS_DIR = REPO_ROOT / "skills" / "spectral-preprocess"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_spectral_preprocess_skill_files_exist() -> None:
    required = [
        "SKILL.md",
        "manifest.yaml",
        "static/core/preprocess-boundary.md",
        "static/core/leakage-rules.md",
        "static/core/output-contract.md",
        "static/fragments/method-selection.md",
        "static/fragments/safety-checks.md",
        "references/preprocess-scenarios.md",
        "schemas/preprocess_action_result.schema.json",
        "scripts/preprocess_spectral_package.py",
    ]
    for rel in required:
        assert (PREPROCESS_DIR / rel).exists(), rel


def test_frontmatter_and_manifest_boundary() -> None:
    skill = _read(PREPROCESS_DIR / "SKILL.md")
    assert "name: spectral-preprocess" in skill
    assert "spectral-splitter" in skill
    assert "split_contract.json" in skill
    assert "train samples only" in skill
    assert "Do not read raw" in skill
    assert "Do not write `X_preprocessed.csv`" in skill

    manifest = yaml.safe_load(_read(PREPROCESS_DIR / "manifest.yaml"))
    assert manifest["name"] == "spectral-preprocess"
    assert manifest["skill_type"] == "preprocessing"
    assert "dataset splitting" in manifest["activation_boundary"]["not_for"]
    assert "standardization" in manifest["preprocess_scope"]["methods"]
    assert "msc" in manifest["preprocess_scope"]["train_fit_methods"]


def test_leakage_rules_are_explicit() -> None:
    text = _read(PREPROCESS_DIR / "static/core/leakage-rules.md")
    for phrase in ["train samples", "validation or test", "mean", "standard deviations", "MSC references"]:
        assert phrase in text


def test_preprocess_action_result_schema_is_valid_json() -> None:
    payload = json.loads(_read(PREPROCESS_DIR / "schemas/preprocess_action_result.schema.json"))
    assert payload["title"] == "Spectral Preprocess Action Result"
    assert payload["properties"]["status"]["enum"] == ["ready", "needs_confirmation", "blocked"]


def test_preprocess_confirmation_cards_show_full_method_menu() -> None:
    text = _read(PREPROCESS_DIR / "static/fragments/method-selection.md")
    for phrase in [
        "Recommended preprocessing",
        "Supported preprocessing methods",
        "Methods requiring extra parameters or caution",
        "You may choose",
        "full supported preprocessing-method menu",
        "中文名称（method_code / English name）",
        "English-only preprocessing",
        "无预处理",
        "标准正态变量校正",
        "多元散射校正",
        "SNV + 去趋势",
        "moving_average",
        "gaussian_smoothing",
        "median_filter",
        "polynomial_baseline",
        "rubberband_baseline",
        "als_baseline",
        "minmax_scaling",
        "robust_scaling",
        "pareto_scaling",
        "l2_normalization",
        "area_normalization",
        "reflectance_to_absorbance",
        "remove_band_ranges",
    ]:
        assert phrase in text
    skill = _read(PREPROCESS_DIR / "SKILL.md")
    for phrase in [
        "show the full supported preprocessing menu in bilingual form",
        "中文名称（method_code / English name）",
        "保留物理波段范围（band_range_select / band range selection）",
        "移除物理波段范围（remove_band_ranges / remove band ranges）",
    ]:
        assert phrase in skill

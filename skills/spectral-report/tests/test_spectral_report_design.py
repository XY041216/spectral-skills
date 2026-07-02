from __future__ import annotations

from pathlib import Path
import csv
import json
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "skills" / "spectral-report"


def _read(relative: str) -> str:
    return (REPORT_DIR / relative).read_text(encoding="utf-8")


def test_required_knowledge_flow_files_exist() -> None:
    required = [
        "SKILL.md",
        "manifest.yaml",
        "static/core/boundary.md",
        "static/core/figure-contract.md",
        "static/core/integrity-rules.md",
        "static/core/typography-export.md",
        "static/core/qa-loop.md",
        "static/fragments/spectra-preprocess.md",
        "static/fragments/feature-interpretation.md",
        "static/fragments/classification.md",
        "static/fragments/regression.md",
        "static/fragments/optimizer.md",
        "static/fragments/revise-existing-figure.md",
        "references/input-artifact-map.md",
        "references/style-reference-index.md",
        "references/chart-decision-matrix.md",
        "references/axis-and-unit-rules.md",
        "references/visual-system.md",
        "references/statistical-reporting.md",
        "references/layout-recipes.md",
        "references/caption-rules.md",
        "references/qa-checklist.md",
        "references/scenario-tests.md",
    ]
    for relative in required:
        assert (REPORT_DIR / relative).is_file(), relative
    assert not (REPORT_DIR / "scripts").exists()
    assert not (REPORT_DIR / "agents" / "openai.yaml").exists()


def test_frontmatter_manifest_and_plugin_default_ui_match_boundary() -> None:
    skill = _read("SKILL.md")
    frontmatter = skill.split("---", 2)[1]
    payload = yaml.safe_load(frontmatter)
    assert set(payload) == {"name", "description"}
    assert payload["name"] == "spectral-report"
    assert "publication-grade" in payload["description"]
    assert "Do not use it to read raw vendor files" in payload["description"]

    manifest = yaml.safe_load(_read("manifest.yaml"))
    assert manifest["name"] == "spectral-report"
    assert manifest["skill_type"] == "reporting"
    assert manifest["safety"]["test_used_for_selection"] is False
    assert not (REPORT_DIR / "agents" / "openai.yaml").exists()


def test_reference_assets_are_complete_and_indexed() -> None:
    expected = {
        "spectra-multipanel.png",
        "model-boxplot-grid.png",
        "hyperparameter-line-grid.png",
        "training-ratio-line-comparison.png",
        "embedding-scatter-dense.png",
        "embedding-scatter-separated.png",
        "nature-control-treatment-multipanel.jpg",
        "nature-pastel-bar-scatter.jpg",
        "nature-pastel-trend-grid.jpg",
        "palette-reference.svg",
    }
    assets = {path.name for path in (REPORT_DIR / "assets" / "style-references").iterdir() if path.is_file()}
    assert assets == expected
    index = _read("references/style-reference-index.md")
    for name in expected:
        assert name in index
    assert "Do not paste" in index or "Never paste" in index
    assert "redistribution rights" in index


def test_data_integrity_and_chart_rules_cover_acceptance_failures() -> None:
    integrity = _read("static/core/integrity-rules.md")
    charts = _read("references/chart-decision-matrix.md")
    optimizer = _read("static/fragments/optimizer.md")
    classification = _read("static/fragments/classification.md")
    boundary = _read("static/core/boundary.md")
    contract = _read("static/core/figure-contract.md")
    visual = _read("references/visual-system.md")
    captions = _read("references/caption-rules.md")
    for phrase in ["single holdout", "statistical unit", "paired", "final test", "traceable"]:
        assert phrase.lower() in integrity.lower()
    for phrase in ["Cleveland", "start at zero", "ordered", "box/violin", "predicted vs measured", "mean-only heatmap", "rounded-margin y-axis"]:
        assert phrase.lower() in charts.lower()
    assert "validation/CV" in optimizer
    assert "best_pipeline.json" in optimizer
    for phrase in ["Classifier Set Gate", "compact", "regular", "spectral modeling", "boxplot + raw repeat points", "classifier_set_source"]:
        assert phrase in classification
    for phrase in [
        "Existing local scripts",
        "Pipeline Semantics Gate",
        "feature_pipeline_statement",
        "PCA(explained_variance=0.95",
        "PLS-LV(n_components=10",
        "three-panel boxplot",
        "0.55-0.70",
        "auto_y_limit=data_range_with_rounded_margin",
        "avoid_overzoom=true",
        "regular-full",
        "Gradient Boosting",
        "same sample may appear",
        "bar_label_rotation=0",
        "font size at least 90%",
        "label offset 2-3 pt",
        "do not solve it by rotating value labels",
        "direct metric values",
        "no repeated statistical units",
        "all_spines_visible=true",
        "gridlines_present=false",
        "panel_label_position=outside_upper_left",
        "display_name_map",
        "CLS-F",
        "Masked AE",
        "30 degrees",
        "tick labels at 10-11 pt",
        "caption must explicitly",
    ]:
        assert phrase in classification
    for phrase in ["report_style.grid=false", "ax.grid(False)", "outside the data", "Multi-method embedding scatter plots must"]:
        assert phrase in charts
    for phrase in ["three-line", "Experiment-setting", "Pipeline", "mean ± SD"]:
        assert phrase.lower() in classification.lower()
    for phrase in ["model-config.json", "candidate_space.json", "Call spectral-modeling"]:
        assert phrase in boundary
    for phrase in ["Confirmation Card Before Plotting", "chart grammar", "candidate_classifiers", "language", "palette", "feature_pipeline_statement", "summary_table_plan", "display_name_map"]:
        assert phrase in contract
    for phrase in ["report_style", "training_audit_summary", "3-column", "2 x 3 layout for six methods", "panel-label position", "legend placement"]:
        assert phrase in contract
    for phrase in [
        "Palette Confirmation",
        "ordinary paper high-distinction",
        "soft paper",
        "colorblind-friendly",
        "not as the default",
        "low-saturation high-distinction fills",
        "Do not use saturated pure blue/orange/purple",
        "Do not use hatch patterns by default",
        "reference-style",
        "grayscale",
        "raw repeat points",
        "default Matplotlib",
        "green triangles",
        "dark red diamond",
    ]:
        assert phrase in visual
    for phrase in ["report_style.grid=false", "full black axis frames", "full black axis frames", "all_spines_visible=true", "gridlines_present=false", "ax.grid(False)"]:
        assert phrase in visual
    for phrase in ["Final Chat Summary", "paper-ready Markdown results table", "experiment-setting table", "QC warning samples"]:
        assert phrase in captions
    for phrase in [
        "single validation split result",
        "final locked test result",
        "10 repeated held-out result",
        "statistical unit",
        "metric unit",
        "test accuracy",
        "held-out accuracy",
    ]:
        assert phrase in captions or phrase in contract
    for phrase in ["training audit table", "method-specific", "not directly comparable"]:
        assert phrase in captions
    for phrase in ["all compared classifiers", "Accuracy (%)", "Balanced accuracy (%)", "Macro-F1 (%)"]:
        assert phrase in captions


def test_output_contract_typography_and_qa_are_explicit() -> None:
    skill = _read("SKILL.md")
    typography = _read("static/core/typography-export.md")
    qa = _read("static/core/qa-loop.md")
    for token in ["report_contract.json", "figures/", "source_data/", "code/", "captions/", "qa/"]:
        assert token in skill
    for phrase in ["A user confirmation for modeling", "panel-letter style", "y-axis strategy", "publication figures", "+/-"]:
        assert phrase in skill
    for phrase in [
        "Times New Roman",
        "SimSun",
        "89 mm",
        "183 mm",
        "editable",
        "final physical size",
        "Times News Roman",
        "font language",
        "No DejaVu fallback",
        "tick labels 10-11 pt",
        "rotation=0",
        "90% of tick-label size",
    ]:
        assert phrase in typography
    for phrase in ["Data and Statistical QA", "Visual and Export QA", "grayscale", "SVG text"]:
        assert phrase in qa
    for phrase in [
        "visible_gridlines=false",
        "grid exception",
        "full black frames",
        "all_spines_visible=true",
        "spine_color=black",
        "boxplots",
        "only left/bottom",
        "panel-label style is consistent",
        "3-column compact layout",
        "2 x 3 layout for six methods",
        "outside panel labels",
        "low-saturation default fills without hatches",
        "readable abbreviated x labels",
        "white background",
        "low-saturation colors",
        "Times New Roman",
        "statistical unit",
        "metric unit",
    ]:
        assert phrase in qa
    for phrase in ["ax.grid(False)", "report_style.grid=false"]:
        assert phrase in typography


def test_backend_confirmation_and_bar_labels_are_required() -> None:
    skill = _read("SKILL.md")
    classification = _read("static/fragments/classification.md")
    contract = _read("static/core/figure-contract.md")
    charts = _read("references/chart-decision-matrix.md")
    captions = _read("references/caption-rules.md")
    qa = _read("static/core/qa-loop.md")
    scenarios = _read("references/scenario-tests.md")

    for phrase in [
        "Python/Matplotlib-Seaborn",
        "R/ggplot2",
        "plotting backend",
        "plot_backend",
        "backend confirmation",
    ]:
        assert phrase in skill or phrase in contract or phrase in scenarios

    for phrase in [
        "Bar Chart Rules",
        "mean ± SD",
        "label the mean above each bar",
        "bar_label_rotation=0",
        "font size at least 90%",
        "label offset 2-3 pt",
        "do not solve it by rotating value labels",
        "saturated pure blue/orange/purple",
        "hatch patterns",
        "low-saturation",
        "colorblind-friendly as the default",
        "error-bar cap",
        "bar_value_labels",
    ]:
        assert phrase in classification or phrase in charts or phrase in contract

    for phrase in [
        "bar chart captions",
        "mean values",
        "horizontal",
        "short numeric labels",
        "value-label format",
        "label collisions",
        "direct plotted metric values",
        "no repeated statistical units",
        "DAE = Denoising",
        "CLS-F = CLS-former",
    ]:
        assert phrase in captions or phrase in qa


def test_feature_embedding_interpretation_is_discovery_only() -> None:
    feature = _read("static/fragments/feature-interpretation.md")
    for phrase in [
        "Kernel PCA",
        "Sparse PCA",
        "NMF",
        "ICA",
        "LDA projection",
        "DCT",
        "FFT",
        "t-SNE/UMAP/Isomap/LLE",
        "visual separation is not performance evidence",
        "feature_mode=visualization_embedding",
        "do not train a new embedding inside spectral-report",
        "transform_available_for_new_samples=false",
        "3 x 3 layout",
        "2 x 3 layout",
        "full black axis frames",
        "shared figure-level axis labels",
        "outside_upper_left",
        "display_name_map",
        "Denoising AE",
        "Masked spectral AE",
        "color-only",
        "not directly comparable across methods",
        "training audit table",
        "no convergence claim",
    ]:
        assert phrase in feature


def test_categorical_method_comparisons_use_distinct_low_saturation_colors_and_record_sorting() -> None:
    classification = _read("static/fragments/classification.md")
    contract = _read("static/core/figure-contract.md")
    qa = _read("static/core/qa-loop.md")
    combined = classification + contract + qa
    for phrase in [
        "distinct low-saturation color",
        "same color across metric panels",
        "Do not default to a colorblind palette",
        "method_palette",
        "method_order",
        "sort_metric",
        "sort_statistic",
        "sort_direction",
        "Same-fill multi-method plots",
        "LR",
        "RBF-SVM",
        "ET",
        "RF",
    ]:
        assert phrase in combined


def test_minimum_scenarios_and_input_layers_are_documented() -> None:
    scenarios = _read("references/scenario-tests.md")
    artifact_map = _read("references/input-artifact-map.md")
    assert scenarios.count("\n") >= 12
    for phrase in ["Single holdout", "Ten-fold CV", "Repeated holdout", "Optimizer trials", "Regression", "Chinese panels", "89 mm", "CVD/grayscale"]:
        assert phrase in scenarios
    for phrase in ["compare classifiers", "classifier-set confirmation", "model-config JSON"]:
        assert phrase in scenarios
    for phrase in ["Seven deep embedding", "3 x 3 layout", "Six deep embedding", "2 x 3 layout", "outside the data region", "no background gridlines", "training audit table", "direct test-set metric values", "no error bars because there are no repeated units"]:
        assert phrase in scenarios
    for layer in ["reader/QC", "split", "preprocess", "feature", "modeling", "optimizer"]:
        assert layer in artifact_map


def test_single_holdout_forward_smoke_writes_complete_dot_plot_package(tmp_path: Path) -> None:
    output = tmp_path / "spectral-report-output"
    for name in ["figures", "source_data", "code", "captions", "qa"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    source_path = output / "source_data" / "fig_01_model_comparison.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "split", "metric", "value", "statistical_unit"])
        for model, value in [("SVM", 0.83), ("PLS-DA", 0.79), ("LDA", 0.74), ("KNN", 0.70), ("RF", 0.76)]:
            writer.writerow([model, "validation", "macro_f1", value, "single_holdout"])

    contract = {
        "core_claim": "Validation Macro-F1 differs across five locked classifiers.",
        "figure_role": "comparison",
        "input_lineage": ["fixture/modeling_contract.json", "fixture/metrics.json"],
        "task_type": "classification",
        "evaluation_scope": "validation",
        "test_isolation_status": "untouched",
        "figure_archetype": "single_panel",
        "statistical_unit": "single_holdout",
        "uncertainty_definition": "none",
        "axis_semantics": {"x": "macro_f1", "scale": "0-1"},
        "final_size_mm": {"width": 89, "height": 65},
        "exports": ["svg", "pdf", "png"],
    }
    (output / "report_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    code = '''from pathlib import Path
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root / "source_data" / "fig_01_model_comparison.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
rows.sort(key=lambda row: float(row["value"]))
models = [row["model"] for row in rows]
values = [float(row["value"]) for row in rows]
mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(89 / 25.4, 65 / 25.4))
ax.scatter(values, models, s=18, color="#2454E6", marker="o", zorder=3)
ax.set_xlabel("Validation Macro-F1")
ax.set_ylabel("Model")
ax.set_xlim(0.65, 0.86)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(False)
fig.tight_layout()
for suffix, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 300})]:
    fig.savefig(root / "figures" / f"fig_01_model_comparison.{suffix}", **kwargs)
plt.close(fig)
'''
    code_path = output / "code" / "fig_01_model_comparison.py"
    code_path.write_text(code, encoding="utf-8")
    assert "scatter" in code and "boxplot" not in code and "errorbar" not in code and ".bar(" not in code
    completed = subprocess.run([sys.executable, str(code_path)], cwd=output, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr

    (output / "captions" / "fig_01_model_comparison.md").write_text(
        "Five locked classifiers compared on one validation holdout using Macro-F1. Each point is one model result; no uncertainty is shown because no repeated units are available. Test metrics were not used.",
        encoding="utf-8",
    )
    (output / "qa" / "figure_qa.md").write_text(
        "# Figure QA\n\n- PASS: source values match plotted points.\n- PASS: single holdout uses dots without boxes or error bars.\n- PASS: SVG/PDF/PNG rendered at 89 mm.\n",
        encoding="utf-8",
    )

    expected = [
        "report_contract.json",
        "figures/fig_01_model_comparison.svg",
        "figures/fig_01_model_comparison.pdf",
        "figures/fig_01_model_comparison.png",
        "source_data/fig_01_model_comparison.csv",
        "code/fig_01_model_comparison.py",
        "captions/fig_01_model_comparison.md",
        "qa/figure_qa.md",
    ]
    for relative in expected:
        path = output / relative
        assert path.is_file() and path.stat().st_size > 0, relative
    svg = (output / "figures" / "fig_01_model_comparison.svg").read_text(encoding="utf-8")
    assert "Validation Macro-F1" in svg
    assert "<text" in svg

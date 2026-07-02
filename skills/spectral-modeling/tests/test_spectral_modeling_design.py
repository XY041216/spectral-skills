from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODELING = ROOT / "skills" / "spectral-modeling"


def read(relative: str) -> str:
    return (MODELING / relative).read_text(encoding="utf-8")


def test_modeling_boundary_and_outputs() -> None:
    skill = read("SKILL.md")
    assert "name: spectral-modeling" in skill
    assert "validation or train-only CV" in skill
    assert "pipeline_bundle/" in skill
    assert "test_access_log.json" in skill
    assert "Do not use it to read raw files" in skill


def test_complete_grouped_bilingual_classifier_menu() -> None:
    menu = read("static/fragments/model-selection.md")
    for phrase in [
        "传统机器学习与化学计量分类器",
        "可选 boosting 分类器",
        "小样本深度学习与实验光谱模型",
        "中文名称（method_code / English name）：",
        "逻辑回归（`logistic_regression` / Logistic Regression）：",
        "RBF 支持向量机（`svm` / RBF-SVM）：",
        "XGBoost 分类器（`xgboost_classifier` / XGBoost classifier）：",
        "skill 还支持但本轮默认不纳入",
        "regular-fast",
    ]:
        assert phrase in menu


def test_deep_models_have_chinese_names_and_small_sample_notes() -> None:
    menu = read("static/fragments/model-selection.md")
    for phrase in [
        "光谱 DKL-GP 分类器（`spectral_dkl_gp_classifier` / Spectral DKL-GP classifier）：",
        "原型光谱分类器（`proto_spectral_classifier` / Prototype spectral classifier）：",
        "CLS-former 分类器（`cls_former_classifier` / CLS-former classifier）：",
        "CLS-former 嵌入 + SVM（`cls_former_embedding_svm` / CLS-former embedding plus SVM）：",
        "embedding_dim=8/16",
        "早停",
        "not proof that tuning is broken",
    ]:
        assert phrase in menu


def test_tuning_is_layered_bounded_and_macro_f1_selected() -> None:
    skill = read("SKILL.md")
    menu = read("static/fragments/model-selection.md")
    combined = skill + menu
    for phrase in [
        "Level 1",
        "Level 2",
        "Level 3",
        "Macro-F1",
        "final-test metrics",
        "class_weight",
        "min_samples_leaf",
        "shrinkage",
        "subsample",
        "reg_lambda",
        "overfitting signal",
    ]:
        assert phrase in combined


def test_best_pipeline_replay_requires_upstream_contracts() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "Best-pipeline replay",
        "--best-pipeline --lock-best-pipeline-params",
        "preprocess contract",
        "feature contract",
        "trial_dir/feature_output/feature_contract.json",
        "block with an explicit error",
        "best_pipeline_reproduction",
    ]:
        assert phrase in skill


def test_holdout_validation_comparison_contract_is_required() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "classifier_validation_summary.csv",
        "one row per classifier family",
        "train/validation accuracy",
        "balanced accuracy",
        "macro-F1",
        "test_accessed=false",
        "authoritative comparison table",
    ]:
        assert phrase in skill


def test_repeated_classifier_comparison_contract_is_required() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "Repeated classifier comparison",
        "classifier_repeat_metrics.csv",
        "classifier_metric_summary.csv",
        "one row per `(repeat_id, model_method)`",
        "Accuracy, Balanced accuracy, and Macro-F1 mean ± SD",
    ]:
        assert phrase in skill


def test_multi_model_completion_requires_full_comparison_table() -> None:
    skill = read("SKILL.md")
    menu = read("static/fragments/model-selection.md")
    combined = skill + menu
    for phrase in [
        "Final response contract",
        "one row per evaluated model",
        "Do not report only the selected model",
        "Logistic Regression",
        "Linear SVM",
        "RBF-SVM",
        "LDA",
        "KNN",
        "Random Forest",
        "Extra Trees",
        "Test accessed",
        "modeling_summary.json",
        "classifier_validation_summary.csv",
    ]:
        assert phrase in combined

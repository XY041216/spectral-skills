from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "skills" / "spectral-workflow"


def read(relative: str) -> str:
    return (WORKFLOW / relative).read_text(encoding="utf-8")


def test_workflow_boundary_and_contract_handoffs() -> None:
    skill = read("SKILL.md")
    assert "name: spectral-workflow" in skill
    for phrase in [
        "spectral-reader",
        "spectral-qc",
        "spectral-splitter",
        "spectral-preprocess",
        "spectral-feature",
        "spectral-modeling",
        "spectral-optimizer",
        "spectral-report",
        "workflow_plan.json",
        "workflow_result.json",
        "locked atomic read-modify-write",
    ]:
        assert phrase in skill


def test_generic_route_card_has_seven_separate_routes() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "确认推荐基线",
        "只读取和 QC",
        "手动选择常规方法",
        "常规优化比较",
        "深度学习模型实验",
        "深度嵌入 + 传统分类器比较",
        "可视化探索",
        "Do not collapse routes 5-7",
        "Do not show full method menus at this point",
    ]:
        assert phrase in skill


def test_stage_cards_are_transparent_bilingual_and_grouped() -> None:
    skill = read("SKILL.md")
    gates = read("static/fragments/confirmation-gates.md")
    combined = skill + gates
    for phrase in [
        "推荐方案",
        "为什么推荐",
        "本轮默认纳入",
        "skill 还支持但本轮默认不纳入",
        "需要额外确认前才能执行",
        "自动调参能力",
        "中文名称（method_code / English name）：",
        "traditional/chemometric",
        "optional boosting",
        "small-sample deep/experimental",
    ]:
        assert phrase in combined


def test_deep_protocol_is_data_aware_not_shared_defaults() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "n_samples",
        "n_train",
        "n_features",
        "embedding/feature dimension",
        "early-stopping status/patience",
        "weight decay",
        "method-specific",
        "n<=120",
        "16 dimensions",
        "2D",
        "high-overfit risk warning",
        "Do not apply one shared default bundle",
    ]:
        assert phrase in skill


def test_optimizer_has_three_levels_four_budgets_and_full_replay() -> None:
    skill = read("SKILL.md")
    gates = read("static/fragments/confirmation-gates.md")
    combined = skill + gates
    for phrase in [
        "Level 1",
        "Level 2",
        "Level 3",
        "`quick`",
        "`regular`",
        "`extended`",
        "`deep`",
        "Macro-F1",
        "Never use final-test metrics",
        "--preview-only",
        "--lock-best-pipeline-params",
        "locked replay must include preprocess, feature, model",
    ]:
        assert phrase in combined


def test_modeling_completion_requires_full_table_not_best_only() -> None:
    skill = read("SKILL.md")
    gates = read("static/fragments/confirmation-gates.md")
    combined = skill + gates
    for phrase in [
        "Modeling completion handoff",
        "metrics.json",
        "modeling_summary.json",
        "modeling_contract.json",
        "one row per model",
        "Do not summarize only the selected model",
        "Train Macro-F1",
        "Validation Macro-F1",
        "Validation Accuracy",
        "Test accessed",
        "Logistic Regression",
        "Linear SVM",
        "RBF-SVM",
        "LDA",
        "KNN",
        "Random Forest",
        "Extra Trees",
    ]:
        assert phrase in combined


def test_report_route_requires_distinct_method_colors_and_sort_rule() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "white background",
        "no grid",
        "full black frames",
        "Times New Roman",
        "outside lowercase panel labels",
        "distinct low-saturation colors by method",
        "sorting rule",
        "Do not default to a colorblind palette",
    ]:
        assert phrase in skill


def test_leakage_and_test_policy_is_explicit() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "Never fit preprocessing/features on full data before split",
        "Never use test metrics",
        "confirmatory, not blind",
        "Visual embedding separation is not predictive-performance evidence",
    ]:
        assert phrase in skill

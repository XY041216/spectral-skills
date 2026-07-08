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
        "spectral-check",
        "spectral-splitter",
        "spectral-preprocess",
        "spectral-feature",
        "spectral-modeling",
        "spectral-optimizer",
        "spectral-report",
        "workflow_plan.json",
        "workflow_result.json",
        "All files created for one user-facing spectral analysis must converge under one",
        "reader_package",
        "optimizer_output",
        "locked atomic read-modify-write",
    ]:
        assert phrase in skill


def test_generic_route_card_is_profile_aware_and_uses_check_name() -> None:
    skill = read("SKILL.md")
    gates = read("static/fragments/confirmation-gates.md")
    combined = skill + gates
    for phrase in [
        "spectral-check",
        "optimization",
        "comparison route",
        "deep-learning model experiment",
        "deep embedding plus traditional",
        "visualization-only exploration",
        "Do not collapse routes 5-7",
        "Do not show full method menus at this point",
        "mandatory reader-completion route card",
        "下一步可以选一个路线继续",
        "do not run spectral-check",
        "User-facing terminology rule",
        "never write `QC`",
        "Canonical route card",
        "all-supported candidate universe",
        "small_sample",
        "high_dimensional",
        "very_high_dimensional",
        "class_imbalance",
        "cls_former_embedding_svm",
        "contrastive_spectral_embedding + linear_svm/svm",
        "Generic route output layout is also mandatory",
        "<stem>_standard_package",
        "<stem>_optimizer_regular72",
    ]:
        assert phrase in combined


def test_stage_cards_are_transparent_bilingual_and_grouped() -> None:
    skill = read("SKILL.md")
    gates = read("static/fragments/confirmation-gates.md")
    combined = skill + gates
    for phrase in [
        "complete stage menu",
        "Long menus may be grouped",
        "Stage confirmation cards",
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


def test_optimizer_route_lists_regular_and_deep_budget_choices() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "Level 1",
        "Level 2",
        "Level 3",
        "`quick`",
        "`regular`",
        "`extended`",
        "`deep`",
        "validation design",
        "Never use final-test metrics",
        "self-developed",
        "small-sample/deep",
        "Do not ask\nonly for `确认 regular 72`",
        "内置自创/深度候选是否加入选优组合",
        "Do not present these candidates as a bare code list",
        "CLS-former embedding + SVM",
        "n=120, p=3401",
    ]:
        assert phrase in skill


def test_workflow_route_index_uses_spectral_check() -> None:
    route_index = read("static/core/route-index.md")
    assert "`spectral-check`" in route_index
    assert "qc_result.json" in route_index

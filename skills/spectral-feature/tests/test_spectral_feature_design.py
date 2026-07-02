from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FEATURE = ROOT / "skills" / "spectral-feature"


def read(relative: str) -> str:
    return (FEATURE / relative).read_text(encoding="utf-8")


def test_feature_skill_boundary_and_outputs() -> None:
    skill = read("SKILL.md")
    assert "name: spectral-feature" in skill
    assert "split_contract.json" in skill
    assert "train only" in skill
    assert "feature_contract.json" in skill
    assert "Do not use it to read raw files" in skill
    assert "spectral-optimizer" in skill


def test_complete_bilingual_grouped_menu() -> None:
    menu = read("static/fragments/method-selection.md")
    for phrase in [
        "传统特征与化学计量方法",
        "投影、信号变换与流形方法",
        "深度嵌入方法",
        "中文名称（method_code / English name）：说明",
        "not the full capability surface",
        "主成分分析（`pca` / Principal Component Analysis, PCA）：",
        "CARS 变量筛选（`cars` / Competitive Adaptive Reweighted Sampling）：",
        "t-SNE 可视化嵌入（`tsne_embedding` / t-SNE embedding）：",
    ]:
        assert phrase in menu


def test_deep_embeddings_have_chinese_names_notes_and_data_aware_card() -> None:
    menu = read("static/fragments/method-selection.md")
    for phrase in [
        "自编码器嵌入（`autoencoder_embedding` / Autoencoder embedding）：",
        "去噪自编码器嵌入（`denoising_autoencoder_embedding` / Denoising autoencoder embedding）：",
        "一维 CNN 光谱嵌入（`cnn_1d_embedding` / 1D CNN spectral embedding）：",
        "ResNet1D 光谱嵌入（`resnet1d_embedding` / ResNet1D spectral embedding）：",
        "CLS-former 光谱嵌入（`cls_former_embedding` / CLS-former spectral embedding）：",
        "掩码光谱自编码器嵌入（`masked_spectral_autoencoder_embedding` / Masked spectral autoencoder embedding）：",
        "对比光谱嵌入（`contrastive_spectral_embedding` / Contrastive spectral embedding）：",
        "Data-aware deep-training card",
        "n=120",
        "n_train<=72",
        "n_components=16",
        "n_components=2",
        "weight_decay=1e-4",
        "fixed epoch count does not prove convergence",
    ]:
        assert phrase in menu


def test_feature_paths_are_resolved_and_recorded() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "Path resolution and contracts",
        "current working directory",
        "resolved relative to that contract file",
        "resolved_paths",
        "split_contract",
        "preprocess directory",
    ]:
        assert phrase in skill


def test_skill_requires_real_deep_confirmation_and_visualization_boundary() -> None:
    skill = read("SKILL.md")
    for phrase in [
        "--confirm-deep-embedding-training",
        "n_components",
        "early-stopping status",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "noise_std",
        "mask_ratio",
        "temperature",
        "patch_size",
        "visualization-only",
        "Visual separation is not",
    ]:
        assert phrase in skill

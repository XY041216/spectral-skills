from __future__ import annotations

from pathlib import Path


READER_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = READER_ROOT.parent


def _shared_root() -> Path:
    source_shared = SKILLS_ROOT / "_shared"
    if source_shared.exists():
        return source_shared
    return SKILLS_ROOT.parent / "shared"


def test_reader_static_fragments_do_not_contain_mojibake() -> None:
    garble_tokens = [
        "\ufffd",
        "\u9239",
        "\u6434\u5fd3",
        "\u7f02\u680f",
        "\u6d63\u7280",
        "\u7eeb\u8bf2",
        "\u95c7\u20ac",
        "\u9477",
        "\u6d93\u5a6d",
        "\u5a09\u682a",
        "\u9365",
        "\u951b",
        "\u9436",
        "\u754d",
        "\u00c3",
        "\u00c2",
        "\u00e2\u20ac",
    ]
    checked = list((READER_ROOT / "static").rglob("*.md")) + list((READER_ROOT / "references").rglob("*.md"))
    checked.append(_shared_root() / "README.md")
    offenders: list[str] = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in garble_tokens) or any("\ue000" <= ch <= "\uf8ff" for ch in text):
            offenders.append(str(path))
    assert offenders == []


def test_chinese_column_name_examples_are_readable() -> None:
    column_names = (READER_ROOT / "static" / "fragments" / "chinese-column-names.md").read_text(encoding="utf-8")
    metadata = (READER_ROOT / "static" / "fragments" / "metadata-before-spectra.md").read_text(encoding="utf-8")
    for term in ["样本编号", "编号", "样品编号", "序号", "备注", "批次", "类别", "等级", "品种", "产地", "含量", "浓度", "波长", "波数", "波段"]:
        assert term in column_names
    for term in ["编号", "序号", "备注"]:
        assert term in metadata
    assert "structured placeholders" not in (_shared_root() / "README.md").read_text(encoding="utf-8")

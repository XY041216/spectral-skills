from __future__ import annotations

from pathlib import Path


READER_ROOT = Path(__file__).resolve().parents[1]


def test_reader_completion_includes_next_step_route_card() -> None:
    skill = (READER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for phrase in [
        "Reader Completion Route Card",
        "incomplete if it omits any of\nthese nine route labels",
        "下一步可以选一个路线继续",
        "自动选优组合",
        "小型自创/深度候选加入",
        "compare preprocessing + feature + model combinations",
        "全量支持方法选优预览",
        "小样本/深度模型实验",
        "深度嵌入 + 传统模型比较",
        "Use `check` or `质量检查`",
        "do not write `QC`",
    ]:
        assert phrase in skill

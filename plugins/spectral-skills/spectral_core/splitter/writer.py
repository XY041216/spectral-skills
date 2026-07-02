"""Write split indices and contracts."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import write_json_file


class SplitWriteError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def write_split_outputs(
    *,
    output_dir: str | Path,
    sample_ids: list[str],
    labels: list[str] | None,
    groups: list[str] | None,
    assignments: dict[str, list[int]] | None,
    folds: list[dict[str, Any]] | None,
    repeats: list[dict[str, Any]] | None,
    split_contract: dict[str, Any],
    summary: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise SplitWriteError("OUTPUT_DIR_EXISTS", "output_dir already exists and is not empty.", output_dir=str(root))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rows = _split_index_rows(
        sample_ids=sample_ids,
        labels=labels,
        groups=groups,
        split_type=str(split_contract.get("split_type") or "holdout"),
        method=str(split_contract.get("method") or ""),
        assignments=assignments,
        folds=folds,
        repeats=repeats,
    )
    _write_rows(root / "split_indices.csv", rows)
    write_json_file(root / "split_contract.json", split_contract, ensure_ascii=False)
    write_json_file(root / "split_summary.json", summary, ensure_ascii=False)
    return {
        "status": "ready",
        "output_dir": str(root),
        "written_files": ["split_indices.csv", "split_contract.json", "split_summary.json"],
        "split_contract": "split_contract.json",
        "split_indices": "split_indices.csv",
        "shape": split_contract["n_samples"],
        "method": split_contract["method"],
        "split_type": split_contract.get("split_type"),
        "handoff_ready": True,
        "next_step_hint": "Use data_contract.json plus split_contract.json for downstream preprocessing, feature, and modeling skills.",
    }


def _split_index_rows(
    *,
    sample_ids: list[str],
    labels: list[str] | None,
    groups: list[str] | None,
    split_type: str,
    method: str,
    assignments: dict[str, list[int]] | None,
    folds: list[dict[str, Any]] | None,
    repeats: list[dict[str, Any]] | None,
) -> list[list[Any]]:
    header = ["split_type", "method", "fold_id", "repeat_id", "role", "sample_index", "sample_id", "label", "group_id"]

    def row(role: str, idx: int, *, fold_id: Any = "", repeat_id: Any = "") -> list[Any]:
        return [
            split_type,
            method,
            fold_id,
            repeat_id,
            role,
            idx,
            sample_ids[idx],
            labels[idx] if labels else "",
            groups[idx] if groups else "",
        ]

    if folds is not None:
        rows: list[list[Any]] = [header]
        for fold in folds:
            for idx in fold.get("train_indices", []):
                rows.append(row("train", idx, fold_id=fold["fold_id"]))
            for idx in fold.get("val_indices", []):
                rows.append(row("val", idx, fold_id=fold["fold_id"]))
        return rows
    if repeats is not None:
        rows = [header]
        for repeat in repeats:
            for idx in repeat.get("train_indices", []):
                rows.append(row("train", idx, repeat_id=repeat["repeat_id"]))
            for idx in repeat.get("val_indices", []):
                rows.append(row("val", idx, repeat_id=repeat["repeat_id"]))
            for idx in repeat.get("test_indices", []):
                rows.append(row("test", idx, repeat_id=repeat["repeat_id"]))
        return rows
    rows = [header]
    for split_name in ["train", "val", "test"]:
        for idx in (assignments or {}).get(split_name, []):
            rows.append(row(split_name, idx))
    return rows


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

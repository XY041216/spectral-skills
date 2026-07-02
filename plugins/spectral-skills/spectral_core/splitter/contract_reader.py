"""Shared split_contract.json reader for downstream spectral skills."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file


class SplitContractError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class SplitPartition:
    iteration_type: str
    iteration_id: str
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    metadata: dict[str, Any]


@dataclass
class SplitContractInfo:
    path: Path
    contract: dict[str, Any]
    split_type: str
    method: str | None
    assignments: dict[str, list[int]]
    partitions: list[SplitPartition]
    folds: list[dict[str, Any]] | None = None
    repeats: list[dict[str, Any]] | None = None

    @property
    def train_indices(self) -> list[int]:
        return self.assignments.get("train", [])


def load_split_contract_info(path: str | Path, *, n_samples: int, sample_ids: list[str]) -> SplitContractInfo:
    contract_path = Path(path)
    if not contract_path.exists():
        raise SplitContractError("SPLIT_CONTRACT_MISSING", "split_contract.json does not exist.", path=str(contract_path))
    contract = load_json_file(contract_path)
    split_type = str(contract.get("split_type") or "holdout")
    method = str(contract.get("method")) if contract.get("method") is not None else None
    if split_type == "holdout":
        assignments = _holdout_assignments(contract, contract_path.parent, sample_ids)
        _validate_holdout(assignments, n_samples)
        partitions = [
            SplitPartition(
                iteration_type="holdout",
                iteration_id="holdout",
                train_indices=assignments["train"],
                val_indices=assignments["val"],
                test_indices=assignments["test"],
                metadata={"split_type": split_type, "method": method},
            )
        ]
        return SplitContractInfo(contract_path, contract, split_type, method, assignments, partitions)
    if split_type == "cross_validation":
        folds = contract.get("folds")
        if not isinstance(folds, list) or not folds:
            raise SplitContractError("SPLIT_FOLDS_MISSING", "cross_validation split_contract.json must include folds.")
        partitions = []
        for ordinal, fold in enumerate(folds, start=1):
            fold_id = fold.get("fold_id", ordinal) if isinstance(fold, dict) else ordinal
            train = _indices(fold.get("train_indices", []) if isinstance(fold, dict) else [], n_samples=n_samples, role="train", iteration=fold_id)
            val = _indices(fold.get("val_indices", []) if isinstance(fold, dict) else [], n_samples=n_samples, role="val", iteration=fold_id)
            test = _indices(fold.get("test_indices", []) if isinstance(fold, dict) else [], n_samples=n_samples, role="test", iteration=fold_id)
            _validate_partition(train, val, test, n_samples=n_samples, require_eval="val", iteration_id=f"fold_{int(fold_id):03d}" if _is_int_like(fold_id) else f"fold_{fold_id}")
            iteration_id = f"fold_{int(fold_id):03d}" if _is_int_like(fold_id) else f"fold_{fold_id}"
            partitions.append(SplitPartition("fold", iteration_id, train, val, test, {"split_type": split_type, "method": method, "fold_id": fold_id}))
        return SplitContractInfo(contract_path, contract, split_type, method, {}, partitions, folds=folds)
    if split_type == "repeated_holdout":
        repeats = contract.get("repeats")
        if not isinstance(repeats, list) or not repeats:
            raise SplitContractError("SPLIT_REPEATS_MISSING", "repeated_holdout split_contract.json must include repeats.")
        partitions = []
        for ordinal, repeat in enumerate(repeats, start=1):
            repeat_id = repeat.get("repeat_id", ordinal) if isinstance(repeat, dict) else ordinal
            train = _indices(repeat.get("train_indices", []) if isinstance(repeat, dict) else [], n_samples=n_samples, role="train", iteration=repeat_id)
            val = _indices(repeat.get("val_indices", []) if isinstance(repeat, dict) else [], n_samples=n_samples, role="val", iteration=repeat_id)
            test = _indices(repeat.get("test_indices", []) if isinstance(repeat, dict) else [], n_samples=n_samples, role="test", iteration=repeat_id)
            _validate_partition(train, val, test, n_samples=n_samples, require_eval="test", iteration_id=f"repeat_{int(repeat_id):03d}" if _is_int_like(repeat_id) else f"repeat_{repeat_id}")
            iteration_id = f"repeat_{int(repeat_id):03d}" if _is_int_like(repeat_id) else f"repeat_{repeat_id}"
            partitions.append(SplitPartition("repeat", iteration_id, train, val, test, {"split_type": split_type, "method": method, "repeat_id": repeat_id}))
        return SplitContractInfo(contract_path, contract, split_type, method, {}, partitions, repeats=repeats)
    raise SplitContractError("SPLIT_TYPE_UNSUPPORTED", "Unsupported split_type in split_contract.json.", split_type=split_type)


def partition_to_dict(partition: SplitPartition) -> dict[str, Any]:
    return {
        "iteration_type": partition.iteration_type,
        "iteration_id": partition.iteration_id,
        "train_indices": partition.train_indices,
        "val_indices": partition.val_indices,
        "test_indices": partition.test_indices,
        "metadata": partition.metadata,
    }


def _holdout_assignments(contract: dict[str, Any], root: Path, sample_ids: list[str]) -> dict[str, list[int]]:
    if isinstance(contract.get("indices"), dict):
        return {split_name: [int(idx) for idx in contract["indices"].get(split_name, [])] for split_name in ["train", "val", "test"]}
    if isinstance(contract.get("sample_ids"), dict):
        return {split_name: [_sample_index(sample_ids, sample_id) for sample_id in contract["sample_ids"].get(split_name, [])] for split_name in ["train", "val", "test"]}
    if isinstance(contract.get("splits"), dict):
        result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for split_name in ["train", "val", "test"]:
            for entry in contract["splits"].get(split_name) or []:
                if isinstance(entry, dict) and "index" in entry:
                    result[split_name].append(int(entry["index"]))
                elif isinstance(entry, dict) and "sample_id" in entry:
                    result[split_name].append(_sample_index(sample_ids, entry["sample_id"]))
        return result
    files = contract.get("split_files") or {}
    split_ref = files.get("split_indices")
    if split_ref:
        split_path = Path(str(split_ref))
        split_path = split_path if split_path.is_absolute() else root / split_path
        if not split_path.exists():
            raise SplitContractError("SPLIT_INDICES_MISSING", "split_indices.csv referenced by split_contract.json is missing.", path=str(split_path))
        return _read_holdout_csv(split_path, sample_ids)
    raise SplitContractError("SPLIT_ASSIGNMENTS_MISSING", "split_contract.json must include indices, sample_ids, splits, or split_files.split_indices.")


def _read_holdout_csv(path: Path, sample_ids: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split_name = str(row.get("split") or row.get("role") or "").strip().lower()
            if split_name not in result:
                raise SplitContractError("SPLIT_NAME_INVALID", "split_indices.csv contains an unsupported split name.", split=split_name)
            if row.get("index") not in {None, ""}:
                idx = int(str(row["index"]))
            elif row.get("sample_index") not in {None, ""}:
                idx = int(str(row["sample_index"]))
            else:
                idx = _sample_index(sample_ids, row.get("sample_id", ""))
            result[split_name].append(idx)
    return result


def _validate_holdout(assignments: dict[str, list[int]], n_samples: int) -> None:
    _validate_partition(assignments.get("train", []), assignments.get("val", []), assignments.get("test", []), n_samples=n_samples, require_eval="", iteration_id="holdout")
    all_indices = [idx for split_name in ["train", "val", "test"] for idx in assignments.get(split_name, [])]
    expected = set(range(n_samples))
    observed = set(all_indices)
    if observed != expected:
        raise SplitContractError("SPLIT_INCOMPLETE", "holdout split assignments must cover every sample exactly once.", missing=sorted(expected - observed), extra=sorted(observed - expected))


def _validate_partition(train: list[int], val: list[int], test: list[int], *, n_samples: int, require_eval: str, iteration_id: str) -> None:
    if not train:
        raise SplitContractError("TRAIN_SPLIT_EMPTY", "split partition must provide a non-empty train split.", iteration_id=iteration_id)
    if require_eval == "val" and not val:
        raise SplitContractError("VAL_SPLIT_EMPTY", "cross_validation fold must provide a non-empty val split.", iteration_id=iteration_id)
    if require_eval == "test" and not test:
        raise SplitContractError("TEST_SPLIT_EMPTY", "holdout or repeated_holdout partition must provide a non-empty test split.", iteration_id=iteration_id)
    all_indices = train + val + test
    if any(idx < 0 or idx >= n_samples for idx in all_indices):
        raise SplitContractError("SPLIT_INDEX_OUT_OF_RANGE", "split indices must refer to existing samples.", n_samples=n_samples, iteration_id=iteration_id)
    if len(all_indices) != len(set(all_indices)):
        raise SplitContractError("SPLIT_DUPLICATE_SAMPLE", "split partition contains duplicate samples across roles.", iteration_id=iteration_id)


def _indices(values: Any, *, n_samples: int, role: str, iteration: Any) -> list[int]:
    if not isinstance(values, list):
        raise SplitContractError("SPLIT_INDICES_INVALID", "split indices must be lists.", role=role, iteration=iteration)
    output = [int(idx) for idx in values]
    if any(idx < 0 or idx >= n_samples for idx in output):
        raise SplitContractError("SPLIT_INDEX_OUT_OF_RANGE", "split indices must refer to existing samples.", role=role, iteration=iteration, n_samples=n_samples)
    return output


def _sample_index(sample_ids: list[str], sample_id: Any) -> int:
    text = str(sample_id)
    if text not in sample_ids:
        raise SplitContractError("SPLIT_SAMPLE_UNKNOWN", "split assignments reference an unknown sample_id.", sample_id=text)
    return sample_ids.index(text)


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True

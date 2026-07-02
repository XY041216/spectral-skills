"""Read standard spectral packages and split contracts for feature engineering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file
from spectral_core.preprocess.io import (
    PreprocessInputError,
    PreprocessPackage,
    SplitInfo,
    load_preprocess_package,
    load_split_info as _load_split_info,
)


class FeatureInputError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


FeaturePackage = PreprocessPackage


def load_feature_package(package_dir: str | Path) -> FeaturePackage:
    try:
        return load_preprocess_package(package_dir)
    except PreprocessInputError as exc:
        raise FeatureInputError(exc.code, _feature_message(exc.message), **exc.details) from exc


def load_split_info(split_contract: str | Path | None, package: FeaturePackage) -> SplitInfo:
    try:
        return _load_split_info(split_contract, package)
    except PreprocessInputError as exc:
        raise FeatureInputError(exc.code, _feature_message(exc.message), **exc.details) from exc


def load_preprocess_contract_feature_inputs(
    preprocess_contract: str | Path,
    *,
    split_contract: str | Path | None = None,
) -> tuple[FeaturePackage, SplitInfo, list[dict[str, Any]]]:
    """Load preprocess output as feature inputs.

    Holdout preprocess contracts return the standard preprocessed package and
    an empty iteration list. Fold/repeat-wise contracts return the first
    iteration package plus one package per partition because train-fitted
    preprocessing may differ per fold or repeat.
    """

    contract_path = Path(preprocess_contract).resolve()
    if not contract_path.exists():
        raise FeatureInputError("PREPROCESS_CONTRACT_MISSING", "preprocess_contract.json does not exist.", path=str(contract_path))
    contract = load_json_file(contract_path)
    if contract.get("contract_type") != "preprocess_contract":
        raise FeatureInputError("PREPROCESS_CONTRACT_INVALID", "Expected a preprocess_contract.json file.", path=str(contract_path))
    iterations = contract.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        output_ref = contract.get("output_package")
        package_dir = _resolve_from(contract_path.parent, output_ref) if output_ref else contract_path.parent
        package = load_feature_package(package_dir.parent if package_dir.name == "data_contract.json" else package_dir)
        split_path = _resolve_external(split_contract) if split_contract is not None else contract.get("split_contract")
        if split_path is None:
            raise FeatureInputError("PREPROCESS_SPLIT_CONTRACT_MISSING", "Holdout preprocess_contract.json must record split_contract or receive --split-contract.", path=str(contract_path))
        split_info = load_split_info(_resolve_from(contract_path.parent, split_path), package)
        return package, split_info, []
    input_package = contract.get("input_package")
    if not input_package:
        raise FeatureInputError("PREPROCESS_INPUT_PACKAGE_MISSING", "preprocess_contract.json must record input_package.")
    input_path = _resolve_from(contract_path.parent, input_package)
    base_package = load_feature_package(input_path.parent if input_path.name == "data_contract.json" else input_path)
    split_path = _resolve_external(split_contract) if split_contract is not None else contract.get("split_contract")
    if split_path is not None:
        split_path = _resolve_from(contract_path.parent, split_path)
    split_info = load_split_info(split_path, base_package)
    by_iteration = {str(row.get("iteration_id")): row for row in iterations if isinstance(row, dict)}
    iteration_packages: list[dict[str, Any]] = []
    for partition in split_info.partitions or []:
        record = by_iteration.get(partition.iteration_id)
        if record is None:
            raise FeatureInputError("PREPROCESS_ITERATION_MISSING", "preprocess_contract.json lacks an iteration required by split_contract.json.", iteration_id=partition.iteration_id)
        package = _assemble_iteration_package(base_package, contract_path, contract, record)
        iteration_packages.append({"partition": partition, "package": package, "preprocess_iteration": record})
    if not iteration_packages:
        raise FeatureInputError("PREPROCESS_ITERATIONS_MISSING", "No usable preprocess iterations matched the split contract.")
    return iteration_packages[0]["package"], split_info, iteration_packages


def _assemble_iteration_package(base_package: FeaturePackage, contract_path: Path, contract: dict[str, Any], record: dict[str, Any]) -> FeaturePackage:
    root = contract_path.parent
    role_files = record.get("role_files")
    if not isinstance(role_files, dict):
        raise FeatureInputError("PREPROCESS_ROLE_FILES_MISSING", "preprocess iteration lacks role_files.", iteration_id=record.get("iteration_id"))

    assembled_X: list[list[float] | None] = [None] * base_package.n_samples
    feature_names: list[str] | None = None
    for role in ["train", "val", "test"]:
        indices = [int(idx) for idx in record.get(f"{role}_indices", [])]
        x_ref = role_files.get(f"X_{role}")
        if not indices and not x_ref:
            continue
        if not x_ref:
            raise FeatureInputError("PREPROCESS_ROLE_X_MISSING", "preprocess iteration is missing an X role file.", iteration_id=record.get("iteration_id"), role=role)
        names, rows = _read_X(_resolve_from(root, x_ref))
        if len(rows) != len(indices):
            raise FeatureInputError("PREPROCESS_ROLE_ROW_MISMATCH", "preprocess role X row count does not match role indices.", iteration_id=record.get("iteration_id"), role=role, expected=len(indices), observed=len(rows))
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise FeatureInputError("PREPROCESS_FEATURE_NAMES_MISMATCH", "preprocess role X headers differ within one iteration.", iteration_id=record.get("iteration_id"), role=role)
        for idx, row in zip(indices, rows):
            assembled_X[idx] = row

    missing = [idx for idx, row in enumerate(assembled_X) if row is None]
    if missing:
        raise FeatureInputError("PREPROCESS_ITERATION_INCOMPLETE", "preprocess iteration does not cover every sample in the partition.", iteration_id=record.get("iteration_id"), missing=missing)
    if feature_names is None:
        raise FeatureInputError("PREPROCESS_ROLE_X_MISSING", "preprocess iteration has no X role files.", iteration_id=record.get("iteration_id"))

    band_ref = role_files.get("band_axis") or "band_axis.csv"
    band_header, band_rows = _read_table(_resolve_from(root, band_ref))
    if len(band_rows) != len(feature_names):
        raise FeatureInputError("PREPROCESS_BAND_AXIS_MISMATCH", "preprocess iteration band_axis length must match X feature count.", iteration_id=record.get("iteration_id"), expected=len(feature_names), observed=len(band_rows))
    return PreprocessPackage(
        root=root,
        contract_path=contract_path,
        contract=contract,
        X=[row for row in assembled_X if row is not None],
        feature_names=feature_names,
        sample_ids=base_package.sample_ids,
        band_axis_header=band_header,
        band_axis_rows=band_rows,
        y_header=base_package.y_header,
        y_rows=base_package.y_rows,
        metadata_header=base_package.metadata_header,
        metadata_rows=base_package.metadata_rows,
    )


def _resolve_from(root: Path, ref: Any) -> Path:
    path = Path(str(ref))
    return (path if path.is_absolute() else root / path).resolve()


def _resolve_external(ref: str | Path) -> Path:
    """Resolve a user-supplied CLI/API path against the current working dir.

    Contract-internal relative references are resolved against the contract
    file. Explicit arguments such as --split-contract should not be re-rooted
    under preprocess_contract.parent, otherwise paths like
    ./run/03_split/split_contract.json become
    ./run/04_preprocess/run/03_split/split_contract.json.
    """

    path = Path(ref)
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def _read_csv(path: Path) -> list[list[str]]:
    if not path.exists():
        raise FeatureInputError("PREPROCESS_ROLE_FILE_MISSING", "Referenced preprocess role file is missing.", path=str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _read_X(path: Path) -> tuple[list[str], list[list[float]]]:
    rows = _read_csv(path)
    if not rows:
        raise FeatureInputError("PREPROCESS_ROLE_X_EMPTY", "Referenced preprocess X role file is empty.", path=str(path))
    return [str(value) for value in rows[0]], [[float(value) for value in row] for row in rows[1:]]


def _read_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _feature_message(message: str) -> str:
    return (
        message.replace("spectral preprocessing", "spectral feature engineering")
        .replace("before preprocessing", "before feature engineering")
        .replace("preprocessing.", "feature engineering.")
        .replace("preprocessing", "feature engineering")
    )

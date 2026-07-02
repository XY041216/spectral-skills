"""Read standard spectral packages and split contracts for modeling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
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


class ModelingInputError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class ModelingPackage:
    source: PreprocessPackage
    y_values: list[str]
    y_name: str

    @property
    def root(self) -> Path:
        return self.source.root

    @property
    def contract_path(self) -> Path:
        return self.source.contract_path

    @property
    def contract(self) -> dict[str, Any]:
        return self.source.contract

    @property
    def X(self) -> list[list[float]]:
        return self.source.X

    @property
    def sample_ids(self) -> list[str]:
        return self.source.sample_ids

    @property
    def n_samples(self) -> int:
        return self.source.n_samples

    @property
    def n_features(self) -> int:
        return self.source.n_features


def load_modeling_package(package_dir: str | Path) -> ModelingPackage:
    try:
        package = load_preprocess_package(package_dir)
    except PreprocessInputError as exc:
        raise ModelingInputError(exc.code, _modeling_message(exc.message), **exc.details) from exc
    if package.y_rows is None:
        raise ModelingInputError("Y_REQUIRED", "spectral-modeling requires y.csv with labels or target values.")
    if not package.y_rows:
        raise ModelingInputError("Y_EMPTY", "y.csv must contain at least one target row.")
    if any(len(row) != 1 for row in package.y_rows):
        raise ModelingInputError("MULTI_TARGET_NOT_IMPLEMENTED", "This first modeling version supports one y column only.")
    y_values = [str(row[0]).strip() for row in package.y_rows]
    if any(value == "" for value in y_values):
        raise ModelingInputError("Y_MISSING_VALUES", "y.csv contains empty target values.")
    y_name = package.y_header[0] if package.y_header else "y"
    return ModelingPackage(source=package, y_values=y_values, y_name=y_name)


def load_modeling_split(split_contract: str | Path | None, package: ModelingPackage) -> SplitInfo:
    if split_contract is None:
        raise ModelingInputError("SPLIT_CONTRACT_REQUIRED", "spectral-modeling requires split_contract.json.")
    try:
        return _load_split_info(split_contract, package.source)
    except PreprocessInputError as exc:
        raise ModelingInputError(exc.code, _modeling_message(exc.message), **exc.details) from exc


def load_modeling_iteration_packages_from_contract(contract_path: str | Path) -> tuple[dict[str, ModelingPackage], SplitInfo, ModelingPackage]:
    path = Path(contract_path)
    if not path.exists():
        raise ModelingInputError("ITERATION_CONTRACT_MISSING", "feature_contract.json or preprocess_contract.json does not exist.", path=str(path))
    contract = load_json_file(path)
    contract_type = str(contract.get("contract_type") or "")
    if contract_type not in {"feature_contract", "preprocess_contract"}:
        raise ModelingInputError("ITERATION_CONTRACT_INVALID", "Expected feature_contract.json or preprocess_contract.json.", path=str(path), contract_type=contract_type)
    iterations = contract.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ModelingInputError("ITERATION_CONTRACT_EMPTY", "Fold/repeat-wise contract must include iterations.")
    input_package = contract.get("input_package")
    if not input_package:
        raise ModelingInputError("ITERATION_INPUT_PACKAGE_MISSING", "Fold/repeat-wise contract must record input_package.")
    split_contract = contract.get("split_contract")
    if not split_contract:
        raise ModelingInputError("ITERATION_SPLIT_CONTRACT_MISSING", "Fold/repeat-wise contract must record split_contract.")

    base_modeling = _base_modeling_package(path.parent, input_package)
    split_info = load_modeling_split(_resolve_from(path.parent, split_contract), base_modeling)
    by_iteration = {str(row.get("iteration_id")): row for row in iterations if isinstance(row, dict)}
    packages: dict[str, ModelingPackage] = {}
    for partition in split_info.partitions or []:
        record = by_iteration.get(partition.iteration_id)
        if record is None:
            raise ModelingInputError("ITERATION_RECORD_MISSING", "Input contract lacks an iteration required by split_contract.json.", iteration_id=partition.iteration_id)
        packages[partition.iteration_id] = _assemble_iteration_modeling_package(base_modeling, path, contract, record)
    if not packages:
        raise ModelingInputError("ITERATION_CONTRACT_EMPTY", "No usable iterations matched the split contract.")
    first = packages[(split_info.partitions or [])[0].iteration_id]
    return packages, split_info, first


def load_modeling_inputs_from_contract(contract_path: str | Path) -> tuple[dict[str, ModelingPackage] | None, SplitInfo, ModelingPackage]:
    """Load a feature/preprocess contract for modeling.

    Fold/repeat-wise contracts return per-iteration packages. Holdout
    preprocess/feature contracts return a standard package plus its split_info.
    Relative references are resolved against the contract file location.
    """

    path = Path(contract_path)
    if not path.exists():
        raise ModelingInputError("CONTRACT_MISSING", "feature_contract.json or preprocess_contract.json does not exist.", path=str(path))
    contract = load_json_file(path)
    contract_type = str(contract.get("contract_type") or "")
    if contract_type not in {"feature_contract", "preprocess_contract"}:
        raise ModelingInputError("CONTRACT_INVALID", "Expected feature_contract.json or preprocess_contract.json.", path=str(path), contract_type=contract_type)
    iterations = contract.get("iterations")
    if isinstance(iterations, list) and iterations:
        packages, split_info, package = load_modeling_iteration_packages_from_contract(path)
        return packages, split_info, package

    output_ref = contract.get("output_package")
    input_ref = output_ref or contract.get("input_package")
    if not input_ref:
        raise ModelingInputError("CONTRACT_INPUT_PACKAGE_MISSING", "Holdout feature/preprocess contract must record output_package or input_package.", path=str(path))
    package = _base_modeling_package(path.parent, input_ref)
    package = _with_contract_source(package, path, contract)
    split_ref = contract.get("split_contract")
    if not split_ref:
        raise ModelingInputError("CONTRACT_SPLIT_CONTRACT_MISSING", "Holdout feature/preprocess contract must record split_contract.", path=str(path))
    split_info = load_modeling_split(_resolve_from(path.parent, split_ref), package)
    return None, split_info, package


def _with_contract_source(package: ModelingPackage, contract_path: Path, contract: dict[str, Any]) -> ModelingPackage:
    source = PreprocessPackage(
        root=package.root,
        contract_path=contract_path,
        contract=contract,
        X=package.X,
        feature_names=package.source.feature_names,
        sample_ids=package.sample_ids,
        band_axis_header=package.source.band_axis_header,
        band_axis_rows=package.source.band_axis_rows,
        y_header=[package.y_name],
        y_rows=[[value] for value in package.y_values],
        metadata_header=package.source.metadata_header,
        metadata_rows=package.source.metadata_rows,
    )
    return ModelingPackage(source=source, y_values=package.y_values, y_name=package.y_name)


def _base_modeling_package(root: Path, input_ref: Any) -> ModelingPackage:
    input_path = _resolve_from(root, input_ref)
    if input_path.name == "data_contract.json":
        input_path = input_path.parent
    if input_path.name in {"preprocess_contract.json", "feature_contract.json"}:
        contract = load_json_file(input_path)
        nested_ref = contract.get("input_package")
        if nested_ref:
            return _base_modeling_package(input_path.parent, nested_ref)
    return load_modeling_package(input_path)


def _assemble_iteration_modeling_package(base_package: ModelingPackage, contract_path: Path, contract: dict[str, Any], record: dict[str, Any]) -> ModelingPackage:
    root = contract_path.parent
    role_files = record.get("role_files")
    if not isinstance(role_files, dict):
        raise ModelingInputError("ITERATION_ROLE_FILES_MISSING", "Iteration contract lacks role_files.", iteration_id=record.get("iteration_id"))
    assembled_X: list[list[float] | None] = [None] * base_package.n_samples
    feature_names: list[str] | None = None
    sample_ids = list(base_package.sample_ids)
    y_rows = [[value] for value in base_package.y_values]
    for role in ["train", "val", "test"]:
        indices = [int(idx) for idx in record.get(f"{role}_indices", [])]
        x_ref = _x_role_ref(role_files, role)
        if not indices and not x_ref:
            continue
        if not x_ref:
            raise ModelingInputError("ITERATION_ROLE_X_MISSING", "Iteration contract is missing an X role file.", iteration_id=record.get("iteration_id"), role=role)
        names, rows = _read_X(_resolve_from(root, x_ref))
        if len(rows) != len(indices):
            raise ModelingInputError("ITERATION_ROLE_ROW_MISMATCH", "Iteration X role row count does not match role indices.", iteration_id=record.get("iteration_id"), role=role, expected=len(indices), observed=len(rows))
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ModelingInputError("ITERATION_FEATURE_NAMES_MISMATCH", "Iteration X headers differ within one iteration.", iteration_id=record.get("iteration_id"), role=role)
        for idx, row in zip(indices, rows):
            assembled_X[idx] = row
    missing = [idx for idx, row in enumerate(assembled_X) if row is None]
    if missing:
        raise ModelingInputError("ITERATION_CONTRACT_INCOMPLETE", "Iteration contract does not provide X rows for every sample in the partition.", iteration_id=record.get("iteration_id"), missing=missing)
    if feature_names is None:
        raise ModelingInputError("ITERATION_ROLE_X_MISSING", "Iteration contract has no X role files.", iteration_id=record.get("iteration_id"))
    axis_ref = role_files.get("feature_axis") or role_files.get("band_axis") or "band_axis.csv"
    axis_header, axis_rows = _read_table(_resolve_from(root, axis_ref))
    source = PreprocessPackage(
        root=root,
        contract_path=contract_path,
        contract=contract,
        X=[row for row in assembled_X if row is not None],
        feature_names=feature_names,
        sample_ids=sample_ids,
        band_axis_header=axis_header,
        band_axis_rows=axis_rows,
        y_header=[base_package.y_name],
        y_rows=y_rows,
        metadata_header=base_package.source.metadata_header,
        metadata_rows=base_package.source.metadata_rows,
    )
    return ModelingPackage(source=source, y_values=base_package.y_values, y_name=base_package.y_name)


def _x_role_ref(role_files: dict[str, Any], role: str) -> Any:
    return role_files.get(f"X_{role}_features") or role_files.get(f"X_{role}")


def _resolve_from(root: Path, ref: Any) -> Path:
    path = Path(str(ref))
    return path if path.is_absolute() else root / path


def _read_csv(path: Path) -> list[list[str]]:
    if not path.exists():
        raise ModelingInputError("ITERATION_ROLE_FILE_MISSING", "Referenced iteration role file is missing.", path=str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle)]


def _read_X(path: Path) -> tuple[list[str], list[list[float]]]:
    rows = _read_csv(path)
    if not rows:
        raise ModelingInputError("ITERATION_ROLE_X_EMPTY", "Referenced iteration X role file is empty.", path=str(path))
    return [str(value) for value in rows[0]], [[float(value) for value in row] for row in rows[1:]]


def _read_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _modeling_message(message: str) -> str:
    return (
        message.replace("spectral preprocessing", "spectral modeling")
        .replace("before preprocessing", "before modeling")
        .replace("preprocessing.", "modeling.")
        .replace("preprocessing", "modeling")
    )

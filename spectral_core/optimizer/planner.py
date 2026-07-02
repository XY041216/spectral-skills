"""Candidate expansion and budget checks for spectral optimization."""

from __future__ import annotations

import itertools
from typing import Any

from spectral_core.modeling.registry import model_spec


def expand_options(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in items:
        scalar = {key: value for key, value in item.items() if not isinstance(value, list)}
        grids = {key: value for key, value in item.items() if isinstance(value, list)}
        if not grids:
            expanded.append(dict(scalar))
            continue
        keys = list(grids)
        for values in itertools.product(*(grids[key] for key in keys)):
            expanded.append({**scalar, **dict(zip(keys, values))})
    return expanded


def build_trials(space: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    preprocess = expand_options(space.get("preprocess") or [{"method": "none"}])
    feature = expand_options(space.get("feature") or [{"method": "none"}])
    modeling = expand_options(space.get("modeling") or [])
    if not modeling:
        raise ValueError("candidate_space must include at least one modeling candidate")
    trials: list[dict[str, Any]] = []
    for idx, (pre, feat, model) in enumerate(itertools.product(preprocess, feature, modeling), start=1):
        try:
            family = model_spec(model["method"]).family
        except Exception:
            family = "unknown"
        trials.append(
            {
                "trial_id": f"trial_{idx:04d}",
                "preprocess_method": pre.get("method", "none"),
                "feature_method": feat.get("method", "none"),
                "model_method": model.get("method"),
                "model_family": family,
                "preprocess_params": {key: value for key, value in pre.items() if key != "method"},
                "feature_params": {key: value for key, value in feat.items() if key != "method"},
                "model_params": {key: value for key, value in model.items() if key != "method"},
                "selection_metric": metric,
                "test_used_for_selection": False,
                "status": "planned",
                "warnings": "",
            }
        )
    return trials


def budget_audit(*, expanded_trials: int, max_trials: int, confirmed: bool) -> dict[str, Any]:
    exceeded = expanded_trials > max_trials
    return {
        "requested_max_trials": int(max_trials),
        "expanded_trials": int(expanded_trials),
        "budget_confirmed": bool(confirmed),
        "budget_exceeded": exceeded,
        "status": "needs_confirmation" if exceeded and not confirmed else "ready",
        "reason": "candidate space expansion exceeds confirmed budget" if exceeded and not confirmed else None,
    }

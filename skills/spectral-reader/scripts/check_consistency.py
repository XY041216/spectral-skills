"""CLI entry for spectral-reader first-stage consistency checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_runtime_root() -> Path:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent)
        candidates.append(parent / "spectral-core")
    for candidate in candidates:
        if (candidate / "spectral_core" / "__init__.py").is_file():
            return candidate
    return here.parents[3]


ROOT = _find_runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.reader.consistency import check_consistency


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check spectral-reader scripts, MCP, manifest, schema, and docs consistency.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = check_consistency(repo_root=ROOT, backend="script")
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

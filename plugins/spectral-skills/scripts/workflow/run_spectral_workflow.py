"""Fallback CLI entry for spectral workflow orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills" / "spectral-workflow" / "scripts" / "run_spectral_workflow.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from run_spectral_workflow import main


if __name__ == "__main__":
    raise SystemExit(main())

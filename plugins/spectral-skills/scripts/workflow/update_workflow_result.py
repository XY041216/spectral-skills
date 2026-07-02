"""Fallback CLI entry for workflow result updates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "skills" / "spectral-workflow" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from update_workflow_result import main


if __name__ == "__main__":
    raise SystemExit(main())

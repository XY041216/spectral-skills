"""Fallback CLI entry for spectral optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills" / "spectral-optimizer" / "scripts" / "optimize_spectral_pipeline.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from optimize_spectral_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

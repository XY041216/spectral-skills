"""Fallback CLI entry for leakage-safe spectral modeling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills" / "spectral-modeling" / "scripts" / "model_spectral_package.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from model_spectral_package import main


if __name__ == "__main__":
    raise SystemExit(main())

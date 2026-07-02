"""Fallback CLI for spectral-splitter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPT = ROOT / "skills" / "spectral-splitter" / "scripts" / "split_spectral_package.py"
if str(SKILL_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPT.parent))

from split_spectral_package import main


if __name__ == "__main__":
    raise SystemExit(main())

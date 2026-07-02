"""Global fallback CLI for spectral-reader server health."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills" / "spectral-reader" / "scripts" / "server_health.py"


def main() -> int:
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

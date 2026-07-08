"""CLI entry for leakage-safe preprocessing of split standard spectral packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_runtime_root() -> Path:
    here = Path(__file__).resolve()
    candidates = list(here.parents) + [parent / "spectral-core" for parent in here.parents]
    for candidate in candidates:
        if (candidate / "spectral_core" / "__init__.py").is_file():
            return candidate
    return here.parents[3]


ROOT = _find_runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.preprocess.workflow import preprocess_spectral_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess a standard spectral package using train-only fitted parameters.")
    parser.add_argument("--package-dir", default=".")
    parser.add_argument("--split-contract")
    parser.add_argument("--output-dir")
    parser.add_argument("--methods", help="Comma-separated methods such as snv,standardization.")
    parser.add_argument("--window-length", type=int)
    parser.add_argument("--polyorder", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--poly-degree", type=int)
    parser.add_argument("--als-lambda", type=float)
    parser.add_argument("--als-p", type=float)
    parser.add_argument("--als-iter", type=int)
    parser.add_argument("--band-range")
    parser.add_argument("--remove-band-ranges")
    parser.add_argument("--confirm-baseline", action="store_true")
    parser.add_argument("--confirm-absorbance", action="store_true")
    parser.add_argument("--confirm-band-change", action="store_true")
    parser.add_argument("--confirm-unsplit-fit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = preprocess_spectral_package(
        package_dir=args.package_dir,
        split_contract=args.split_contract,
        output_dir=args.output_dir,
        methods=args.methods,
        window_length=args.window_length,
        polyorder=args.polyorder,
        sigma=args.sigma,
        poly_degree=args.poly_degree,
        als_lambda=args.als_lambda,
        als_p=args.als_p,
        als_iter=args.als_iter,
        band_range=args.band_range,
        remove_band_ranges=args.remove_band_ranges,
        confirm_baseline=args.confirm_baseline,
        confirm_absorbance=args.confirm_absorbance,
        confirm_band_change=args.confirm_band_change,
        confirm_unsplit_fit=args.confirm_unsplit_fit,
        overwrite=args.overwrite,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

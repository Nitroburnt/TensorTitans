"""
main.py — CLI entry point for Business Entity Resolution pipeline.

Usage:
  # Inference only (model.pkl must already exist in output_dir):
  python code/business_entity_resolution/src/main.py \\
      --test_dir student_resource/dataset/test \\
      --output_dir output/

  # Inference with auto-train if no model found:
  python code/business_entity_resolution/src/main.py \\
      --test_dir student_resource/dataset/test \\
      --output_dir output/ \\
      --train_dir student_resource/dataset/train

  # Train only (no inference):
  python code/business_entity_resolution/src/main.py \\
      --train_only \\
      --train_dir student_resource/dataset/train \\
      --output_dir output/
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# ── sys.path fix for script execution ────────────────────────────────────────
# When invoked as `python code/business_entity_resolution/src/main.py`
# the workspace root is NOT on sys.path, so relative imports fail.
# We add the workspace root (3 levels up from this file) to sys.path
# unconditionally so the package is importable whether run as a script
# or as a module (`python -m code.business_entity_resolution.src.main`).
_THIS_FILE = pathlib.Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[3]  # .../student_resource/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


# Import after sys.path fix — supports both script and module invocation
from code.business_entity_resolution.src.pipeline import run_train, run_inference


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Business Entity Resolution — Amazon ML Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to test data directory (required unless --train_only).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to output directory (created if it does not exist).",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to train data directory. Used for auto-training when "
            "model.pkl is missing, or required with --train_only."
        ),
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        default=False,
        help="Run training pipeline only; skip inference.",
    )
    parser.add_argument(
        "--skip_cache",
        action="store_true",
        default=False,
        help="Ignore existing blocking/feature caches and recompute from scratch.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Override decision threshold τ (default: load from tau_star.txt "
            "or use 0.78). Range [0.40, 0.95]."
        ),
    )

    return parser


def main() -> None:
    """Entry point — parse args and dispatch to pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.train_only and args.train_dir is None:
        parser.error("--train_only requires --train_dir")

    if not args.train_only and args.test_dir is None:
        parser.error("--test_dir is required unless --train_only is set")

    # ── Create output_dir ─────────────────────────────────────────────────────
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_wall = time.time()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.train_only:
        print(f"[main] Mode: TRAIN ONLY")
        print(f"[main] train_dir  : {args.train_dir}")
        print(f"[main] output_dir : {args.output_dir}")
        run_train(
            train_dir=args.train_dir,
            output_dir=output_dir,
            skip_blocking_cache=args.skip_cache,
        )
    else:
        print(f"[main] Mode: INFERENCE")
        print(f"[main] test_dir   : {args.test_dir}")
        print(f"[main] output_dir : {args.output_dir}")
        if args.train_dir:
            print(f"[main] train_dir  : {args.train_dir} (auto-train if no model.pkl)")
        if args.tau is not None:
            print(f"[main] tau override: {args.tau:.4f}")

        # If tau override is requested, patch tau_star.txt before inference
        if args.tau is not None:
            tau_path = output_dir / "tau_star.txt"
            tau_path.write_text(str(args.tau))
            print(f"[main] Wrote tau override {args.tau:.4f} → {tau_path}")

        run_inference(
            test_dir=args.test_dir,
            output_dir=output_dir,
            train_dir=args.train_dir,
        )

    elapsed = time.time() - t_wall
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n[main] Total wall-clock time: {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    main()

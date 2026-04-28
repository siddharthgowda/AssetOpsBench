"""CLI entry point for the battery MCP server profiler.

Examples
--------
# Capture baseline (no model weights required for preprocessing + stats sections):
uv run python -m servers.battery.profiling.run_profile --label baseline

# Capture baseline including TF model inference (requires weights):
uv run python -m servers.battery.profiling.run_profile --label baseline --with-model

# Profile only specific sections:
uv run python -m servers.battery.profiling.run_profile --label baseline \\
    --sections preprocessing stats

# Quick smoke-test (fewer cycles, skip slow model section):
uv run python -m servers.battery.profiling.run_profile --label smoke \\
    --n-cycles 20 --sections preprocessing sliding_windows stats memory

# After implementing optimizations, capture the post run:
uv run python -m servers.battery.profiling.run_profile --label optimized --with-model

# Compare the two:
uv run python -m servers.battery.profiling.compare \\
    "profiles/baseline_*.json" "profiles/optimized_*.json"

# Include TF op-level trace (saved for TensorBoard):
uv run python -m servers.battery.profiling.run_profile \\
    --label baseline --with-model --tf-trace
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("battery-profiler")

_ALL_SECTIONS = ["preprocessing", "sliding_windows", "inference", "stats", "memory"]

_SECTION_ALIASES = {
    # Allow short-hand names on the CLI
    "prep": "preprocessing",
    "preprocess": "preprocessing",
    "sw": "sliding_windows",
    "windows": "sliding_windows",
    "infer": "inference",
    "model": "inference",
    "stat": "stats",
    "statistical": "stats",
    "mem": "memory",
}


def _parse_sections(raw: list[str]) -> list[str]:
    resolved = []
    for s in raw:
        canonical = _SECTION_ALIASES.get(s.lower(), s.lower())
        if canonical not in _ALL_SECTIONS:
            raise argparse.ArgumentTypeError(
                f"Unknown section '{s}'. Choose from: {', '.join(_ALL_SECTIONS)}"
            )
        resolved.append(canonical)
    return resolved or _ALL_SECTIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m servers.battery.profiling.run_profile",
        description=(
            "Profile the battery MCP server and save results to a JSON file.\n"
            "Run twice (before and after optimizations) then compare with compare.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--label",
        default="baseline",
        help='Short label for this run (e.g. "baseline", "optimized"). Default: baseline',
    )
    parser.add_argument(
        "--output-dir",
        default="profiles",
        metavar="DIR",
        help="Directory to save the JSON result (default: profiles/)",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=None,
        metavar="SECTION",
        help=(
            f"Sections to run. Choices: {', '.join(_ALL_SECTIONS)}. "
            "Default: all sections."
        ),
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=3,
        metavar="N",
        help="Number of synthetic cells to use (default: 3). Higher = more realistic boot timing.",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Cycles per cell (default: 100). "
            "Must be ≥100 for model inference; use fewer for quick preprocessing profiling."
        ),
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help=(
            "Attempt to load TF model weights from BATTERY_MODEL_WEIGHTS_DIR. "
            "Required for the 'inference' section. Skipped gracefully if weights are missing."
        ),
    )
    parser.add_argument(
        "--tf-trace",
        action="store_true",
        help=(
            "Record a TF op-level trace with tf.profiler.experimental "
            "(requires --with-model). Saved to <output-dir>/tf_profile/ for TensorBoard."
        ),
    )
    parser.add_argument(
        "--tf-trace-dir",
        default=None,
        metavar="DIR",
        help="Override directory for TF profiler output (default: <output-dir>/tf_profile).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary printed to stdout.",
    )

    args = parser.parse_args(argv)

    # Resolve sections
    try:
        sections = _parse_sections(args.sections) if args.sections else _ALL_SECTIONS
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Validate n_cycles for inference
    if "inference" in sections and args.n_cycles < 100:
        logger.warning(
            "--n-cycles=%d is below 100. The model inference section needs ≥100 cycles "
            "for the sliding-window; it will be skipped or may raise a ValueError.",
            args.n_cycles,
        )

    logger.info("Battery profiler starting  label=%s  sections=%s", args.label, sections)

    from servers.battery.profiling.profiler import BatteryProfiler

    profiler = BatteryProfiler(
        label=args.label,
        output_dir=args.output_dir,
        n_cells=args.n_cells,
        n_cycles=args.n_cycles,
        use_real_model=args.with_model,
        verbose=not args.quiet,
    )

    results = profiler.run_all(
        sections=sections,
        tf_trace=args.tf_trace,
        tf_trace_dir=args.tf_trace_dir,
    )

    thr = results.get("sections", {}).get("throughput_summary", {})
    if thr.get("inference_ms_per_cycle"):
        logger.info(
            "Inference: %.4f ms/cycle  (target <13 ms from battery.md)",
            thr["inference_ms_per_cycle"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

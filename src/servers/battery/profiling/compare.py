"""Pre/post optimization comparison for battery profiler results.

Loads two JSON profile files produced by profiler.py and prints a side-by-side
table with absolute values, delta, and percentage improvement.

Usage
-----
As a CLI (two concrete files):
    python -m servers.battery.profiling.compare \\
        profiles/baseline_20260428T120000Z.json \\
        profiles/optimized_20260428T130000Z.json

As a CLI with glob expansion (picks newest match automatically):
    python -m servers.battery.profiling.compare \\
        "profiles/baseline_*.json" "profiles/optimized_*.json"

As a Python import:
    from servers.battery.profiling.compare import load_profile, compare_profiles
    report = compare_profiles(baseline_path, optimized_path)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Optional


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_path(pattern: str) -> Path:
    """Resolve a literal path or a glob pattern to the most recently modified file."""
    p = Path(pattern)
    if p.exists():
        return p
    matches = sorted(glob.glob(pattern), key=lambda f: Path(f).stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return Path(matches[-1])


def load_profile(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _extract_flat_metrics(profile: dict) -> dict[str, float]:
    """Flatten the nested 'sections' and 'memory' dicts into a single {key: value} dict.

    Only numeric leaves are included. Keys are dot-separated paths, e.g.
    ``"preprocessing.inp_500.mean_ms"``.
    """
    flat: dict[str, float] = {}

    def _walk(node: dict, prefix: str) -> None:
        for k, v in node.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _walk(v, full_key)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[full_key] = float(v)

    secs = profile.get("sections", {})
    _walk(secs, "")
    mem = profile.get("memory", {})
    _walk(mem, "memory")
    return flat


# ── comparison logic ───────────────────────────────────────────────────────────

_LOWER_IS_BETTER = {
    "mean_ms", "total_ms", "min_ms", "max_ms", "ms_per_cycle", "ms_per_row",
    "ms_per_cell", "memory_net_mb", "net_alloc_mb",
    "preprocessing_one_cell_net_alloc_mb", "process_rss_mb", "process_vms_mb",
}

_HIGHER_IS_BETTER = {
    "throughput_per_sec", "calls_per_sec", "cycles_per_sec",
    "boot_cells_per_min", "cell_preprocessing_throughput_cycles_per_sec",
    "inp_500_calls_per_sec", "preprocess_cycles_per_sec",
}


def _improvement_direction(key: str) -> str:
    """Return 'lower' or 'higher' based on the metric leaf name."""
    leaf = key.rsplit(".", 1)[-1]
    if leaf in _LOWER_IS_BETTER or any(kw in leaf for kw in ("ms", "alloc", "rss", "vms")):
        return "lower"
    if leaf in _HIGHER_IS_BETTER or any(kw in leaf for kw in ("throughput", "per_sec", "per_min")):
        return "higher"
    return "lower"  # conservative default: assume lower is better


def compare_profiles(
    baseline_path: str | Path,
    optimized_path: str | Path,
    min_pct_change: float = 1.0,
    show_unchanged: bool = True,
) -> dict:
    """Compare two profile JSON files and return a structured report dict.

    Parameters
    ----------
    baseline_path:
        Path (or glob pattern) for the baseline profile JSON.
    optimized_path:
        Path (or glob pattern) for the optimized profile JSON.
    min_pct_change:
        Omit rows where |delta %| < this threshold (reduces noise).
    show_unchanged:
        Include rows where the value didn't change meaningfully.

    Returns
    -------
    dict with keys:
        ``baseline_label``, ``optimized_label``, ``rows``, ``summary``
    Each row: ``{metric, baseline, optimized, delta, pct_change, status}``
    """
    baseline = load_profile(_resolve_path(str(baseline_path)))
    optimized = load_profile(_resolve_path(str(optimized_path)))

    b_flat = _extract_flat_metrics(baseline)
    o_flat = _extract_flat_metrics(optimized)

    all_keys = sorted(set(b_flat) | set(o_flat))
    rows = []

    for key in all_keys:
        b_val = b_flat.get(key)
        o_val = o_flat.get(key)
        if b_val is None or o_val is None:
            continue
        if b_val == 0.0:
            pct = 0.0
        else:
            pct = (o_val - b_val) / abs(b_val) * 100.0
        delta = o_val - b_val
        direction = _improvement_direction(key)
        if direction == "lower":
            improved = delta < 0
            regressed = delta > 0
        else:
            improved = delta > 0
            regressed = delta < 0

        if not show_unchanged and abs(pct) < min_pct_change:
            continue

        if abs(pct) < min_pct_change:
            status = "—"
        elif improved:
            status = "IMPROVED"
        elif regressed:
            status = "REGRESSED"
        else:
            status = "—"

        rows.append(
            {
                "metric": key,
                "baseline": round(b_val, 4),
                "optimized": round(o_val, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct, 2),
                "status": status,
            }
        )

    improved_rows = [r for r in rows if r["status"] == "IMPROVED"]
    regressed_rows = [r for r in rows if r["status"] == "REGRESSED"]

    return {
        "baseline_label": baseline.get("label", "baseline"),
        "baseline_timestamp": baseline.get("timestamp", ""),
        "optimized_label": optimized.get("label", "optimized"),
        "optimized_timestamp": optimized.get("timestamp", ""),
        "rows": rows,
        "summary": {
            "total_metrics": len(rows),
            "improved": len(improved_rows),
            "regressed": len(regressed_rows),
            "unchanged": len(rows) - len(improved_rows) - len(regressed_rows),
        },
    }


# ── pretty printer ─────────────────────────────────────────────────────────────

_ANSI_GREEN = "\033[92m"
_ANSI_RED = "\033[91m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"


def _colorize(text: str, status: str, use_color: bool) -> str:
    if not use_color:
        return text
    if status == "IMPROVED":
        return f"{_ANSI_GREEN}{text}{_ANSI_RESET}"
    if status == "REGRESSED":
        return f"{_ANSI_RED}{text}{_ANSI_RESET}"
    return f"{_ANSI_DIM}{text}{_ANSI_RESET}"


def print_report(
    report: dict,
    use_color: bool = True,
    group_by_section: bool = True,
    min_pct_abs: float = 0.5,
) -> None:
    """Print the comparison report as a formatted table."""
    b_label = report["baseline_label"]
    o_label = report["optimized_label"]
    rows = [r for r in report["rows"] if abs(r["pct_change"]) >= min_pct_abs or r["status"] != "—"]

    width = 110
    col_metric = 52
    col_val = 12
    col_pct = 10
    col_status = 10

    header = (
        f"{'METRIC':<{col_metric}} "
        f"{b_label[:col_val]:>{col_val}} "
        f"{o_label[:col_val]:>{col_val}} "
        f"{'DELTA %':>{col_pct}} "
        f"{'STATUS':>{col_status}}"
    )

    def _bold(s: str) -> str:
        return f"{_ANSI_BOLD}{s}{_ANSI_RESET}" if use_color else s

    print()
    print(_bold("=" * width))
    print(_bold(f"  Battery MCP Server — Profiling Comparison"))
    print(_bold(f"  Baseline:  {b_label}  ({report.get('baseline_timestamp', '')})"))
    print(_bold(f"  Optimized: {o_label}  ({report.get('optimized_timestamp', '')})"))
    print(_bold("=" * width))
    print()
    print(_bold(header))
    print("-" * width)

    current_section = ""
    for row in rows:
        metric = row["metric"]
        # Section header
        top_section = metric.split(".")[0]
        if group_by_section and top_section != current_section:
            current_section = top_section
            section_title = f"\n  [{current_section.upper()}]"
            print(_bold(section_title) if use_color else section_title)

        pct = row["pct_change"]
        pct_str = f"{pct:+.1f}%"
        status = row["status"]

        metric_short = metric[len(top_section) + 1:] if group_by_section else metric
        line = (
            f"  {metric_short:<{col_metric - 2}} "
            f"{row['baseline']:>{col_val}.4g} "
            f"{row['optimized']:>{col_val}.4g} "
            f"{pct_str:>{col_pct}} "
            f"{status:>{col_status}}"
        )
        print(_colorize(line, status, use_color))

    print()
    print("-" * width)
    summ = report["summary"]
    improved = summ["improved"]
    regressed = summ["regressed"]
    unchanged = summ["unchanged"]
    summary_line = (
        f"  {summ['total_metrics']} metrics  |  "
        f"IMPROVED: {improved}  |  REGRESSED: {regressed}  |  UNCHANGED: {unchanged}"
    )
    if use_color and regressed > 0:
        print(f"{_ANSI_RED}{summary_line}{_ANSI_RESET}")
    elif use_color and improved > 0:
        print(f"{_ANSI_GREEN}{summary_line}{_ANSI_RESET}")
    else:
        print(summary_line)
    print("=" * width + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two battery profiler JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("baseline", help="Path or glob for baseline profile JSON")
    parser.add_argument("optimized", help="Path or glob for optimized profile JSON")
    parser.add_argument(
        "--min-pct", type=float, default=0.5,
        help="Minimum |delta %%| to show in table (default: 0.5)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour output"
    )
    parser.add_argument(
        "--no-group", action="store_true", help="Don't group rows by section"
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Also save the report as JSON to FILE",
    )
    args = parser.parse_args(argv)

    try:
        report = compare_profiles(args.baseline, args.optimized)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(
        report,
        use_color=not args.no_color,
        group_by_section=not args.no_group,
        min_pct_abs=args.min_pct,
    )

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report saved → {args.save}")

    return 1 if report["summary"]["regressed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

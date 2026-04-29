#!/usr/bin/env python3
"""Run a small, repeatable scenario diagnostic.

For each scenario index, runs `plan-execute` as a subprocess, captures stdout/
stderr to a log file, and times the full call. Boots the battery server with a
limited cell subset (BATTERY_BOOT_CELL_SUBSET) so cold-start tax is bounded.

Outputs land under profiles/scenarios_<timestamp>/:
  - summary.json — env, config, per-scenario timings, exit codes
  - scenario_<n>.log — raw stdout+stderr of the plan-execute subprocess

Usage:
    python scripts/run_scenario_diagnostic.py
    python scripts/run_scenario_diagnostic.py --scenarios 0,1,3 \\
        --cells B0005,B0006,B0018 --timeout 240
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCENARIOS = "0,1,3"  # 0-indexed → scenarios 1, 2, 4
_DEFAULT_CELLS = "B0005,B0006,B0018"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _env_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenarios",
        default=_DEFAULT_SCENARIOS,
        help=f"Comma-separated 0-indexed scenario IDs (default: {_DEFAULT_SCENARIOS})",
    )
    ap.add_argument(
        "--cells",
        default=_DEFAULT_CELLS,
        help=f"BATTERY_BOOT_CELL_SUBSET value (default: {_DEFAULT_CELLS})",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-scenario timeout in seconds (default: 300)",
    )
    ap.add_argument(
        "--scenarios-file",
        type=Path,
        default=_REPO_ROOT / "battery_scenarios.json",
    )
    ap.add_argument("--label", default="scenarios")
    ap.add_argument(
        "--profiles-dir",
        type=Path,
        default=_REPO_ROOT / "profiles",
    )
    args = ap.parse_args()

    scenarios_data = json.loads(args.scenarios_file.read_text(encoding="utf-8"))
    if not isinstance(scenarios_data, list):
        print(
            f"ERROR: {args.scenarios_file} not a top-level array",
            file=sys.stderr,
        )
        return 2

    indices = [int(x.strip()) for x in args.scenarios.split(",") if x.strip()]
    for i in indices:
        if i < 0 or i >= len(scenarios_data):
            print(f"ERROR: scenario index {i} out of range (0..{len(scenarios_data) - 1})", file=sys.stderr)
            return 2

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = args.profiles_dir / f"{args.label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BATTERY_BOOT_CELL_SUBSET"] = args.cells

    summary: dict[str, Any] = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "env": _env_info(),
        "config": {
            "scenarios_indices": indices,
            "cell_subset": args.cells,
            "per_scenario_timeout_s": args.timeout,
        },
        "results": [],
    }

    print(f"Output dir: {out_dir}")
    print(f"Cell subset: {args.cells}")
    print(f"Scenarios: {indices}\n")

    for i in indices:
        sc = scenarios_data[i]
        query = sc.get("query", "")
        persona = sc.get("persona", "?")
        focus = sc.get("focus", "?")
        log_path = out_dir / f"scenario_{i + 1}.log"

        cmd = ["uv", "run", "plan-execute", "--show-plan", "--show-times", query]
        print(f"=== Scenario {i + 1}: {persona} ({focus}) ===")
        print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd[:3] + ['...'])}")
        print(f"  log: {log_path}")

        start_wall = time.perf_counter()
        timed_out = False
        rc: int | None = None
        try:
            with open(log_path, "wb") as fh:
                fh.write(f"=== Scenario {i + 1}: {persona} ===\n".encode())
                fh.write(f"=== Query ===\n{query}\n\n=== Output ===\n".encode())
                fh.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=_REPO_ROOT,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -1

        elapsed = time.perf_counter() - start_wall
        print(f"  elapsed: {elapsed:.2f}s  exit={rc}  timeout={timed_out}\n")

        # Snip log preview
        log_preview = ""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_preview = text[-2000:] if len(text) > 2000 else text
        except Exception as e:
            log_preview = f"<read-failed: {e}>"

        summary["results"].append(
            {
                "scenario_index": i,
                "scenario_number": i + 1,
                "persona": persona,
                "focus": focus,
                "query_chars": len(query),
                "wall_seconds": round(elapsed, 3),
                "exit_code": rc,
                "timed_out": timed_out,
                "log_path": str(log_path.relative_to(_REPO_ROOT)),
                "log_tail": log_preview,
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")

    # Print short table
    print("\n=== Scenario timing summary ===")
    print(f"{'#':<3}{'persona':<35}{'wall_s':>10}{'exit':>6}")
    for r in summary["results"]:
        print(
            f"{r['scenario_number']:<3}{(r['persona'] or '')[:34]:<35}"
            f"{r['wall_seconds']:>10.2f}{str(r['exit_code']):>6}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

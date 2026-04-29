"""Battery diagnostic profile — find bottlenecks in cold-start preprocessing.

Times each phase of preprocess_cell_from_couchdb for N cells with memory and
cProfile breakdowns. Writes JSON to profiles/ in a shape compatible with
scripts/compare_battery_profiles.py.

Usage:
    uv run python -m servers.battery.profile_diagnostic \\
        --cell-ids B0005,B0006,B0018 --repeats 3
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_PHASE_ORDER = (
    "fetch_charge",
    "fetch_discharge",
    "sort_filter",
    "preprocess_cycle_loop",
    "stack_arrays",
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _stats_ms(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(sum(samples) / len(samples)) * 1000,
        "min": float(min(samples)) * 1000,
        "max": float(max(samples)) * 1000,
        "n": len(samples),
    }


def _env_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
    }


def time_phases(cell_id: str, client) -> dict[str, float | int]:
    """Run preprocess_cell pipeline phase-by-phase, return seconds per phase."""
    from servers.battery.preprocessing import preprocess_cycle

    out: dict[str, float | int] = {}

    t0 = time.perf_counter()
    charges = client.fetch_cycles(cell_id, cycle_type="charge") or []
    out["fetch_charge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    discharges = client.fetch_cycles(cell_id, cycle_type="discharge") or []
    out["fetch_discharge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    charges = sorted(charges, key=lambda c: c.get("cycle_index", 0))
    discharges = sorted(discharges, key=lambda c: c.get("cycle_index", 0))
    out["sort_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = min(len(charges), len(discharges), 100)
    ch_list, dis_list = [], []
    for i in range(n):
        c_t = preprocess_cycle(charges[i].get("data", {}))
        d_t = preprocess_cycle(discharges[i].get("data", {}))
        if c_t is None or d_t is None:
            continue
        ch_list.append(c_t)
        dis_list.append(d_t)
    out["preprocess_cycle_loop"] = time.perf_counter() - t0
    out["_n_charge_docs"] = len(charges)
    out["_n_discharge_docs"] = len(discharges)
    out["_n_clean_pairs"] = len(ch_list)

    t0 = time.perf_counter()
    if ch_list:
        np.array(ch_list[:100])
        np.array(dis_list[:100])
    out["stack_arrays"] = time.perf_counter() - t0

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cell-ids",
        default="B0005,B0006,B0018",
        help="Comma-separated cell IDs (default: B0005,B0006,B0018)",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="diagnostic")
    ap.add_argument("--profiles-dir", type=Path, default=None)
    args = ap.parse_args()

    from servers.battery.couchdb_client import CouchDBClient
    from servers.battery.preprocessing import preprocess_cell_from_couchdb

    cells = [c.strip() for c in args.cell_ids.split(",") if c.strip()]
    if not cells:
        print("ERROR: no --cell-ids", file=sys.stderr)
        return 2

    client = CouchDBClient()
    if not client.available:
        print(
            "ERROR: CouchDB unavailable.\n"
            "  Start: docker-compose -f docker-compose.couchdb.yml up -d\n"
            "  Seed:  ./src/couchdb/couchdb_setup.sh",
            file=sys.stderr,
        )
        return 1

    phase_samples: dict[str, list[float]] = {k: [] for k in _PHASE_ORDER}
    total_samples: list[float] = []
    cell_meta: dict[str, dict] = {}
    skipped: list[dict] = []

    for cell in cells:
        for rep in range(args.repeats):
            try:
                phases = time_phases(cell, client)
                for k in _PHASE_ORDER:
                    phase_samples[k].append(float(phases[k]))
                if rep == 0:
                    cell_meta[cell] = {
                        "n_charge_docs": int(phases["_n_charge_docs"]),
                        "n_discharge_docs": int(phases["_n_discharge_docs"]),
                        "n_clean_pairs": int(phases["_n_clean_pairs"]),
                    }
            except Exception as e:
                skipped.append({"cell": cell, "rep": rep, "phase": "phases", "error": str(e)})
                continue

            try:
                t0 = time.perf_counter()
                preprocess_cell_from_couchdb(cell, client)
                total_samples.append(time.perf_counter() - t0)
            except ValueError as e:
                cell_meta.setdefault(cell, {})["preprocess_error"] = str(e)
            except Exception as e:
                skipped.append({"cell": cell, "rep": rep, "phase": "total", "error": str(e)})

    # Memory: peak alloc for one full call on first cell
    mem_peak_mb = 0.0
    try:
        tracemalloc.start()
        try:
            preprocess_cell_from_couchdb(cells[0], client)
        except Exception:
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_peak_mb = peak / 1e6
    except Exception:
        try:
            tracemalloc.stop()
        except Exception:
            pass

    # cProfile: top 15 cumtime functions for one full call
    cprofile_text = ""
    try:
        pr = cProfile.Profile()
        pr.enable()
        try:
            preprocess_cell_from_couchdb(cells[0], client)
        except Exception:
            pass
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(15)
        cprofile_text = buf.getvalue()
    except Exception as e:
        cprofile_text = f"cProfile failed: {e}"

    measurements: dict[str, dict[str, Any]] = {
        k: {"wall_ms": _stats_ms(v)} for k, v in phase_samples.items()
    }
    measurements["total_preprocess_cell"] = {"wall_ms": _stats_ms(total_samples)}

    total_mean = measurements["total_preprocess_cell"]["wall_ms"]["mean"]
    if total_mean > 0:
        for k in _PHASE_ORDER:
            measurements[k]["pct_of_total"] = round(
                measurements[k]["wall_ms"]["mean"] / total_mean * 100, 1
            )

    output: dict[str, Any] = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "cell_ids": cells,
            "repeats": args.repeats,
            "data_source": "real_couchdb",
        },
        "env": _env_info(),
        "measurements": measurements,
        "cell_meta": cell_meta,
        "memory_peak_mb_one_cell": round(mem_peak_mb, 2),
        "cprofile_top15": cprofile_text,
        "skipped": skipped,
    }

    profiles_dir = args.profiles_dir or (_REPO_ROOT / "profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = profiles_dir / f"{args.label}_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Human-readable report
    print()
    print("=== Diagnostic profile ===")
    print(f"cells={cells}  repeats={args.repeats}  git_sha={output['git_sha'][:8]}")
    print()
    print("WALL TIME BREAKDOWN (mean ms, % of total):")
    for k in _PHASE_ORDER:
        m = measurements[k]
        wm = m["wall_ms"]
        pct = m.get("pct_of_total", 0.0)
        print(f"  {k:<28} {wm['mean']:8.2f} ms  ({pct:5.1f}%)")
    tm = measurements["total_preprocess_cell"]["wall_ms"]
    print(f"  {'TOTAL':<28} {tm['mean']:8.2f} ms  (n={tm['n']})")
    print()
    print(f"MEMORY peak (1 cell): {mem_peak_mb:.2f} MB")
    print()
    print("CELL META:")
    for c, m in cell_meta.items():
        print(f"  {c}: {m}")
    if skipped:
        print()
        print(f"SKIPPED: {len(skipped)} run(s)")
        for s in skipped[:5]:
            print(f"  {s}")
    print()
    print("CPU HOT FUNCTIONS (cProfile cumtime, top 15):")
    for line in cprofile_text.splitlines()[:30]:
        print(f"  {line}")
    if total_mean > 0 and any(phase_samples[k] for k in _PHASE_ORDER):
        winner = max(
            _PHASE_ORDER,
            key=lambda k: (sum(phase_samples[k]) / len(phase_samples[k])) if phase_samples[k] else 0,
        )
        print()
        print("DIAGNOSIS:")
        print(f"  Dominant phase: {winner} ({measurements[winner].get('pct_of_total', 0)}% of wall)")
    print()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

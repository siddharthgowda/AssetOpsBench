"""Cold-start probe — three small experiments to diagnose cold-start cost.

A) Order swap: in a fresh client, is the FIRST fetch slow regardless of type?
B) Repeated fetch: how much does the same fetch speed up on calls 2..N?
C) Boot decomposition: _load_once() vs preprocess_cell_from_couchdb vs precompute_cell.

Writes profiles/cold_start_probe_<timestamp>.json + prints a one-line diagnosis
per experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
    ap.add_argument("--profiles-dir", type=Path, default=_REPO_ROOT / "profiles")
    ap.add_argument("--label", default="cold_start_probe")
    args = ap.parse_args()

    from servers.battery.couchdb_client import CouchDBClient
    from servers.battery.preprocessing import preprocess_cell_from_couchdb

    if not CouchDBClient().available:
        print("ERROR: CouchDB unreachable.", file=sys.stderr)
        return 1

    # ── Experiment A: order swap ────────────────────────────────────────
    print("[A] Order-swap test (fresh client per pair)")
    a_results = []
    pairs = [
        ("B0005", "discharge", "charge"),
        ("B0006", "charge", "discharge"),
        ("B0018", "discharge", "charge"),
    ]
    for cell, first_t, second_t in pairs:
        c = CouchDBClient()
        t0 = time.perf_counter()
        c.fetch_cycles(cell, cycle_type=first_t)
        first_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        c.fetch_cycles(cell, cycle_type=second_t)
        second_ms = (time.perf_counter() - t0) * 1000

        ratio = round(first_ms / second_ms, 2) if second_ms > 0 else None
        a_results.append(
            {
                "cell": cell,
                "first_type": first_t,
                "first_ms": round(first_ms, 2),
                "second_type": second_t,
                "second_ms": round(second_ms, 2),
                "ratio_first_over_second": ratio,
            }
        )
        print(
            f"  {cell}: first={first_t} {first_ms:7.1f}ms   "
            f"second={second_t} {second_ms:7.1f}ms   ratio={ratio}"
        )

    # ── Experiment B: repeated fetch ────────────────────────────────────
    print("\n[B] Repeated fetch (5 calls, same client, B0005 discharge)")
    c = CouchDBClient()
    b_times = []
    for i in range(5):
        t0 = time.perf_counter()
        c.fetch_cycles("B0005", cycle_type="discharge")
        ms = (time.perf_counter() - t0) * 1000
        b_times.append(round(ms, 2))
        print(f"  call {i + 1}: {ms:7.1f}ms")
    b_speedup = round(b_times[0] / b_times[-1], 2) if b_times[-1] > 0 else None

    # ── Experiment C: boot decomposition ────────────────────────────────
    print("\n[C] Boot decomposition (fresh client)")

    # C1 — _load_once() includes the first `import tensorflow` and weight load
    t0 = time.perf_counter()
    from servers.battery.model_wrapper import _load_once

    _load_once()
    c1_ms = (time.perf_counter() - t0) * 1000
    print(f"  C1 _load_once (TF import + 4 .h5 + 4 .npy): {c1_ms:8.1f}ms")

    from servers.battery import model_wrapper as _mw

    c_block: dict[str, Any]
    if not _mw._MODEL_AVAILABLE:
        print("  WARNING: model not available — skipping C2/C3")
        c_block = {"load_once_ms": round(c1_ms, 2), "model_available": False}
    else:
        from servers.battery.model_wrapper import precompute_cell

        # C2 — data-side preprocessing only (no model)
        client = CouchDBClient()
        t0 = time.perf_counter()
        ch, dis, summ = preprocess_cell_from_couchdb("B0005", client)
        c2_ms = (time.perf_counter() - t0) * 1000
        print(f"  C2 preprocess_cell_from_couchdb (B0005): {c2_ms:8.1f}ms")

        # C3 — model-side only (TF inference + windowing)
        t0 = time.perf_counter()
        precompute_cell("B0005", ch, dis, summ)
        c3_ms = (time.perf_counter() - t0) * 1000
        print(f"  C3 precompute_cell (TF inference):       {c3_ms:8.1f}ms")

        total = c1_ms + c2_ms + c3_ms
        print(f"  ── SUM:                                  {total:8.1f}ms")

        c_block = {
            "load_once_ms": round(c1_ms, 2),
            "preprocess_cell_ms": round(c2_ms, 2),
            "precompute_cell_ms": round(c3_ms, 2),
            "sum_ms": round(total, 2),
            "model_available": True,
        }

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "env": _env_info(),
        "experiments": {
            "A_order_swap": {
                "description": "Fresh client per pair; tests whether the FIRST fetch is slow regardless of cycle type (session warmup) vs charge intrinsically heavier (payload).",
                "results": a_results,
            },
            "B_repeated_fetch": {
                "description": "5 sequential fetches in one client; tests warm/cold gap for caching value.",
                "wall_ms_per_call": b_times,
                "warm_over_cold_speedup": b_speedup,
            },
            "C_boot_decomposition": c_block,
        },
    }

    args.profiles_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = args.profiles_dir / f"{args.label}_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # ── One-line diagnosis per experiment ───────────────────────────────
    print("\n=== Diagnosis ===")
    n_first_slow = sum(
        1 for r in a_results if r["ratio_first_over_second"] and r["ratio_first_over_second"] > 2
    )
    if n_first_slow >= 2:
        print(
            f"A: first call slow in {n_first_slow}/{len(a_results)} cells → "
            "session warmup theory holds (charge isn't intrinsically slower)."
        )
    else:
        print(
            f"A: first call NOT consistently slow → "
            "type matters more than position (charge payload likely heavier)."
        )

    if b_speedup is not None:
        if b_speedup > 5:
            verdict = "caching is HIGH value — strong warm/cold gap."
        elif b_speedup > 2:
            verdict = "caching is moderately useful."
        else:
            verdict = "caching pointless — cost is mostly fixed."
        print(f"B: warm/cold speedup {b_speedup}× → {verdict}")

    if c_block.get("model_available"):
        parts = [
            ("TF load", c_block["load_once_ms"]),
            ("preprocess", c_block["preprocess_cell_ms"]),
            ("precompute", c_block["precompute_cell_ms"]),
        ]
        winner = max(parts, key=lambda p: p[1])
        share = winner[1] / c_block["sum_ms"] * 100
        print(
            f"C: boot dominated by '{winner[0]}' ({winner[1]:.0f}ms, "
            f"{share:.0f}% of {c_block['sum_ms']:.0f}ms total)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

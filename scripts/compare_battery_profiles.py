#!/usr/bin/env python3
"""Side-by-side JSON profile diff; flags measurement deltas above a threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _means_for_blob(blob: dict, use_cpu: bool) -> dict[str, float]:
    key = "cpu_process_ms" if use_cpu else "wall_ms"
    out: dict[str, float] = {}
    for name, sub in blob.items():
        if not isinstance(sub, dict):
            continue
        wm = sub.get(key)
        if isinstance(wm, dict) and "mean" in wm:
            out[name] = float(wm["mean"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two battery profile JSON files.")
    ap.add_argument("a_json", type=Path)
    ap.add_argument("b_json", type=Path)
    ap.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Percent change to flag (default: 5).",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Compare cpu_process_ms means instead of wall_ms.",
    )
    args = ap.parse_args()

    a = json.loads(args.a_json.read_text(encoding="utf-8"))
    b = json.loads(args.b_json.read_text(encoding="utf-8"))
    metric = "cpu_process_ms" if args.cpu else "wall_ms"
    ma = _means_for_blob(a.get("measurements", {}), args.cpu)
    mb = _means_for_blob(b.get("measurements", {}), args.cpu)
    keys = sorted(set(ma) | set(mb))
    pct = args.threshold
    print(f"A: {args.a_json.name}  label={a.get('label')}  ts={a.get('timestamp')}")
    print(f"B: {args.b_json.name}  label={b.get('label')}  ts={b.get('timestamp')}")
    print(f"Metric: {metric}")
    print(f"{'key':<44} {'A':>12} {'B':>12} {'Δ%':>8}")
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None:
            print(f"{k:<44} {va or 0:>12.4f} {vb or 0:>12.4f} {'n/a':>8}")
            continue
        if va == 0:
            print(f"{k:<44} {va:>12.4f} {vb:>12.4f} {'n/a':>8}")
            continue
        delta_pct = 100.0 * (vb - va) / va
        flag = " ***" if abs(delta_pct) > pct else ""
        print(f"{k:<44} {va:>12.4f} {vb:>12.4f} {delta_pct:>7.1f}%{flag}")


if __name__ == "__main__":
    main()

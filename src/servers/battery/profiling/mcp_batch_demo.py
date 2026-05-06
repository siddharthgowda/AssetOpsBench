"""Compare N per-cell ``predict_rul`` MCP calls vs one ``predict_rul_batch`` MCP call.

Quantifies the subprocess-spawn savings of the batch tool. No planner, no LLM —
direct ``_call_tool`` invocation, deterministic.

Run::

    uv run python -m servers.battery.profiling.mcp_batch_demo
    uv run python -m servers.battery.profiling.mcp_batch_demo \\
        --cells B0005,B0006,B0018 --repeats 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_BATTERY_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BATTERY_DIR.parent.parent.parent
_DEFAULT_PROFILES_DIR = _BATTERY_DIR / "profiles"

load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_CELLS = [
    "B0005", "B0006", "B0007", "B0018",
    "B0033", "B0034", "B0036",
    "B0054", "B0055", "B0056",
]


async def run(cells: list[str], repeats: int) -> dict:
    from agent.plan_execute.executor import _call_tool

    walls_per_cell: list[float] = []
    walls_batched: list[float] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        for aid in cells:
            await _call_tool("battery-mcp-server", "predict_rul", {"asset_id": aid})
        walls_per_cell.append(time.perf_counter() - t0)

    for _ in range(repeats):
        t0 = time.perf_counter()
        await _call_tool(
            "battery-mcp-server", "predict_rul_batch", {"asset_ids": cells}
        )
        walls_batched.append(time.perf_counter() - t0)

    per_cell = float(np.mean(walls_per_cell))
    batched = float(np.mean(walls_batched))
    return {
        "n_cells": len(cells),
        "n_repeats": repeats,
        "cells": cells,
        "mcp_per_cell": {
            "wall_s": round(per_cell, 3),
            "n_subprocess_spawns": len(cells),
        },
        "mcp_batched": {
            "wall_s": round(batched, 3),
            "n_subprocess_spawns": 1,
        },
        "speedup": round(per_cell / batched, 2) if batched else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cells",
        default=",".join(_DEFAULT_CELLS),
        help="Comma-separated asset IDs (default: 10 NASA model-ready cells).",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed repeats (default: 1). Each per-cell run is ~14s × N cells, so this gets slow fast.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: profiles/mcp_batch_demo_<ts>.json).",
    )
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out = asyncio.run(run(cells, args.repeats))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = args.output or (_DEFAULT_PROFILES_DIR / f"mcp_batch_demo_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

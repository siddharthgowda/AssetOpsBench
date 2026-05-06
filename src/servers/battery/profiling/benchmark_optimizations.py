"""5-rung ablation of the RUL pipeline optimizations.

Each rung adds one optimization on top of the previous:

  rung 1  naive_baseline      serial fetch, raw Keras, per-cell predict, no cache
  rung 2  + parallel_fetch    + ThreadPoolExecutor over CouchDB fetches
  rung 3  + graph_precompile  + flexible-shape compiled tf.function graphs
  rung 4  + batched_predict   + concat all cells → 3 TF predicts total
  rung 5  + disk_cache        + cached rul_trajectory per cell (warm hits skip TF)

Run::

    uv run python -m servers.battery.profiling.benchmark_optimizations
    uv run python -m servers.battery.profiling.benchmark_optimizations \\
        --repeats 3 --cells B0005,B0006,B0018
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def serial_fetch(cells, client):
    from servers.battery.preprocessing import preprocess_cell_from_couchdb

    out = []
    for aid in cells:
        try:
            ch, dis, summ = preprocess_cell_from_couchdb(aid, client)
            out.append((aid, ch, dis, summ))
        except Exception:  # noqa: BLE001
            pass
    return out


def parallel_fetch(cells, client, workers: int = 4):
    from servers.battery.preprocessing import preprocess_cell_from_couchdb

    def _one(aid):
        return aid, *preprocess_cell_from_couchdb(aid, client)

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, aid): aid for aid in cells}
        for fut in as_completed(futures):
            try:
                out.append(fut.result())
            except Exception:  # noqa: BLE001
                pass
    order = {aid: i for i, aid in enumerate(cells)}
    out.sort(key=lambda t: order.get(t[0], 999))
    return out


def do_one_pass(cells, client, use_parallel_fetch, use_compiled, batched, use_cache, mw):
    fetch_t0 = time.perf_counter()
    if use_parallel_fetch:
        cells_ok = parallel_fetch(cells, client)
    else:
        cells_ok = serial_fetch(cells, client)
    fetch_s = time.perf_counter() - fetch_t0

    predict_t0 = time.perf_counter()
    if use_cache:
        misses_data = []
        for aid, ch, dis, summ in cells_ok:
            n_ch, n_dis = int(ch.shape[0]), int(dis.shape[0])
            cached = mw.cache_load(aid, n_ch, n_dis)
            if cached is None:
                misses_data.append((aid, ch, dis, summ, n_ch, n_dis))
        if misses_data:
            tensors = [(ch, dis, summ) for _, ch, dis, summ, _, _ in misses_data]
            trajs = mw.predict_rul_for_cells(tensors, use_compiled=use_compiled, batched=batched)
            for (aid, _, _, _, n_ch, n_dis), traj in zip(misses_data, trajs):
                mw.cache_save(aid, traj, n_ch, n_dis)
    else:
        cells_data = [(ch, dis, summ) for _, ch, dis, summ in cells_ok]
        mw.predict_rul_for_cells(cells_data, use_compiled=use_compiled, batched=batched)
    predict_s = time.perf_counter() - predict_t0
    return fetch_s, predict_s


def run_rung(
    *,
    cells: list[str],
    use_parallel_fetch: bool,
    use_compiled: bool,
    batched: bool,
    use_cache: bool,
    repeats: int,
):
    from servers.battery import model_wrapper as mw
    from servers.battery.couchdb_client import CouchDBClient

    mw.cache_clear()

    client = CouchDBClient()
    walls: list[float] = []
    fetches: list[float] = []
    predicts: list[float] = []

    # Warmup populates the cache when use_cache=True; timed repeats then hit it.
    do_one_pass(cells, client, use_parallel_fetch, use_compiled, batched, use_cache, mw)

    for _ in range(repeats):
        t0 = time.perf_counter()
        f, p = do_one_pass(
            cells, client, use_parallel_fetch, use_compiled, batched, use_cache, mw
        )
        walls.append(time.perf_counter() - t0)
        fetches.append(f)
        predicts.append(p)

    return {
        "wall_s": round(float(np.mean(walls)), 3),
        "fetch_s": round(float(np.mean(fetches)), 3),
        "predict_s": round(float(np.mean(predicts)), 3),
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
        default=3,
        help="Timed repeats per rung (default: 3). One untimed warmup runs before each rung.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: profiles/benchmark_<ts>.json).",
    )
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    from servers.battery import model_wrapper as mw

    mw._load_once()
    if not mw.model_available():
        print("Models unavailable. Set BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR.")
        return 1
    mw.get_compiled_models()

    rung_specs = [
        ("naive_baseline",     False, False, False, False),
        ("+ parallel_fetch",   True,  False, False, False),
        ("+ graph_precompile", True,  True,  False, False),
        ("+ batched_predict",  True,  True,  True,  False),
        ("+ disk_cache",       True,  True,  True,  True),
    ]

    run_dir = args.output or (_DEFAULT_PROFILES_DIR / f"benchmark_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, pf, gp, bp, ca in rung_specs:
        m = run_rung(
            cells=cells,
            use_parallel_fetch=pf,
            use_compiled=gp,
            batched=bp,
            use_cache=ca,
            repeats=args.repeats,
        )
        rung_record = {
            "name": name,
            "parallel_fetch": pf,
            "graph_precompile": gp,
            "batched_predict": bp,
            "disk_cache": ca,
            "n_cells": len(cells),
            "n_repeats": args.repeats,
            "cells": cells,
            **m,
        }
        fname = name.lstrip("+ ").strip().replace(" ", "_") + ".json"
        (run_dir / fname).write_text(json.dumps(rung_record, indent=2), encoding="utf-8")
        print(f"  wrote {run_dir / fname}  wall={rung_record['wall_s']}s")

    mw.cache_clear()
    print(f"\nWrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

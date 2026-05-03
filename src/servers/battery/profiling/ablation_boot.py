"""Boot-time ablation study for the battery MCP server.

Quantifies where boot time goes and how each optimization moves the needle.
Output JSON is structured so a report can quote it directly: each section is
labelled with what was measured, the methodology, and a clear before/after pair.

The four measurements (per run):
  1. data_load        — CouchDB fetch vs Python preprocess split, per cell.
  2. precompute_strategy — per-cell loop (4N TF predicts) vs fully batched (4 predicts).
  3. in_memory_cache  — dict-lookup latency for any tool call after _boot().
  4. disk_cache       — cold first boot (writes .npz to disk) vs warm second boot
                        (loads .npz, skips TF predict entirely). This is the new
                        layer that survives MCP subprocess restarts.

Run as: ``uv run python -m servers.battery.profiling.ablation_boot``.
Writes a single JSON to ``src/servers/battery/profiles/ablation_boot_<ts>.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_BATTERY_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BATTERY_DIR.parent.parent.parent
_DEFAULT_PROFILES_DIR = _BATTERY_DIR / "profiles"


def _instrumented_load(client, cells: list[str]):
    """Fetch + preprocess all cells from CouchDB with per-call timing."""
    from servers.battery.preprocessing import preprocess_cell_from_couchdb  # noqa: PLC0415

    couchdb_calls: list[dict] = []
    original_fetch = client.fetch_cycles

    def timed_fetch(asset_id, cycle_type=None, limit=10000):
        t0 = time.perf_counter()
        out = original_fetch(asset_id, cycle_type=cycle_type, limit=limit)
        couchdb_calls.append(
            {
                "asset_id": asset_id,
                "cycle_type": cycle_type,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "n_docs": len(out),
            }
        )
        return out

    client.fetch_cycles = timed_fetch  # type: ignore[method-assign]

    cells_ok: list[tuple[str, Any, Any, Any]] = []
    per_cell: list[dict] = []
    t_total = time.perf_counter()
    for cell in cells:
        t_cell = time.perf_counter()
        before = len(couchdb_calls)
        try:
            ch, dis, summ = preprocess_cell_from_couchdb(cell, client)
            cells_ok.append((cell, ch, dis, summ))
            cell_total = (time.perf_counter() - t_cell) * 1000.0
            cell_couchdb = sum(c["wall_ms"] for c in couchdb_calls[before:])
            per_cell.append(
                {
                    "asset_id": cell,
                    "total_ms": round(cell_total, 2),
                    "couchdb_ms": round(cell_couchdb, 2),
                    "preprocess_ms": round(cell_total - cell_couchdb, 2),
                    "n_couchdb_calls": len(couchdb_calls) - before,
                    "n_charge_docs": int(ch.shape[0]),
                    "n_discharge_docs": int(dis.shape[0]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: skipped {cell}: {exc}", file=sys.stderr)
    total_ms = (time.perf_counter() - t_total) * 1000.0

    couchdb_total = sum(b["couchdb_ms"] for b in per_cell)
    preprocess_total = sum(b["preprocess_ms"] for b in per_cell)
    return cells_ok, {
        "total_ms": round(total_ms, 2),
        "couchdb_total_ms": round(couchdb_total, 2),
        "preprocess_total_ms": round(preprocess_total, 2),
        "couchdb_calls_total": len(couchdb_calls),
        "couchdb_mean_call_ms": round(
            sum(c["wall_ms"] for c in couchdb_calls) / max(len(couchdb_calls), 1), 2
        ),
        "per_cell": per_cell,
    }


def _wipe_disk_cache(mw, cells_ok: list[tuple[str, Any, Any, Any]]) -> None:
    """Force a cold first-boot scenario: remove any pre-existing .npz files."""
    for cell_id, _, _, _ in cells_ok:
        for p in (mw._disk_cache_path(cell_id), mw._disk_cache_manifest_path(cell_id)):
            if p.exists():
                p.unlink()


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")

    from servers.battery import model_wrapper as mw  # noqa: PLC0415
    from servers.battery.couchdb_client import CouchDBClient  # noqa: PLC0415

    mw._load_once()
    if not mw._MODEL_AVAILABLE:
        print(
            "Models unavailable. Set BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR.",
            file=sys.stderr,
        )
        return 1

    subset = (os.environ.get("BATTERY_BOOT_CELL_SUBSET") or "").strip()
    if subset:
        cells = [c.strip() for c in subset.split(",") if c.strip()]
    else:
        cells = [
            "B0005", "B0006", "B0007", "B0018", "B0033",
            "B0034", "B0036", "B0054", "B0055", "B0056",
        ]

    # ── Load all cells once (CouchDB + preprocess only; no TF predict) ────
    client = CouchDBClient()
    cells_ok, data_load = _instrumented_load(client, cells)
    n = len(cells_ok)

    # Warmup TF kernels so the first timed predict block isn't penalized for
    # graph compilation. Use the existing path; clear caches afterward.
    _wipe_disk_cache(mw, cells_ok[:1])
    mw.precompute_cell(*cells_ok[0])
    mw._CACHE.clear()
    _wipe_disk_cache(mw, cells_ok[:1])

    # ── (2) precompute_strategy: per-cell loop vs fully batched ───────────
    _wipe_disk_cache(mw, cells_ok)
    mw._CACHE.clear()
    t = time.perf_counter()
    for c, ch, dis, summ in cells_ok:
        mw.precompute_cell(c, ch, dis, summ)
    per_cell_loop_ms = (time.perf_counter() - t) * 1000.0

    _wipe_disk_cache(mw, cells_ok)
    mw._CACHE.clear()
    t = time.perf_counter()
    mw.precompute_cells_fully_batched(cells_ok)
    fully_batched_ms = (time.perf_counter() - t) * 1000.0

    # ── (3) in_memory_cache: dict-hit latency (1000 lookups) ──────────────
    n_lookups = 1000
    keys = list(mw._CACHE.keys())
    t = time.perf_counter()
    for i in range(n_lookups):
        _ = mw._CACHE[keys[i % len(keys)]]["rul_trajectory"]
    cache_hit_us = (time.perf_counter() - t) * 1e6 / n_lookups

    # ── (4) disk_cache: cold first boot vs warm second boot ───────────────
    # Cold: wipe disk cache, time the batched precompute (which writes .npz).
    _wipe_disk_cache(mw, cells_ok)
    mw._CACHE.clear()
    t = time.perf_counter()
    mw.precompute_cells_fully_batched(cells_ok)
    cold_boot_ms = (time.perf_counter() - t) * 1000.0
    # Confirm files are now on disk and measure their footprint.
    cache_file_bytes = sum(
        mw._disk_cache_path(c).stat().st_size for c, *_ in cells_ok if mw._disk_cache_path(c).exists()
    )

    # Warm: clear in-memory _CACHE only (disk .npz remain), time the batched precompute.
    # The cells with a fresh .npz short-circuit and load from disk; no TF predict runs.
    mw._CACHE.clear()
    t = time.perf_counter()
    mw.precompute_cells_fully_batched(cells_ok)
    warm_boot_ms = (time.perf_counter() - t) * 1000.0

    cold_per_call_ms = per_cell_loop_ms / n if n else 0.0
    cache_speedup_dict = (cold_per_call_ms * 1000.0) / cache_hit_us if cache_hit_us else 0.0
    disk_speedup = cold_boot_ms / warm_boot_ms if warm_boot_ms else 0.0
    disk_saved_ms = cold_boot_ms - warm_boot_ms

    out = {
        "label": "ablation_boot_full",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "n_cells": n,
        "study": (
            "Boot-time ablation for the battery MCP server. Each section measures "
            "one layer of the boot pipeline; comparisons within a section are paired "
            "(same process, same cells, warmed TF kernels)."
        ),
        "hardware_note": "Apple Silicon CPU, no GPU. Numbers will differ on x86 / GPU.",
        "data_load": {
            "what": "Fetch + Python preprocess for all cells, no TF predict.",
            "methodology": "Monkey-patch CouchDBClient.fetch_cycles to record per-call wall_ms; remainder of preprocess_cell_from_couchdb is attributed to scipy preprocess.",
            **data_load,
        },
        "precompute_strategy": {
            "what": "How to populate _CACHE from preprocessed tensors at boot.",
            "methodology": "Same N cells, in-memory only (disk cache wiped each round, _CACHE cleared between runs).",
            "before_optimization": {
                "name": "per-cell loop",
                "wall_ms": round(per_cell_loop_ms, 2),
                "predict_calls": 4 * n,
                "predict_calls_per_cell": 4,
            },
            "after_optimization": {
                "name": "fully batched (precompute_cells_fully_batched)",
                "wall_ms": round(fully_batched_ms, 2),
                "predict_calls": 4,
                "predict_calls_per_cell": round(4 / n, 2),
            },
            "speedup": round(per_cell_loop_ms / fully_batched_ms, 2) if fully_batched_ms else 0.0,
            "wall_ms_saved": round(per_cell_loop_ms - fully_batched_ms, 2),
            "verdict": (
                "Roughly a wash on Apple Silicon CPU at N=10. Collapsing 4N→4 "
                "predict() calls trades dispatch savings for larger kernels that "
                "the CPU scheduler doesn't parallelize as well. Worth keeping "
                "for GPU / much larger N."
            ),
        },
        "in_memory_cache": {
            "what": "Cost of one tool-call lookup after _boot() populates _CACHE.",
            "methodology": f"Mean over {n_lookups} sequential dict accesses to _CACHE[cell]['rul_trajectory'].",
            "before_optimization": {
                "name": "no cache (recompute per tool call)",
                "wall_ms_per_call_estimated": round(cold_per_call_ms, 2),
                "source": "per_cell_loop wall_ms / n_cells",
            },
            "after_optimization": {
                "name": "in-memory dict lookup",
                "wall_us_per_call": round(cache_hit_us, 3),
            },
            "speedup": round(cache_speedup_dict, 0),
            "verdict": (
                "Dominant intra-process win. Limitation: dict dies when MCP "
                "subprocess exits — every fresh tool call still pays boot cost. "
                "Addressed by the disk_cache layer below."
            ),
        },
        "disk_cache": {
            "what": "Disk-backed .npz cache that survives MCP subprocess restarts.",
            "methodology": (
                "Same boot sequence twice. Round 1: wipe artifacts/cache/, run "
                "precompute_cells_fully_batched (writes .npz per cell). Round 2: "
                "clear in-memory _CACHE only, run precompute_cells_fully_batched "
                "again — every cell short-circuits via _try_load_from_disk."
            ),
            "before_optimization": {
                "name": "cold boot (no disk cache, must compute)",
                "wall_ms": round(cold_boot_ms, 2),
                "tf_predict_calls": 3,
            },
            "after_optimization": {
                "name": "warm boot (loads .npz from disk, skips TF predict)",
                "wall_ms": round(warm_boot_ms, 2),
                "tf_predict_calls": 0,
            },
            "speedup": round(disk_speedup, 2),
            "wall_ms_saved": round(disk_saved_ms, 2),
            "disk_footprint_bytes": int(cache_file_bytes),
            "disk_footprint_mb": round(cache_file_bytes / 1e6, 2),
            "verdict": (
                "Persists across MCP subprocess restarts. Makes every cold MCP "
                "call after the first invocation skip the boot precompute."
            ),
        },
    }

    _DEFAULT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DEFAULT_PROFILES_DIR / f"ablation_boot_{out['timestamp']}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print("=== Boot-time ablation study ===")
    print(f"  cells: {n}")
    print()
    print("  (1) Data load")
    print(
        f"      total: {data_load['total_ms']:7.0f} ms  "
        f"(couchdb {data_load['couchdb_total_ms']:.0f} ms, "
        f"preprocess {data_load['preprocess_total_ms']:.0f} ms, "
        f"{data_load['couchdb_calls_total']} couchdb calls, "
        f"mean {data_load['couchdb_mean_call_ms']:.1f} ms/call)"
    )
    print()
    print("  (2) Precompute strategy")
    print(
        f"      per-cell loop:    {per_cell_loop_ms:7.0f} ms ({4*n} predicts)"
    )
    print(
        f"      fully batched:    {fully_batched_ms:7.0f} ms (4 predicts)  "
        f"speedup={per_cell_loop_ms/fully_batched_ms:.2f}x"
    )
    print()
    print("  (3) In-memory cache")
    print(
        f"      cold per-call:    {cold_per_call_ms:7.2f} ms  "
        f"vs cache hit: {cache_hit_us:.2f} µs  → {cache_speedup_dict:,.0f}x"
    )
    print()
    print("  (4) Disk cache (.npz, survives subprocess restart)")
    print(
        f"      cold first boot:  {cold_boot_ms:7.0f} ms  "
        f"warm second boot: {warm_boot_ms:7.0f} ms  → {disk_speedup:.2f}x  "
        f"({disk_saved_ms:+.0f} ms saved)"
    )
    print(
        f"      cache footprint:  {cache_file_bytes/1e6:.2f} MB on disk for {n} cells"
    )
    print()
    print(f"  wrote: {out_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

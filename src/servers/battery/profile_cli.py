"""battery-profile — run cold-start diagnostic + probe + full _boot() timing.

Writes a markdown report combining all three measurements, plus the raw JSON
files. Runs regardless of BATTERY_* env-var state — it just profiles whatever
you have configured. To compare old vs new behavior, prefix the invocation
with the env vars you want to test (see Usage).

Usage:
  battery-profile                                 # default: label "baseline"
  battery-profile --label custom_label
  BATTERY_BOOT_BATCH_FS=1 battery-profile --label new_batched_fs

Env-var knobs that change behavior (for old-vs-new testing):
  BATTERY_BOOT_PARALLEL_FETCH=0  → disable parallel fetch (sequential, "old")
  BATTERY_BOOT_BATCH_FS=1        → batched feature selectors (new)
  BATTERY_KERAS_USE_CALL=1       → model(x, training=False) instead of .predict()
  BATTERY_LAZY_VOLTAGE=1         → skip voltage head at boot
  BATTERY_FS_BATCH_SIZE=N        → feature-selector inner batch size (default 128)
  BATTERY_HEAD_BATCH_SIZE=N      → RUL/voltage head batch size (default 256)
  BATTERY_TF_INTRA_OP_THREADS=N  → TF intra-op pool size

Outputs (under profiles/):
  <label>_<ts>.md                        — combined markdown report
  <label>_diagnostic_<ts>.json           — raw cold-start diagnostic
  <label>_probe_<ts>.json                — raw cold-start probe
  <label>_boot_<ts>.json                 — raw _boot() timing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# The optimization knobs are now baked-in defaults inside model_wrapper.py
# and main.py:
#   - precompute_cells_fully_batched (one TF call per model, all cells)
#   - lazy voltage at boot (computed on first voltage tool call)
#   - pre-compiled flexible-shape tf.function wrappers (no per-shape retracing)
#   - parallel CouchDB fetch (kept env-driven for tunability)
#
# To re-test ablations of failed/superseded experiments, edit the source:
#   - precompute_cell loop  → revert main.py:_boot's batched-call line
#   - SavedModel path       → instantiate SavedModelWrapper in _load_once
#   - TFLite quantized      → instantiate TFLiteWrapper in _load_once
# Each is ~3 lines of edit in model_wrapper.py.
_ENV_VARS_OF_INTEREST = [
    "BATTERY_BOOT_CELL_SUBSET",
    "BATTERY_BOOT_PARALLEL_FETCH",
    "BATTERY_BOOT_FETCH_WORKERS",
    "BATTERY_FS_BATCH_SIZE",
    "BATTERY_HEAD_BATCH_SIZE",
    "BATTERY_TF_INTRA_OP_THREADS",
    "BATTERY_TF_INTER_OP_THREADS",
    "BATTERY_GPU_FORCE_FLOAT32",
]


def _captured_env() -> dict[str, str]:
    return {k: os.environ.get(k, "(unset)") for k in _ENV_VARS_OF_INTEREST}


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


def _find_latest(pattern: str) -> Path | None:
    matches = sorted((_REPO_ROOT / "profiles").glob(pattern))
    return matches[-1] if matches else None


def _run_diagnostic(cells: str, repeats: int, label: str) -> Path | None:
    print("\n[1/3] Cold-start preprocessing diagnostic …")
    cmd = [
        "uv", "run", "--group", "battery",
        "python", "-m", "servers.battery.profile_diagnostic",
        "--cell-ids", cells,
        "--repeats", str(repeats),
        "--label", f"{label}_diagnostic",
    ]
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
    if rc != 0:
        print(f"  diagnostic failed (exit {rc})")
        return None
    return _find_latest(f"{label}_diagnostic_*.json")


def _run_probe(label: str) -> Path | None:
    print("\n[2/3] Cold-start probe (order swap, repeated fetch, boot decomposition) …")
    cmd = [
        "uv", "run", "--group", "battery",
        "python", "-m", "servers.battery.probe_cold_start",
        "--label", f"{label}_probe",
    ]
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
    if rc != 0:
        print(f"  probe failed (exit {rc})")
        return None
    return _find_latest(f"{label}_probe_*.json")


def _run_boot_timing(label: str, cells: str) -> Path | None:
    """Time the full _boot() in a fresh subprocess so TF + weights load from cold."""
    print("\n[3/3] Full _boot() timing (fresh subprocess, env-var sensitive) …")
    snippet = (
        "import json, os, time\n"
        f"os.environ['BATTERY_BOOT_CELL_SUBSET']='{cells}'\n"
        "t0=time.perf_counter()\n"
        "import servers.battery.main as m\n"
        "from servers.battery import model_wrapper as mw\n"
        "elapsed=time.perf_counter()-t0\n"
        "out={'boot_wall_ms': elapsed*1000.0, 'cache_size': len(m._CACHE),\n"
        "     'cells_loaded': sorted(m._CACHE.keys()),\n"
        "     'usable_cells_targeted': list(m._USABLE_MODEL_CELLS),\n"
        "     'stage_timings_ms': dict(mw._LAST_BATCH_TIMINGS)}\n"
        "print('BOOT_JSON_BEGIN'+json.dumps(out)+'BOOT_JSON_END')\n"
    )
    cmd = [sys.executable, "-c", snippet]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  boot timing failed (exit {proc.returncode})")
        print(proc.stderr[-500:] if proc.stderr else "")
        return None
    out_text = proc.stdout
    try:
        start = out_text.index("BOOT_JSON_BEGIN") + len("BOOT_JSON_BEGIN")
        end = out_text.index("BOOT_JSON_END")
        boot_data = json.loads(out_text[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  could not parse boot output: {e}")
        return None
    boot_data["env_state"] = _captured_env()
    boot_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    boot_data["git_sha"] = _git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = _REPO_ROOT / "profiles" / f"{label}_boot_{ts}.json"
    out_path.write_text(json.dumps(boot_data, indent=2), encoding="utf-8")
    print(
        f"  _boot() wall: {boot_data['boot_wall_ms']:.0f} ms  "
        f"cache: {boot_data['cache_size']} cells"
    )
    return out_path


def _md_diagnostic(data: dict[str, Any]) -> str:
    lines = ["## 1. Cold-start preprocessing diagnostic\n"]
    cfg = data.get("config", {})
    lines.append(
        f"**Cells:** {', '.join(cfg.get('cell_ids', []))} × "
        f"{cfg.get('repeats', '?')} repeats  "
        f"**Source:** {cfg.get('data_source', '?')}\n"
    )
    lines.append("### Phase breakdown\n")
    lines.append("| Phase | Mean (ms) | Min (ms) | Max (ms) | % of total |")
    lines.append("|---|---|---|---|---|")
    for phase in (
        "fetch_charge",
        "fetch_discharge",
        "sort_filter",
        "preprocess_cycle_loop",
        "stack_arrays",
    ):
        m = data["measurements"].get(phase, {})
        wm = m.get("wall_ms", {})
        pct = m.get("pct_of_total", "?")
        lines.append(
            f"| {phase} | {wm.get('mean', 0):.2f} | {wm.get('min', 0):.2f} | "
            f"{wm.get('max', 0):.2f} | {pct}% |"
        )
    tot = data["measurements"].get("total_preprocess_cell", {}).get("wall_ms", {})
    lines.append(
        f"| **TOTAL** | **{tot.get('mean', 0):.2f}** | {tot.get('min', 0):.2f} | "
        f"{tot.get('max', 0):.2f} | n={tot.get('n', 0)} |"
    )
    lines.append(f"\n**Memory peak (1 cell):** {data.get('memory_peak_mb_one_cell', '?')} MB\n")
    lines.append("### Per-cell metadata\n")
    lines.append("| Cell | Charge docs | Discharge docs | Clean paired |")
    lines.append("|---|---|---|---|")
    for cell, meta in data.get("cell_meta", {}).items():
        lines.append(
            f"| {cell} | {meta.get('n_charge_docs', '?')} | "
            f"{meta.get('n_discharge_docs', '?')} | {meta.get('n_clean_pairs', '?')} |"
        )
    return "\n".join(lines)


def _md_probe(data: dict[str, Any]) -> str:
    lines = ["## 2. Cold-start probe\n"]
    e = data.get("experiments", {})

    a = e.get("A_order_swap", {})
    lines.append("### Experiment A — Order swap (charge vs discharge)\n")
    lines.append("| Cell | First | First (ms) | Second | Second (ms) | First/Second ratio |")
    lines.append("|---|---|---|---|---|---|")
    for r in a.get("results", []):
        lines.append(
            f"| {r['cell']} | {r['first_type']} | {r['first_ms']} | "
            f"{r['second_type']} | {r['second_ms']} | "
            f"{r.get('ratio_first_over_second', '?')} |"
        )

    b = e.get("B_repeated_fetch", {})
    lines.append("\n### Experiment B — Repeated fetch (B0005 discharge)\n")
    lines.append("| Call | Wall (ms) |")
    lines.append("|---|---|")
    for i, t in enumerate(b.get("wall_ms_per_call", []), 1):
        lines.append(f"| {i} | {t} |")
    lines.append(f"\n**Warm/cold speedup:** {b.get('warm_over_cold_speedup', '?')}×\n")

    c = e.get("C_boot_decomposition", {})
    lines.append("### Experiment C — Boot decomposition (single cell)\n")
    if c.get("model_available"):
        lines.append("| Phase | Time (ms) |")
        lines.append("|---|---|")
        lines.append(f"| TF + weight load (`_load_once`) | {c['load_once_ms']} |")
        lines.append(f"| Data-side (`preprocess_cell_from_couchdb`) | {c['preprocess_cell_ms']} |")
        lines.append(f"| Model-side (`precompute_cell`) | {c['precompute_cell_ms']} |")
        lines.append(f"| **SUM** | **{c['sum_ms']}** |")
    else:
        lines.append(f"_Model unavailable. TF load only: {c.get('load_once_ms', '?')} ms_")
    return "\n".join(lines)


def _md_boot(data: dict[str, Any]) -> str:
    lines = ["## 3. Full `_boot()` timing\n"]
    lines.append(
        f"Fresh subprocess; env vars at time of run are inherited by `_boot()`. "
        f"Cell subset: `{','.join(data.get('cells_loaded', []))}`.\n"
    )
    lines.append(f"**Boot wall:** {data.get('boot_wall_ms', 0):.0f} ms  ")
    lines.append(f"**Cells preloaded:** {data.get('cache_size', '?')}  ")
    lines.append(f"**Cells targeted:** {len(data.get('usable_cells_targeted', []))}\n")
    stages = data.get("stage_timings_ms") or {}
    if stages:
        lines.append("### Stage breakdown (full-batch path only)\n")
        lines.append("| Stage | ms |")
        lines.append("|---|---|")
        for k in (
            "feature_selectors_ms",
            "sliding_windows_ms",
            "rul_predict_ms",
            "volt_predict_ms",
            "total_ms",
        ):
            if k in stages:
                lines.append(f"| {k} | {stages[k]:.1f} |")
        if "n_cells" in stages and "n_cycles_per_cell" in stages:
            lines.append(
                f"\n_(n_cells={int(stages['n_cells'])}, "
                f"n_cycles_per_cell={int(stages['n_cycles_per_cell'])})_\n"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--cells", default="B0005,B0006,B0018")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-boot", action="store_true", help="Skip step 3.")
    ap.add_argument(
        "--only-boot",
        action="store_true",
        help="Run only step 3 (boot timing). Useful for env-var A/B comparisons.",
    )
    args = ap.parse_args()

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    timestamp_compact = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    env_state = _captured_env()
    git_sha = _git_sha()

    print(f"battery-profile  label={args.label}  cells={args.cells}")
    print(f"git_sha={git_sha[:8]}  timestamp={timestamp_iso}")
    print("Env vars at run time:")
    for k, v in env_state.items():
        print(f"  {k} = {v}")

    t_start = time.perf_counter()
    diag_path = None if args.only_boot else _run_diagnostic(args.cells, args.repeats, args.label)
    probe_path = None if args.only_boot else _run_probe(args.label)
    boot_path = None if args.skip_boot else _run_boot_timing(args.label, args.cells)
    elapsed_min = (time.perf_counter() - t_start) / 60

    lines: list[str] = []
    lines.append(f"# Battery profile report — `{args.label}`\n")
    lines.append(f"**Timestamp:** {timestamp_iso}  ")
    lines.append(f"**Git SHA:** `{git_sha}`  ")
    lines.append(f"**Wall time (full report):** {elapsed_min:.1f} min\n")
    lines.append("## Environment overrides at run time\n")
    lines.append("| Variable | Value |")
    lines.append("|---|---|")
    for k, v in env_state.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

    if diag_path:
        try:
            lines.append(_md_diagnostic(json.loads(diag_path.read_text(encoding="utf-8"))))
            lines.append(f"\n_Raw JSON: `{diag_path.relative_to(_REPO_ROOT)}`_\n")
        except Exception as e:
            lines.append(f"## 1. Cold-start preprocessing diagnostic\n_Failed to parse: {e}_\n")
    else:
        lines.append("## 1. Cold-start preprocessing diagnostic\n_Not run or failed._\n")

    if probe_path:
        try:
            lines.append(_md_probe(json.loads(probe_path.read_text(encoding="utf-8"))))
            lines.append(f"\n_Raw JSON: `{probe_path.relative_to(_REPO_ROOT)}`_\n")
        except Exception as e:
            lines.append(f"## 2. Cold-start probe\n_Failed to parse: {e}_\n")
    else:
        lines.append("## 2. Cold-start probe\n_Not run or failed._\n")

    if boot_path:
        try:
            lines.append(_md_boot(json.loads(boot_path.read_text(encoding="utf-8"))))
            lines.append(f"\n_Raw JSON: `{boot_path.relative_to(_REPO_ROOT)}`_\n")
        except Exception as e:
            lines.append(f"## 3. Full `_boot()` timing\n_Failed to parse: {e}_\n")
    elif not args.skip_boot:
        lines.append("## 3. Full `_boot()` timing\n_Not run or failed._\n")

    out_path = _REPO_ROOT / "profiles" / f"{args.label}_{timestamp_compact}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport: {out_path.relative_to(_REPO_ROOT)}")
    if diag_path:
        print(f"  diagnostic JSON: {diag_path.relative_to(_REPO_ROOT)}")
    if probe_path:
        print(f"  probe JSON:      {probe_path.relative_to(_REPO_ROOT)}")
    if boot_path:
        print(f"  boot JSON:       {boot_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

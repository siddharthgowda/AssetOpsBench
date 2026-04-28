"""Battery MCP Server — comprehensive CPU performance profiler.

Sections
--------
1. preprocessing     inp_500 interpolation, preprocess_cycle, full cell pipeline
2. sliding_windows   The np.stack loop that builds (100, 50, 12) windows in precompute_cell
3. model_inference   feature_selector, concat_data, RUL head, voltage head, full precompute_cell
4. statistical_tools np.polyfit (impedance), z-score (outliers), row construction (summary)
5. memory            tracemalloc allocation deltas, tensor byte sizes, process RSS

No CouchDB or model weights are required — all sections run on synthetic data generated
by mock_data.py. Model inference is skipped with a clear message if weights are missing.

Output
------
JSON saved to <output_dir>/<label>_<timestamp>.json.
Run compare.py to diff two result files.

Why not PyTorch / NVIDIA profilers?
------------------------------------
• The acctouhou model uses TensorFlow / Keras 2, not PyTorch — PyTorch profiler does not apply.
• model_wrapper._load_once() detects no GPU and forces float32 CPU. NVIDIA Nsight / nvprof
  require CUDA hardware and are therefore irrelevant for this deployment.
• TF Profiler (tf.profiler.experimental) is included as an optional section because it traces
  individual TF ops on CPU and produces TensorBoard-compatible logs.
"""
from __future__ import annotations

import cProfile
import io
import json
import logging
import os
import platform
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np

logger = logging.getLogger("battery-profiler")

# ── optional heavy deps ────────────────────────────────────────────────────────

try:
    import psutil as _psutil  # type: ignore[import-untyped]
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    logger.warning("psutil not installed — install with: uv sync --group profiling")

try:
    import line_profiler as _lp  # type: ignore[import-untyped]
    _LINE_PROFILER = True
except ImportError:
    _LINE_PROFILER = False

# ── src on path ────────────────────────────────────────────────────────────────

_SRC = Path(__file__).resolve().parent.parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from servers.battery.preprocessing import inp_500, preprocess_cycle, preprocess_cell_from_couchdb
from servers.battery.profiling.mock_data import make_cell_cycles, make_fleet, MockCouchDBClient


# ── metric containers ──────────────────────────────────────────────────────────

@dataclass
class TimingStats:
    """Accumulates wall-clock timing across repeated calls."""
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def throughput_per_sec(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 4),
            "min_ms": round(self.min_ms, 4) if self.min_ms != float("inf") else None,
            "max_ms": round(self.max_ms, 3),
            "throughput_per_sec": round(self.throughput_per_sec, 1),
        }


# ── helpers ────────────────────────────────────────────────────────────────────

@contextmanager
def _tracemem() -> Generator[dict, None, None]:
    """Measure net memory allocation in MB over a block."""
    out: dict = {}
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    try:
        yield out
    finally:
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diffs = snap_after.compare_to(snap_before, "lineno")
        net_bytes = sum(s.size_diff for s in diffs if s.size_diff > 0)
        out["net_alloc_mb"] = round(net_bytes / 1e6, 3)


def _timed(fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def _cprofile_top(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_top: int = 25,
) -> tuple[Any, str]:
    """Run fn under cProfile; return (result, formatted top-N stats string)."""
    kwargs = kwargs or {}
    pr = cProfile.Profile()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(n_top)
    return result, buf.getvalue()


def _line_profile_text(fn: Callable, *args: Any, **kwargs: Any) -> str:
    """Run fn under line_profiler if available; return formatted text."""
    if not _LINE_PROFILER:
        return ""
    lp = _lp.LineProfiler(fn)
    lp.runcall(fn, *args, **kwargs)
    buf = io.StringIO()
    lp.print_stats(stream=buf)
    return buf.getvalue()


def _process_rss_mb() -> float:
    if _PSUTIL:
        return round(_psutil.Process().memory_info().rss / 1e6, 1)
    return 0.0


# ── main profiler ──────────────────────────────────────────────────────────────

class BatteryProfiler:
    """Run all profiling sections and produce a structured JSON result file.

    Parameters
    ----------
    label:
        Short identifier for this run, e.g. ``"baseline"`` or ``"optimized"``.
    output_dir:
        Directory where ``<label>_<timestamp>.json`` will be written.
    n_cells:
        Number of synthetic cells to include (higher = more realistic boot timing).
    n_cycles:
        Cycles per cell. 100 is the minimum for the model; use fewer only for
        quick smoke-tests of the preprocessing section.
    use_real_model:
        If True and the model weights are found, load and profile TF inference.
        If False the model_inference section is still attempted but skipped if
        weights are missing.
    verbose:
        Print a human-readable summary to stdout after saving.
    """

    def __init__(
        self,
        label: str = "baseline",
        output_dir: str = "profiles",
        n_cells: int = 3,
        n_cycles: int = 100,
        use_real_model: bool = True,
        verbose: bool = True,
    ) -> None:
        self.label = label
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.n_cells = n_cells
        self.n_cycles = n_cycles
        self.use_real_model = use_real_model

        logger.info("Generating synthetic fleet (%d cells × %d cycles)…", n_cells, n_cycles)
        self._fleet = make_fleet(n_cells=n_cells, n_cycles=n_cycles)
        self._mock_client = MockCouchDBClient(fleet=self._fleet, n_cycles=n_cycles)

        self._results: dict = {
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_cells": n_cells,
                "n_cycles": n_cycles,
                "use_real_model": use_real_model,
            },
            "system": self._system_info(),
            "sections": {},
            "memory": {},
            "cprofile": {},
            "line_profiles": {},
        }

    # ── system info ────────────────────────────────────────────────────────────

    def _system_info(self) -> dict:
        info: dict = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(),
        }
        if _PSUTIL:
            vm = _psutil.virtual_memory()
            info["total_ram_gb"] = round(vm.total / 1e9, 1)
            freq = _psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = round(freq.current, 0)
            info["process_rss_mb_at_start"] = _process_rss_mb()
        return info

    # ── section 1: preprocessing ───────────────────────────────────────────────

    def profile_preprocessing(self) -> dict:
        """Profile inp_500, preprocess_cycle, and the full cell pipeline."""
        logger.info("[1/5] Profiling preprocessing…")
        section: dict = {}

        first_id = list(self._fleet.keys())[0]
        dis_docs = self._fleet[first_id]["discharge"]
        ch_docs = self._fleet[first_id]["charge"]
        cell_ids = list(self._fleet.keys())

        # ── inp_500 ────────────────────────────────────────────────────────────
        logger.info("  inp_500 — 500 timed calls (10 warm-up)…")
        sample = dis_docs[0]["data"]
        t_arr = np.asarray(sample["Time"], dtype=float)
        v_arr = np.asarray(sample["Voltage_measured"], dtype=float)
        for _ in range(10):
            inp_500(v_arr, t_arr)  # warm up JIT / import caches
        inp500_stats = TimingStats()
        for _ in range(500):
            _, ms = _timed(inp_500, v_arr, t_arr)
            inp500_stats.record(ms)
        section["inp_500"] = inp500_stats.to_dict()

        # Line profile of inp_500 (shows cost of interp1d construction vs eval)
        if _LINE_PROFILER:
            lp_text = _line_profile_text(inp_500, v_arr, t_arr)
            if lp_text:
                self._results["line_profiles"]["inp_500"] = lp_text

        # ── preprocess_cycle (discharge) ───────────────────────────────────────
        logger.info("  preprocess_cycle — %d discharge cycles…", len(dis_docs))
        pc_dis_stats = TimingStats()
        mem_pc: dict = {}
        with _tracemem() as mem_pc:
            for doc in dis_docs:
                _, ms = _timed(preprocess_cycle, doc["data"])
                pc_dis_stats.record(ms)
        section["preprocess_cycle_discharge"] = {
            **pc_dis_stats.to_dict(),
            "memory_net_mb": mem_pc.get("net_alloc_mb", 0.0),
            "output_shape": [4, 500],
        }

        # ── preprocess_cycle (charge) ──────────────────────────────────────────
        pc_ch_stats = TimingStats()
        for doc in ch_docs:
            _, ms = _timed(preprocess_cycle, doc["data"])
            pc_ch_stats.record(ms)
        section["preprocess_cycle_charge"] = pc_ch_stats.to_dict()

        # Line profile of preprocess_cycle (shows inp_500 vs Coulomb counting split)
        if _LINE_PROFILER:
            lp_text = _line_profile_text(preprocess_cycle, dis_docs[0]["data"])
            if lp_text:
                self._results["line_profiles"]["preprocess_cycle"] = lp_text

        # ── preprocess_cell_from_couchdb (full pipeline) ───────────────────────
        logger.info(
            "  preprocess_cell_from_couchdb — %d cells × %d cycles…",
            len(cell_ids),
            self.n_cycles,
        )
        cell_stats = TimingStats()
        cell_mem: dict = {}
        with _tracemem() as cell_mem:
            for cid in cell_ids:
                try:
                    _, ms = _timed(preprocess_cell_from_couchdb, cid, self._mock_client)
                    cell_stats.record(ms)
                except ValueError as exc:
                    logger.warning("  Skipped %s: %s", cid, exc)
        section["preprocess_cell_from_couchdb"] = {
            **cell_stats.to_dict(),
            "memory_net_mb": cell_mem.get("net_alloc_mb", 0.0),
            "cycles_per_sec": round(
                self.n_cycles * cell_stats.calls / max(cell_stats.total_ms / 1000.0, 1e-9),
                1,
            ),
        }

        # cProfile call tree for one full cell preprocessing run
        logger.info("  cProfile call tree for preprocess_cell_from_couchdb…")
        try:
            _, stats_str = _cprofile_top(
                preprocess_cell_from_couchdb,
                args=(cell_ids[0], self._mock_client),
                n_top=25,
            )
            self._results["cprofile"]["preprocess_cell_from_couchdb"] = stats_str
        except ValueError:
            pass

        self._results["sections"]["preprocessing"] = section
        return section

    # ── section 2: sliding window construction ─────────────────────────────────

    def profile_sliding_windows(self, cell_feat: Optional[np.ndarray] = None) -> dict:
        """Profile the sliding-window np.stack loop from precompute_cell.

        This is the bottleneck that creates the (100, 50, 12) windows tensor —
        a target for np.lib.stride_tricks optimization.
        """
        logger.info("[2/5] Profiling sliding window construction…")
        if cell_feat is None:
            cell_feat = np.random.randn(100, 12).astype(np.float32)

        from servers.battery.model_wrapper import _pad_edge

        def _build_windows_current(feat: np.ndarray) -> np.ndarray:
            """Exact replica of the loop in precompute_cell."""
            return np.stack(
                [_pad_edge(feat[max(0, k - 49): k + 1], 50) for k in range(len(feat))]
            )

        # warm up
        for _ in range(5):
            _build_windows_current(cell_feat)

        window_stats = TimingStats()
        win_mem: dict = {}
        with _tracemem() as win_mem:
            for _ in range(100):
                _, ms = _timed(_build_windows_current, cell_feat)
                window_stats.record(ms)

        result = {
            "sliding_window_stack_100x50x12": {
                **window_stats.to_dict(),
                "memory_net_mb": win_mem.get("net_alloc_mb", 0.0),
                "output_shape": [100, 50, 12],
                "output_bytes": 100 * 50 * 12 * 4,
            }
        }

        if _LINE_PROFILER:
            lp_text = _line_profile_text(_build_windows_current, cell_feat)
            if lp_text:
                self._results["line_profiles"]["sliding_window_build"] = lp_text

        self._results["sections"]["sliding_windows"] = result
        return result

    # ── section 3: model inference ─────────────────────────────────────────────

    def profile_inference(self) -> dict:
        """Profile TF feature extractors, RUL head, voltage head, and full precompute_cell.

        Skipped gracefully if model weights are unavailable.
        """
        from servers.battery import model_wrapper

        if self.use_real_model:
            model_wrapper._load_once()

        if not model_wrapper._MODEL_AVAILABLE:
            logger.info(
                "[3/5] Skipping model inference — weights not found. "
                "Run scripts/setup_battery_artifacts.sh to set up weights."
            )
            self._results["sections"]["model_inference"] = {
                "skipped": True,
                "reason": "model weights unavailable (BATTERY_MODEL_WEIGHTS_DIR)",
            }
            return {}

        logger.info("[3/5] Profiling model inference…")
        section: dict = {}
        cell_ids = list(self._fleet.keys())

        from servers.battery.model_wrapper import (
            _CACHE,
            _MODELS,
            _NORMS,
            _pad_edge,
            concat_data,
            feature_selector,
            precompute_cell,
        )

        # Pre-build one cell's preprocessed tensors for inference profiling
        ch, dis, summ = preprocess_cell_from_couchdb(cell_ids[0], self._mock_client)

        # ── feature_selector — charge ──────────────────────────────────────────
        logger.info("  feature_selector (charge, %d cycles, 5 runs)…", self.n_cycles)
        feature_selector(_MODELS["fs_ch"], ch, _NORMS["charge"])  # JIT warm-up
        fs_ch_stats = TimingStats()
        for _ in range(5):
            _, ms = _timed(feature_selector, _MODELS["fs_ch"], ch, _NORMS["charge"])
            fs_ch_stats.record(ms)
        section["feature_selector_charge"] = {
            **fs_ch_stats.to_dict(),
            "batch_size": 128,
            "input_shape": list(ch.shape),
        }

        # ── feature_selector — discharge ───────────────────────────────────────
        feature_selector(_MODELS["fs_dis"], dis, _NORMS["discharge"])
        fs_dis_stats = TimingStats()
        for _ in range(5):
            _, ms = _timed(feature_selector, _MODELS["fs_dis"], dis, _NORMS["discharge"])
            fs_dis_stats.record(ms)
        section["feature_selector_discharge"] = {
            **fs_dis_stats.to_dict(),
            "batch_size": 128,
            "input_shape": list(dis.shape),
        }

        # ── concat_data ────────────────────────────────────────────────────────
        ch_feat = feature_selector(_MODELS["fs_ch"], ch, _NORMS["charge"])
        dis_feat = feature_selector(_MODELS["fs_dis"], dis, _NORMS["discharge"])
        concat_stats = TimingStats()
        for _ in range(200):
            _, ms = _timed(concat_data, ch_feat, dis_feat, summ, _NORMS["summary"])
            concat_stats.record(ms)
        section["concat_data_normalization"] = concat_stats.to_dict()

        # ── sliding window construction ────────────────────────────────────────
        cell_feat = concat_data(ch_feat, dis_feat, summ, _NORMS["summary"])
        sw_stats = TimingStats()
        for _ in range(20):
            _, ms = _timed(
                lambda f: np.stack(
                    [_pad_edge(f[max(0, k - 49): k + 1], 50) for k in range(len(f))]
                ),
                cell_feat,
            )
            sw_stats.record(ms)
        section["sliding_window_construction"] = {
            **sw_stats.to_dict(),
            "output_shape": [100, 50, 12],
        }

        windows = np.stack(
            [_pad_edge(cell_feat[max(0, k - 49): k + 1], 50) for k in range(len(cell_feat))]
        )

        # ── RUL head ───────────────────────────────────────────────────────────
        logger.info("  rul.predict (100 windows, batch_size=256, 5 runs)…")
        _MODELS["rul"].predict(windows, batch_size=256, verbose=0)  # warm up
        rul_stats = TimingStats()
        for _ in range(5):
            _, ms = _timed(_MODELS["rul"].predict, windows, batch_size=256, verbose=0)
            rul_stats.record(ms)
        section["rul_predict"] = {
            **rul_stats.to_dict(),
            "batch_size": 256,
            "n_cycles": len(windows),
            "ms_per_cycle": round(rul_stats.mean_ms / max(len(windows), 1), 4),
        }

        # ── voltage head ──────────────────────────────────────────────────────
        logger.info("  volt.predict (100 windows, batch_size=256, 5 runs)…")
        second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)
        _MODELS["volt"].predict([windows, second_input], batch_size=256, verbose=0)
        volt_stats = TimingStats()
        for _ in range(5):
            _, ms = _timed(
                _MODELS["volt"].predict, [windows, second_input], batch_size=256, verbose=0
            )
            volt_stats.record(ms)
        section["volt_predict"] = {
            **volt_stats.to_dict(),
            "batch_size": 256,
            "n_cycles": len(windows),
            "ms_per_cycle": round(volt_stats.mean_ms / max(len(windows), 1), 4),
        }

        # ── precompute_cell end-to-end ─────────────────────────────────────────
        logger.info("  precompute_cell end-to-end (3 runs)…")
        cell_e2e_stats = TimingStats()
        cell_mem: dict = {}
        with _tracemem() as cell_mem:
            for _ in range(3):
                _CACHE.pop(cell_ids[0], None)
                _, ms = _timed(precompute_cell, cell_ids[0], ch, dis, summ)
                cell_e2e_stats.record(ms)
        section["precompute_cell_full"] = {
            **cell_e2e_stats.to_dict(),
            "n_cycles": self.n_cycles,
            "ms_per_cycle": round(cell_e2e_stats.mean_ms / max(self.n_cycles, 1), 4),
            "memory_net_mb": cell_mem.get("net_alloc_mb", 0.0),
        }

        # ── full boot simulation (all cells) ──────────────────────────────────
        logger.info("  Simulating _boot() for %d cells…", len(cell_ids))
        CACHE_ref = _CACHE
        CACHE_ref.clear()
        boot_ms_per_cell: dict = {}
        t_boot = time.perf_counter()
        for cid in cell_ids:
            try:
                CACHE_ref.pop(cid, None)
                ch2, dis2, summ2 = preprocess_cell_from_couchdb(cid, self._mock_client)
                t0 = time.perf_counter()
                precompute_cell(cid, ch2, dis2, summ2)
                boot_ms_per_cell[cid] = round((time.perf_counter() - t0) * 1000.0, 1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  Boot skipped %s: %s", cid, exc)
        total_boot_ms = (time.perf_counter() - t_boot) * 1000.0
        section["boot_all_cells"] = {
            "n_cells": len(boot_ms_per_cell),
            "total_ms": round(total_boot_ms, 1),
            "mean_cell_ms": round(
                sum(boot_ms_per_cell.values()) / max(len(boot_ms_per_cell), 1), 1
            ),
            "per_cell_ms": boot_ms_per_cell,
        }

        # cProfile for precompute_cell
        CACHE_ref.pop(cell_ids[0], None)
        _, cprofile_str = _cprofile_top(precompute_cell, args=(cell_ids[0], ch, dis, summ))
        self._results["cprofile"]["precompute_cell"] = cprofile_str

        self._results["sections"]["model_inference"] = section
        return section

    # ── section 4: statistical tools ──────────────────────────────────────────

    def profile_statistical_tools(self) -> dict:
        """Profile the pure numpy/scipy computations inside the statistical tools.

        These are profiled directly (not through the MCP tool wrappers) so that
        DB I/O is isolated from algorithm cost.
        """
        logger.info("[4/5] Profiling statistical tools…")
        section: dict = {}

        first_id = list(self._fleet.keys())[0]
        dis_docs = self._fleet[first_id]["discharge"]
        imp_docs = self._fleet[first_id]["impedance"]

        # ── np.polyfit — impedance growth ──────────────────────────────────────
        logger.info("  np.polyfit for impedance growth (1 000 calls)…")
        rcts = [(d["cycle_index"], d["data"]["Rct"]) for d in imp_docs]
        cycles_arr = np.array([r[0] for r in rcts], dtype=float)
        rct_arr = np.array([r[1] for r in rcts], dtype=float)
        log_rct = np.log(np.where(np.abs(rct_arr) == 0, 1e-12, np.abs(rct_arr)))

        polyfit_stats = TimingStats()
        for _ in range(1_000):
            _, ms = _timed(np.polyfit, cycles_arr, log_rct, 1)
            polyfit_stats.record(ms)
        section["np_polyfit_impedance"] = polyfit_stats.to_dict()

        # ── capacity fade rates — detect_capacity_outliers inner loop ──────────
        logger.info("  capacity fade rate loop (%d cells, 100 calls)…", self.n_cells)

        def _compute_fade_rates(fleet: dict, window: int = 50) -> dict[str, float]:
            rates: dict[str, float] = {}
            for cid, cell_data in fleet.items():
                caps = sorted(
                    [
                        (d["cycle_index"], d["data"].get("Capacity", 0.0))
                        for d in cell_data["discharge"]
                    ],
                    key=lambda x: x[0],
                )
                if len(caps) < 2:
                    continue
                head = caps[: min(window, len(caps))]
                rates[cid] = (float(head[0][1]) - float(head[-1][1])) / max(len(head), 1)
            return rates

        fade_stats = TimingStats()
        for _ in range(100):
            _, ms = _timed(_compute_fade_rates, self._fleet)
            fade_stats.record(ms)
        section["capacity_fade_rate_loop"] = {
            **fade_stats.to_dict(),
            "n_cells": self.n_cells,
            "ms_per_cell": round(fade_stats.mean_ms / max(self.n_cells, 1), 4),
        }

        # ── z-score computation ────────────────────────────────────────────────
        fade_rates = _compute_fade_rates(self._fleet)
        rate_vals = np.array(list(fade_rates.values()))

        zscore_stats = TimingStats()
        for _ in range(10_000):
            mean_r = float(rate_vals.mean())
            std_r = float(rate_vals.std()) or 1.0
            _, ms = _timed(
                lambda: {c: (r - mean_r) / std_r for c, r in fade_rates.items()}
            )
            zscore_stats.record(ms)
        section["zscore_dict_comprehension"] = zscore_stats.to_dict()

        # ── summary row construction — get_battery_cycle_summary inner loop ────
        logger.info("  cycle summary row construction (%d cycles, 20 calls)…", len(dis_docs))
        imp_by_cycle = {d.get("cycle_index"): d for d in imp_docs}

        def _build_summary_rows(dis: list, imp_idx: dict) -> list:
            rows = []
            for d in dis:
                i = d.get("cycle_index")
                data = d.get("data", {})
                temps = data.get("Temperature_measured", [])
                volts = data.get("Voltage_measured", [])
                imp_data = imp_idx.get(i, {}).get("data", {})
                rows.append(
                    {
                        "cycle_index": i,
                        "capacity_ah": data.get("Capacity"),
                        "max_temp_c": max(temps) if temps else None,
                        "min_temp_c": min(temps) if temps else None,
                        "avg_voltage": (sum(volts) / len(volts)) if volts else None,
                        "rct": imp_data.get("Rct"),
                        "re": imp_data.get("Re"),
                    }
                )
            return rows

        summary_stats = TimingStats()
        for _ in range(20):
            _, ms = _timed(_build_summary_rows, dis_docs, imp_by_cycle)
            summary_stats.record(ms)
        section["summary_row_construction"] = {
            **summary_stats.to_dict(),
            "n_cycles": len(dis_docs),
            "ms_per_row": round(summary_stats.mean_ms / max(len(dis_docs), 1), 5),
        }

        if _LINE_PROFILER:
            lp_text = _line_profile_text(_build_summary_rows, dis_docs, imp_by_cycle)
            if lp_text:
                self._results["line_profiles"]["summary_row_construction"] = lp_text

        # cProfile for the full outlier computation
        _, stats_str = _cprofile_top(
            _compute_fade_rates,
            args=(self._fleet,),
            n_top=15,
        )
        self._results["cprofile"]["capacity_fade_rates"] = stats_str

        self._results["sections"]["statistical_tools"] = section
        return section

    # ── section 5: memory footprint ───────────────────────────────────────────

    def profile_memory_footprint(self) -> dict:
        """Measure tensor byte sizes, per-section allocation deltas, and process RSS."""
        logger.info("[5/5] Profiling memory footprint…")
        section: dict = {}

        # ── theoretical array sizes ────────────────────────────────────────────
        section["tensor_sizes_bytes"] = {
            "single_cycle_4x500_float32": 4 * 500 * 4,
            "100_cycles_charge_discharge_100x4x500_float32": 100 * 4 * 500 * 4,
            "cell_feat_100x12_float32": 100 * 12 * 4,
            "sliding_windows_100x50x12_float32": 100 * 50 * 12 * 4,
            "rul_output_100x2_float32": 100 * 2 * 4,
            "volt_curves_100x100_float32": 100 * 100 * 4,
        }

        # ── live allocation delta for preprocessing one cell ───────────────────
        first_id = list(self._fleet.keys())[0]
        mem_preprocess: dict = {}
        with _tracemem() as mem_preprocess:
            preprocess_cell_from_couchdb(first_id, self._mock_client)
        section["preprocessing_one_cell_net_alloc_mb"] = mem_preprocess.get("net_alloc_mb", 0.0)

        # ── process-level memory ───────────────────────────────────────────────
        if _PSUTIL:
            proc = _psutil.Process()
            mi = proc.memory_info()
            section["process_rss_mb"] = round(mi.rss / 1e6, 1)
            section["process_vms_mb"] = round(mi.vms / 1e6, 1)
            section["process_rss_mb_at_end"] = _process_rss_mb()

        self._results["memory"] = section
        return section

    # ── optional: TF profiler trace ────────────────────────────────────────────

    def profile_tf_ops(self, trace_dir: Optional[str] = None) -> dict:
        """Record a TF op-level trace using tf.profiler.experimental.

        Saves a TensorBoard-compatible log to ``<trace_dir>/tf_profile/``.
        Only runs if the model is available.
        """
        from servers.battery import model_wrapper

        if not model_wrapper._MODEL_AVAILABLE:
            logger.info("TF profiler skipped — model unavailable.")
            return {"skipped": True}

        try:
            import tensorflow as tf
        except ImportError:
            return {"skipped": True, "reason": "tensorflow not installed"}

        if trace_dir is None:
            trace_dir = str(self.output_dir / "tf_profile")
        Path(trace_dir).mkdir(parents=True, exist_ok=True)

        from servers.battery.model_wrapper import _MODELS, _NORMS, _pad_edge, feature_selector, concat_data

        ch, dis, summ = preprocess_cell_from_couchdb(
            list(self._fleet.keys())[0], self._mock_client
        )
        ch_feat = feature_selector(_MODELS["fs_ch"], ch, _NORMS["charge"])
        dis_feat = feature_selector(_MODELS["fs_dis"], dis, _NORMS["discharge"])
        cell_feat = concat_data(ch_feat, dis_feat, summ, _NORMS["summary"])
        windows = np.stack(
            [_pad_edge(cell_feat[max(0, k - 49): k + 1], 50) for k in range(len(cell_feat))]
        )
        second_input = np.full((len(windows), 1), 0.5, dtype=np.float32)

        logger.info("Starting TF op trace → %s …", trace_dir)
        tf.profiler.experimental.start(trace_dir)
        with tf.profiler.experimental.Trace("rul_predict", step_num=1):
            _MODELS["rul"].predict(windows, batch_size=256, verbose=0)
        with tf.profiler.experimental.Trace("volt_predict", step_num=2):
            _MODELS["volt"].predict([windows, second_input], batch_size=256, verbose=0)
        tf.profiler.experimental.stop()
        logger.info("TF profile trace saved → %s (view with tensorboard --logdir %s)", trace_dir, trace_dir)

        result = {"trace_dir": trace_dir, "view_cmd": f"tensorboard --logdir {trace_dir}"}
        self._results["sections"]["tf_profile"] = result
        return result

    # ── throughput summary ─────────────────────────────────────────────────────

    def _compute_throughput_summary(self) -> dict:
        secs = self._results["sections"]
        pre = secs.get("preprocessing", {})
        inf = secs.get("model_inference", {})
        summary: dict = {}

        pc = pre.get("preprocess_cycle_discharge", {})
        if pc.get("throughput_per_sec"):
            summary["preprocess_cycles_per_sec"] = pc["throughput_per_sec"]

        cell_pp = pre.get("preprocess_cell_from_couchdb", {})
        if cell_pp.get("cycles_per_sec"):
            summary["cell_preprocessing_throughput_cycles_per_sec"] = cell_pp["cycles_per_sec"]

        inp = pre.get("inp_500", {})
        if inp.get("throughput_per_sec"):
            summary["inp_500_calls_per_sec"] = inp["throughput_per_sec"]

        if not inf.get("skipped"):
            pcc = inf.get("precompute_cell_full", {})
            if pcc.get("ms_per_cycle"):
                summary["inference_ms_per_cycle"] = pcc["ms_per_cycle"]
            boot = inf.get("boot_all_cells", {})
            if boot.get("total_ms") and boot.get("n_cells"):
                summary["boot_cells_per_min"] = round(
                    boot["n_cells"] / (boot["total_ms"] / 60_000.0), 2
                )

        return summary

    # ── run all ───────────────────────────────────────────────────────────────

    def run_all(
        self,
        sections: Optional[list[str]] = None,
        tf_trace: bool = False,
        tf_trace_dir: Optional[str] = None,
    ) -> dict:
        """Run requested profiling sections, compute throughput summary, and save.

        Parameters
        ----------
        sections:
            Subset of ``["preprocessing", "sliding_windows", "inference",
            "stats", "memory"]``. Default: all.
        tf_trace:
            If True, also run the optional TF op-level profiler (requires model).
        tf_trace_dir:
            Directory for TF profiler output. Defaults to ``<output_dir>/tf_profile``.
        """
        all_sections = sections or ["preprocessing", "sliding_windows", "inference", "stats", "memory"]

        if "preprocessing" in all_sections:
            self.profile_preprocessing()
        if "sliding_windows" in all_sections:
            self.profile_sliding_windows()
        if "inference" in all_sections:
            self.profile_inference()
        if "stats" in all_sections:
            self.profile_statistical_tools()
        if "memory" in all_sections:
            self.profile_memory_footprint()
        if tf_trace:
            self.profile_tf_ops(trace_dir=tf_trace_dir)

        self._results["sections"]["throughput_summary"] = self._compute_throughput_summary()
        return self.save()

    # ── save + print ──────────────────────────────────────────────────────────

    def save(self) -> dict:
        """Write results to JSON and return the results dict."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = self.output_dir / f"{self.label}_{ts}.json"
        with open(fname, "w") as fh:
            json.dump(self._results, fh, indent=2, default=str)
        logger.info("Profile saved → %s", fname)
        if self.verbose:
            self._print_summary()
        return self._results

    def _print_summary(self) -> None:
        width = 72
        print("\n" + "=" * width)
        print(f"  Battery Profiler — {self.label}  |  {self._results['timestamp']}")
        print("=" * width)

        secs = self._results["sections"]
        pre = secs.get("preprocessing", {})
        if pre:
            print("\n  PREPROCESSING")
            for k, v in pre.items():
                if isinstance(v, dict) and "mean_ms" in v:
                    thr = v.get("throughput_per_sec", 0)
                    print(f"    {k:<46} {v['mean_ms']:>8.3f} ms   {thr:>8.0f}/s")

        sw = secs.get("sliding_windows", {})
        if sw:
            print("\n  SLIDING WINDOWS")
            for k, v in sw.items():
                if isinstance(v, dict) and "mean_ms" in v:
                    print(f"    {k:<46} {v['mean_ms']:>8.3f} ms")

        inf = secs.get("model_inference", {})
        if inf and not inf.get("skipped"):
            print("\n  MODEL INFERENCE")
            for k, v in inf.items():
                if isinstance(v, dict) and "mean_ms" in v:
                    mpc = v.get("ms_per_cycle", "")
                    suffix = f"  ({mpc:.4f} ms/cycle)" if mpc else ""
                    print(f"    {k:<46} {v['mean_ms']:>8.1f} ms{suffix}")
        elif inf.get("skipped"):
            print("\n  MODEL INFERENCE  (skipped — weights unavailable)")

        stats = secs.get("statistical_tools", {})
        if stats:
            print("\n  STATISTICAL TOOLS")
            for k, v in stats.items():
                if isinstance(v, dict) and "mean_ms" in v:
                    print(f"    {k:<46} {v['mean_ms']:>8.4f} ms")

        mem = self._results.get("memory", {})
        if mem:
            print("\n  MEMORY")
            for k, v in mem.items():
                if isinstance(v, (int, float)):
                    unit = " MB" if "mb" in k.lower() else " B"
                    print(f"    {k:<46} {v:>10.1f}{unit}")

        thr = secs.get("throughput_summary", {})
        if thr:
            print("\n  THROUGHPUT SUMMARY")
            for k, v in thr.items():
                print(f"    {k:<46} {v:>10.1f}")

        print("=" * width + "\n")

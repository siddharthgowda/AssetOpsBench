"""Battery MCP Server - lithium-ion analytics using the acctouhou pretrained model.

Tool surface (10):
    list_batteries, get_battery_cycle_summary,
    predict_rul, predict_rul_batch,
    predict_voltage_curve, predict_voltage_milestones,
    get_actual_voltage_milestones, get_impedance_trajectory,
    analyze_impedance_growth, detect_capacity_outliers,
    diagnose_battery.

Design:
    Boot does the absolute minimum - load TF models, precompile flexible-shape
    graphs, that's it. Tools fetch + preprocess + predict on demand. The fleet
    tool ``predict_rul_batch`` parallelizes CouchDB fetch and runs one batched
    TF call across all requested cells; the single-cell tool ``predict_rul``
    uses the precompiled graphs but predicts one cell at a time.

    The voltage tools (``predict_voltage_curve``, ``predict_voltage_milestones``)
    are deliberately the unoptimized reference: serial fetch, raw Keras models,
    per-cell predict. They exist to show what an un-optimized tool looks like
    and to anchor the benchmark's "naive_baseline" rung.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .couchdb_client import CouchDBClient
from .model_wrapper import (
    _load_once,
    cache_load,
    cache_save,
    get_compiled_models,
    model_available,
    predict_rul_for_cells,
    predict_voltage_for_cell,
)
from .preprocessing import preprocess_cell_from_couchdb

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("battery-mcp-server")

mcp = FastMCP("battery")

# Cells in the NASA B0xx dataset that have ≥100 paired charge/discharge cycles
# (the model's required input window). Used as ``predict_rul_batch``'s default
# when no ``asset_ids`` is supplied. Shorter cells fail in preprocessing and
# come back as per-row errors in the batch result.
_DEFAULT_MODEL_CELLS = [
    "B0005", "B0006", "B0007", "B0018",
    "B0033", "B0034", "B0036",
    "B0054", "B0055", "B0056",
]

_FETCH_WORKERS = max(1, int(os.environ.get("BATTERY_FETCH_WORKERS", "4") or 4))

# Optimization toggles. All default ON. The ablation profiler flips these via
# os.environ before spawning the MCP subprocess so we can A/B real scenarios.
_PARALLEL_FETCH = os.environ.get("BATTERY_PARALLEL_FETCH", "1").strip() == "1"
_GRAPH_PRECOMPILE = os.environ.get("BATTERY_GRAPH_PRECOMPILE", "1").strip() == "1"
_BATCHED_PREDICT = os.environ.get("BATTERY_BATCHED_PREDICT", "1").strip() == "1"
_DISK_CACHE = os.environ.get("BATTERY_DISK_CACHE", "1").strip() == "1"


# ── Pydantic result models ───────────────────────────────────────────────────


class ErrorResult(BaseModel):
    error: str


class BatteryListResult(BaseModel):
    cells: list[dict]


class CycleSummaryResult(BaseModel):
    asset_id: str
    n_discharge_cycles: int
    rows: list[dict]


class RULResult(BaseModel):
    asset_id: str
    rul_cycles: float
    from_cycle: int
    mae_cycles: Optional[float] = None


class RULBatchRow(BaseModel):
    asset_id: str
    rul_cycles: Optional[float] = None
    from_cycle: int = 0
    mae_cycles: Optional[float] = None
    error: Optional[str] = None


class RULBatchResult(BaseModel):
    rows: list[RULBatchRow]


class VoltageCurveResult(BaseModel):
    asset_id: str
    cycle_index: int
    voltage: list[float]


class MilestonesResult(BaseModel):
    asset_id: str
    crossings: dict[str, int]


class ActualMilestonesResult(BaseModel):
    asset_id: str
    crossings: dict[str, int]


class ImpedanceTrajectoryResult(BaseModel):
    asset_id: str
    cycles: list[int]
    rct: list[float]
    re: list[float]
    rectified_impedance_mag: list[Optional[float]]


class ImpedanceResult(BaseModel):
    asset_id: str
    rct_growth_per_cycle: float
    initial_rct: float
    final_rct: float
    alarm: bool


class OutlierResult(BaseModel):
    flagged_cells: list[str]
    z_scores: dict[str, float]


class DiagnosisResult(BaseModel):
    asset_id: str
    primary_mode: str
    severity: str
    explanation: str
    recommendations: list[str]
    numerical_findings: dict[str, float]


# ── Boot ─────────────────────────────────────────────────────────────────────


def _boot() -> None:
    """Minimal boot: load TF models and precompile flexible-shape graphs.
    No CouchDB fetches, no precompute - tools handle all of that on demand.
    Precompile is gated on ``BATTERY_GRAPH_PRECOMPILE`` so the ablation can
    measure boot cost both ways."""
    _load_once()
    if model_available() and _GRAPH_PRECOMPILE:
        get_compiled_models()  # warms _MODELS_COMPILED


_boot()


# ── Helpers used by RUL tools ────────────────────────────────────────────────


def _fetch_and_preprocess(asset_id: str, client: CouchDBClient):
    """Fetch one cell from CouchDB and produce its (charges, discharges, summary)
    tensors. Returns ``None`` on failure (logs the reason)."""
    try:
        return preprocess_cell_from_couchdb(asset_id, client)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipped %s: %s", asset_id, e)
        return None


def _parallel_fetch(asset_ids: list[str], client: CouchDBClient):
    """Parallel fetch + preprocess across N cells with a ThreadPoolExecutor.
    Returns a list of ``(asset_id, charges, discharges, summary)`` for cells
    that loaded successfully, plus a dict ``{asset_id: error_str}`` for the rest."""
    ok: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as ex:
        futures = {ex.submit(_fetch_and_preprocess, aid, client): aid for aid in asset_ids}
        for fut in as_completed(futures):
            aid = futures[fut]
            try:
                result = fut.result()
                if result is None:
                    errors[aid] = "preprocess failed (likely <100 paired cycles)"
                else:
                    ch, dis, summ = result
                    ok.append((aid, ch, dis, summ))
            except Exception as e:  # noqa: BLE001
                errors[aid] = f"{type(e).__name__}: {e}"
    # Preserve input order of successful cells.
    order = {aid: i for i, aid in enumerate(asset_ids)}
    ok.sort(key=lambda t: order.get(t[0], 999))
    return ok, errors


def _serial_fetch(asset_ids: list[str], client: CouchDBClient):
    """Serial fetch counterpart for the ``BATTERY_PARALLEL_FETCH=0`` ablation rung."""
    ok: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    errors: dict[str, str] = {}
    for aid in asset_ids:
        try:
            result = _fetch_and_preprocess(aid, client)
            if result is None:
                errors[aid] = "preprocess failed (likely <100 paired cycles)"
            else:
                ch, dis, summ = result
                ok.append((aid, ch, dis, summ))
        except Exception as e:  # noqa: BLE001
            errors[aid] = f"{type(e).__name__}: {e}"
    return ok, errors


def _ground_truth_rul(asset_id: str, from_cycle: int) -> Optional[float]:
    """Actual RUL = (first cycle where Capacity<1.4) - from_cycle, if known."""
    client = CouchDBClient()
    dis = client.fetch_cycles(asset_id, cycle_type="discharge")
    if not dis:
        return None
    caps: list[tuple[int, float]] = []
    for d in dis:
        c = d.get("data", {}).get("Capacity")
        if c is not None:
            caps.append((d.get("cycle_index", 0), float(c)))
    caps.sort(key=lambda x: x[0])
    eol_idx = next((i for i, (_, c) in enumerate(caps) if c < 1.4), None)
    if eol_idx is None:
        return None
    return float(eol_idx - from_cycle)


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_batteries(site_name: Optional[str] = None) -> Union[BatteryListResult, ErrorResult]:
    """List available lithium-ion battery cells for the fleet. Use this whenever the
    user asks about multiple cells, fleet rankings, top-N at-risk batteries, or
    when no specific cell_id is given - it is the entry point for most battery
    scenarios including RUL, voltage, impedance, and diagnosis queries."""
    _ = site_name or "MAIN"
    client = CouchDBClient()
    ids = client.list_cell_ids()
    if not ids:
        return ErrorResult(error="CouchDB battery database is empty or unreachable")
    rows = [{"asset_id": a, "model_ready": a in _DEFAULT_MODEL_CELLS} for a in ids]
    return BatteryListResult(cells=rows)


@mcp.tool()
def get_battery_cycle_summary(asset_id: str) -> Union[CycleSummaryResult, ErrorResult]:
    """Return per-cycle Capacity (Ah), max/avg Temperature, average Voltage, and Rct/Re for
    a lithium-ion cell. Use this when the user wants capacity degradation curves, SOH
    trajectories, or raw per-cycle health metrics. Works for any cell in CouchDB,
    including short-cycle cells that can't feed the pretrained model."""
    client = CouchDBClient()
    discharges = client.fetch_cycles(asset_id, cycle_type="discharge")
    impedances = client.fetch_cycles(asset_id, cycle_type="impedance")
    if not discharges:
        return ErrorResult(error=f"No discharge cycles for {asset_id}")
    imp_by_cycle = {imp.get("cycle_index"): imp for imp in impedances}
    rows: list[dict] = []
    for d in discharges:
        i = d.get("cycle_index")
        data = d.get("data", {})
        cap = data.get("Capacity")
        temps = data.get("Temperature_measured", [])
        volts = data.get("Voltage_measured", [])
        imp_data = imp_by_cycle.get(i, {}).get("data", {})
        rows.append(
            {
                "cycle_index": i,
                "capacity_ah": cap,
                "max_temp_c": max(temps) if temps else None,
                "min_temp_c": min(temps) if temps else None,
                "avg_voltage": (sum(volts) / len(volts)) if volts else None,
                "rct": imp_data.get("Rct"),
                "re": imp_data.get("Re"),
            }
        )
    return CycleSummaryResult(asset_id=asset_id, n_discharge_cycles=len(rows), rows=rows)


@mcp.tool()
def predict_rul(
    asset_id: str, from_cycle: int = 0
) -> Union[RULResult, ErrorResult]:
    """Predict remaining useful life (in cycles) to 30% capacity fade (1.4 Ah) for a
    lithium-ion cell. Use this when the user asks for RUL, cycles-to-failure, EOL
    timing, at-risk rankings, warranty analysis, or second-life assessment. Returns
    MAE vs ground truth when the cell has complete NASA history.

    Note: for fleet questions use ``predict_rul_batch`` instead - it parallelizes
    CouchDB fetch and runs one batched TF call across all cells."""
    if not model_available():
        return ErrorResult(error="Pretrained model unavailable")
    client = CouchDBClient()
    fetched = _fetch_and_preprocess(asset_id, client)
    if fetched is None:
        return ErrorResult(error=f"Could not preprocess {asset_id} (need ≥100 paired cycles)")
    ch, dis, summ = fetched
    n_ch, n_dis = int(ch.shape[0]), int(dis.shape[0])

    traj = cache_load(asset_id, n_ch, n_dis) if _DISK_CACHE else None
    if traj is None:
        traj = predict_rul_for_cells(
            [(ch, dis, summ)], use_compiled=_GRAPH_PRECOMPILE, batched=False
        )[0]
        if _DISK_CACHE:
            cache_save(asset_id, traj, n_ch, n_dis)
    idx = max(1, min(from_cycle, len(traj))) - 1
    predicted = float(traj[idx])
    gt = _ground_truth_rul(asset_id, from_cycle)
    mae = abs(predicted - gt) if gt is not None else None
    return RULResult(
        asset_id=asset_id,
        rul_cycles=predicted,
        from_cycle=from_cycle,
        mae_cycles=mae,
    )


@mcp.tool()
def predict_rul_batch(
    asset_ids: Optional[list[str]] = None,
    from_cycle: int = 0,
) -> Union[RULBatchResult, ErrorResult]:
    """Predict RUL for MULTIPLE lithium-ion cells in ONE call. Pass the full list
    of cells in ``asset_ids``. Defaults to the 10 NASA model-ready cells.

    USAGE:
      ✓ One call:        predict_rul_batch(asset_ids=["B0005","B0006","B0018"])
      ✓ All defaults:    predict_rul_batch()
      ✗ DO NOT foreach:  calling per-cell with {"asset_id": "B0005"} defeats batching.

    Preferred entry point for fleet-wide RUL queries (top-N at-risk, rankings, all
    cells). Each per-cell ``predict_rul`` would otherwise spawn a fresh MCP server
    process; one batched call amortizes that cold-start across the whole fleet."""
    if not model_available():
        return ErrorResult(error="Pretrained model unavailable")
    ids = asset_ids if asset_ids else _DEFAULT_MODEL_CELLS
    client = CouchDBClient()
    if _PARALLEL_FETCH:
        cells_ok, errors = _parallel_fetch(ids, client)
    else:
        cells_ok, errors = _serial_fetch(ids, client)

    rows: list[RULBatchRow] = []
    if cells_ok:
        # Disk cache: partition into hits / misses. Only the misses go through TF predict.
        if _DISK_CACHE:
            traj_by_aid: dict[str, np.ndarray] = {}
            misses_meta: list[tuple[str, int, int]] = []
            misses_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for aid, ch, dis, summ in cells_ok:
                n_ch, n_dis = int(ch.shape[0]), int(dis.shape[0])
                cached = cache_load(aid, n_ch, n_dis)
                if cached is not None:
                    traj_by_aid[aid] = cached
                else:
                    misses_meta.append((aid, n_ch, n_dis))
                    misses_data.append((ch, dis, summ))
            if misses_data:
                try:
                    miss_trajs = predict_rul_for_cells(
                        misses_data, use_compiled=_GRAPH_PRECOMPILE, batched=_BATCHED_PREDICT
                    )
                except ValueError:
                    miss_trajs = predict_rul_for_cells(
                        misses_data, use_compiled=_GRAPH_PRECOMPILE, batched=False
                    )
                for (aid, n_ch, n_dis), traj in zip(misses_meta, miss_trajs):
                    cache_save(aid, traj, n_ch, n_dis)
                    traj_by_aid[aid] = traj
            trajectories = [traj_by_aid[aid] for aid, _, _, _ in cells_ok]
        else:
            try:
                trajectories = predict_rul_for_cells(
                    [(ch, dis, summ) for _, ch, dis, summ in cells_ok],
                    use_compiled=_GRAPH_PRECOMPILE,
                    batched=_BATCHED_PREDICT,
                )
            except ValueError:
                trajectories = predict_rul_for_cells(
                    [(ch, dis, summ) for _, ch, dis, summ in cells_ok],
                    use_compiled=_GRAPH_PRECOMPILE,
                    batched=False,
                )
        for (aid, _, _, _), traj in zip(cells_ok, trajectories):
            idx = max(1, min(from_cycle, len(traj))) - 1
            predicted = float(traj[idx])
            gt = _ground_truth_rul(aid, from_cycle)
            mae = abs(predicted - gt) if gt is not None else None
            rows.append(
                RULBatchRow(
                    asset_id=aid,
                    rul_cycles=predicted,
                    from_cycle=from_cycle,
                    mae_cycles=mae,
                )
            )
    for aid in ids:
        if aid in errors:
            rows.append(RULBatchRow(asset_id=aid, from_cycle=from_cycle, error=errors[aid]))

    # Preserve the caller's original ordering.
    rank = {aid: i for i, aid in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r.asset_id, 999))
    return RULBatchResult(rows=rows)


# ── Voltage tools (deliberately UNOPTIMIZED - serial fetch, raw model) ────────
# These exist as the reference for "what does an unoptimized on-demand tool look
# like." They anchor the benchmark's naive_baseline rung. Do not add parallel
# fetch or compiled graphs here - that's the whole point.


def _voltage_curves_naive(asset_id: str) -> Union[np.ndarray, ErrorResult]:
    if not model_available():
        return ErrorResult(error="Pretrained model unavailable")
    client = CouchDBClient()
    fetched = _fetch_and_preprocess(asset_id, client)
    if fetched is None:
        return ErrorResult(error=f"Could not preprocess {asset_id} (need ≥100 paired cycles)")
    ch, dis, summ = fetched
    return predict_voltage_for_cell(ch, dis, summ)


@mcp.tool()
def predict_voltage_curve(
    asset_id: str, cycle_index: int = 0
) -> Union[VoltageCurveResult, ErrorResult]:
    """Predict the 100-point voltage-vs-SOC discharge curve for a lithium-ion cell at
    a given cycle. Use this when the user asks for voltage trajectories, discharge
    profiles, V-SOC curves, or end-of-discharge visualization at a specific cell age.

    Note: this tool intentionally uses the naive single-cell predict path (no
    parallel fetch, no compiled graphs). For fleet voltage queries, prefer the
    optimized RUL tools and reach for voltage only when needed per cell."""
    curves = _voltage_curves_naive(asset_id)
    if isinstance(curves, ErrorResult):
        return curves
    idx = max(0, min(cycle_index, len(curves) - 1))
    return VoltageCurveResult(
        asset_id=asset_id, cycle_index=idx, voltage=curves[idx].tolist()
    )


@mcp.tool()
def predict_voltage_milestones(
    asset_id: str, thresholds: list[float] = [2.9, 2.8, 2.7]
) -> Union[MilestonesResult, ErrorResult]:
    """Find the cycle where a lithium-ion cell's predicted voltage first drops below
    each threshold (EOD timing). Use this when the user asks about BMS alarms,
    end-of-discharge prediction, deep-discharge events, or 'when will voltage drop
    below X'. Default thresholds target auxiliary-system cutoffs at 2.9/2.8/2.7V."""
    curves = _voltage_curves_naive(asset_id)
    if isinstance(curves, ErrorResult):
        return curves
    crossings: dict[str, int] = {}
    for th in thresholds:
        found = next((i for i, curve in enumerate(curves) if curve.min() < th), -1)
        crossings[f"{th:.2f}V"] = int(found)
    return MilestonesResult(asset_id=asset_id, crossings=crossings)


@mcp.tool()
def get_actual_voltage_milestones(
    asset_id: str, thresholds: list[float] = [2.9, 2.8, 2.7]
) -> Union[ActualMilestonesResult, ErrorResult]:
    """Scan the cell's actual recorded discharge ``Voltage_measured`` arrays for
    threshold crossings - ground truth counterpart to ``predict_voltage_milestones``.
    Use this in combination with ``predict_voltage_milestones`` when the user asks
    for MAE/RMSE on EOD timing, voltage crossings, or predicted-vs-actual voltage
    drop comparisons. Works for any cell with at least one discharge cycle."""
    client = CouchDBClient()
    discharges = client.fetch_cycles(asset_id, cycle_type="discharge") or []
    if not discharges:
        return ErrorResult(error=f"No discharge cycles for {asset_id}")
    discharges.sort(key=lambda x: x.get("cycle_index", 0))
    crossings: dict[str, int] = {}
    for th in thresholds:
        found = -1
        for d in discharges:
            volts = d.get("data", {}).get("Voltage_measured") or []
            if volts and min(volts) < th:
                found = int(d.get("cycle_index", 0))
                break
        crossings[f"{th:.2f}V"] = found
    return ActualMilestonesResult(asset_id=asset_id, crossings=crossings)


@mcp.tool()
def get_impedance_trajectory(asset_id: str) -> Union[ImpedanceTrajectoryResult, ErrorResult]:
    """Return the per-cycle impedance trajectory (Rct, Re, and Rectified_Impedance
    magnitude) for a lithium-ion cell. Ground truth for RMSE computation against
    predicted impedance trajectories. Use this when the user asks for predicted-vs-
    actual impedance curves, sensor-drift validation, or impedance RMSE metrics."""
    client = CouchDBClient()
    impedances = client.fetch_cycles(asset_id, cycle_type="impedance") or []
    if not impedances:
        return ErrorResult(error=f"No impedance cycles for {asset_id}")
    impedances.sort(key=lambda x: x.get("cycle_index", 0))
    cycles: list[int] = []
    rct_vals: list[float] = []
    re_vals: list[float] = []
    rect_mag: list[Optional[float]] = []
    for imp in impedances:
        data = imp.get("data", {})
        rct = data.get("Rct")
        re_ = data.get("Re")
        if rct is None:
            continue
        try:
            cycles.append(int(imp.get("cycle_index", 0)))
            rct_vals.append(float(rct))
            re_vals.append(float(re_) if re_ is not None else 0.0)
        except (TypeError, ValueError):
            cycles.pop() if len(cycles) > len(rct_vals) else None
            continue
        ri_raw = data.get("Rectified_Impedance") or []
        mag: Optional[float] = None
        if ri_raw:
            try:
                mags = [abs(complex(s.strip())) for s in ri_raw if isinstance(s, str) and s.strip()]
                if mags:
                    mag = float(sum(mags) / len(mags))
            except (TypeError, ValueError):
                mag = None
        rect_mag.append(mag)
    if not cycles:
        return ErrorResult(error=f"No parseable impedance cycles for {asset_id}")
    return ImpedanceTrajectoryResult(
        asset_id=asset_id,
        cycles=cycles,
        rct=rct_vals,
        re=re_vals,
        rectified_impedance_mag=rect_mag,
    )


@mcp.tool()
def analyze_impedance_growth(asset_id: str) -> Union[ImpedanceResult, ErrorResult]:
    """Analyze charge-transfer resistance (Rct) growth for a lithium-ion cell via
    exponential curve fit across impedance cycles; returns fractional growth per cycle
    and an alarm flag. Use this for internal-resistance trends, electrolyte degradation,
    sensor drift checks, second-life safety screening, or thermal-runaway precursor
    detection. Works for any cell with ≥3 impedance cycles."""
    client = CouchDBClient()
    impedances = client.fetch_cycles(asset_id, cycle_type="impedance")
    rcts: list[tuple[int, float]] = []
    for imp in impedances:
        rct = imp.get("data", {}).get("Rct")
        if rct is not None:
            try:
                rcts.append((imp.get("cycle_index", 0), float(rct)))
            except (TypeError, ValueError):
                continue
    if len(rcts) < 3:
        return ErrorResult(
            error=f"Insufficient impedance data for {asset_id}: {len(rcts)} cycles (need ≥3)"
        )
    rcts.sort(key=lambda x: x[0])
    cycles = np.array([r[0] for r in rcts], dtype=float)
    rct_vals = np.array([r[1] for r in rcts], dtype=float)
    abs_rct = np.abs(rct_vals)
    abs_rct = np.where(abs_rct == 0, 1e-12, abs_rct)
    slope, _ = np.polyfit(cycles, np.log(abs_rct), 1)
    growth_per_cycle = float(np.exp(slope) - 1.0)
    initial = float(rct_vals[0])
    final = float(rct_vals[-1])
    alarm = bool(abs(final) > 1.5 * abs(initial))
    return ImpedanceResult(
        asset_id=asset_id,
        rct_growth_per_cycle=growth_per_cycle,
        initial_rct=initial,
        final_rct=final,
        alarm=alarm,
    )


def _scalar_capacity(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        try:
            return float(value[-1])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@mcp.tool()
def detect_capacity_outliers(
    asset_ids: Optional[list[str]] = None, window: int = 50
) -> Union[OutlierResult, ErrorResult]:
    """Flag lithium-ion cells degrading faster than fleet baseline via z-score on capacity
    fade rate. Use this when the user asks for top/bottom N cells, QA anomaly detection,
    manufacturing-defect screening, warranty-risk cells, 'which batteries are degrading
    fastest', or fleet ranking by health. Returns ranked list of flagged cells (z>2)
    plus per-cell z-scores for the planner to sort."""
    client = CouchDBClient()
    cells = asset_ids or client.list_cell_ids()
    if not cells:
        return ErrorResult(error="No cells provided and CouchDB has none")
    fade_rates: dict[str, float] = {}
    for cell in cells:
        dis = client.fetch_cycles(cell, cycle_type="discharge")
        caps = []
        for d in dis:
            sc = _scalar_capacity(d.get("data", {}).get("Capacity"))
            if sc is not None:
                caps.append((d.get("cycle_index", 0), sc))
        caps.sort(key=lambda x: x[0])
        if len(caps) < 2:
            continue
        head = caps[: min(window, len(caps))]
        fade = (float(head[0][1]) - float(head[-1][1])) / max(len(head), 1)
        fade_rates[cell] = fade
    if not fade_rates:
        return OutlierResult(flagged_cells=[], z_scores={})
    rates = np.array(list(fade_rates.values()))
    mean_r = float(rates.mean())
    std_r = float(rates.std()) or 1.0
    z_scores = {c: (r - mean_r) / std_r for c, r in fade_rates.items()}
    flagged = [c for c, z in sorted(z_scores.items(), key=lambda kv: -kv[1]) if z > 2.0]
    return OutlierResult(flagged_cells=flagged, z_scores=z_scores)


@mcp.tool()
def diagnose_battery(asset_id: str) -> Union[DiagnosisResult, ErrorResult]:
    """Run a full diagnostic for a lithium-ion cell: combines RUL prediction, impedance
    growth fit, and fleet-outlier z-score, then uses an LLM to classify the primary
    degradation mode (capacity_fade / lithium_plating / impedance_growth / healthy) with
    severity and recommendations. Use this when the user asks for a cell assessment,
    failure-mode analysis, 'what's wrong with X', or a one-shot summary."""
    from .diagnosis import diagnose

    return diagnose(asset_id)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

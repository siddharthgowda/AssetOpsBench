"""Battery MCP Server — lithium-ion analytics using the acctouhou pretrained model.

Tool surface (8):
    list_batteries, get_battery_cycle_summary,
    predict_rul, predict_voltage_curve, predict_voltage_milestones,
    analyze_impedance_growth, detect_capacity_outliers,
    diagnose_battery.

Startup:
    _boot() loads the pretrained weights (if available) and precomputes RUL +
    voltage curves for the 10 model-ready NASA cells into an in-memory cache.
    Each tool then returns slices of the cache or runs a fast CouchDB-backed
    statistical analysis.

Docstrings emphasize *when to use* each tool because the plan-execute agent
routes based on the first line. Abstract client scenarios are orchestrated by
the planner: tools stay single-cell primitives.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from . import model_wrapper
from .couchdb_client import CouchDBClient
from .model_wrapper import _CACHE, _load_once, precompute_cell
from .preprocessing import preprocess_cell_from_couchdb


def _model_available() -> bool:
    """Live accessor — `from module import flag` captures at import time and
    won't see the flag flip to True inside `_load_once()`."""
    return model_wrapper._MODEL_AVAILABLE

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("battery-mcp-server")

mcp = FastMCP("battery")


# ── Pydantic result models ───────────────────────────────────────────────────


class ErrorResult(BaseModel):
    error: str


class BatteryListResult(BaseModel):
    cells: list[dict]  # [{asset_id, model_ready, n_cycles}]


class CycleSummaryResult(BaseModel):
    asset_id: str
    n_discharge_cycles: int
    rows: list[dict]


class RULResult(BaseModel):
    asset_id: str
    rul_cycles: float
    from_cycle: int
    inference_ms: float
    mae_cycles: Optional[float] = None  # vs ground truth when known


class VoltageCurveResult(BaseModel):
    asset_id: str
    cycle_index: int
    voltage: list[float]  # 100-point predicted V-SOC curve (output length from predictor2.h5)


class MilestonesResult(BaseModel):
    asset_id: str
    crossings: dict[str, int]  # "2.90V" → cycle_index where first crossed; -1 if never


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


# ── Cells preloaded at boot (model-ready subset) ─────────────────────────────

# These 10 cells have ≥100 clean paired charge/discharge cycles, so they feed
# the pretrained model's 100-cycle history window. Statistical tools work for
# all 34 cells.
_USABLE_MODEL_CELLS = [
    "B0005", "B0006", "B0007", "B0018",
    "B0033", "B0034", "B0036",
    "B0054", "B0055", "B0056",
]


def _boot() -> None:
    _load_once()
    client = CouchDBClient()
    if not client.available:
        logger.warning("CouchDB unavailable — battery tools will return errors until DB is reachable")
        return
    if not _model_available():
        logger.warning(
            "Pretrained model unavailable — only statistical tools will work. "
            "Check BATTERY_MODEL_WEIGHTS_DIR and BATTERY_NORMS_DIR."
        )
        return
    for cell in _USABLE_MODEL_CELLS:
        try:
            ch, dis, summ = preprocess_cell_from_couchdb(cell, client)
            precompute_cell(cell, ch, dis, summ)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipped %s: %s", cell, e)
    logger.info("%d cells preloaded in battery cache", len(_CACHE))


_boot()


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_batteries(site_name: str = "MAIN") -> Union[BatteryListResult, ErrorResult]:
    """List available lithium-ion battery cells for the fleet. Use this whenever the
    user asks about multiple cells, fleet rankings, top-N at-risk batteries, or
    when no specific cell_id is given — it is the entry point for most battery
    scenarios including RUL, voltage, impedance, and diagnosis queries."""
    client = CouchDBClient()
    ids = client.list_cell_ids()
    if not ids:
        return ErrorResult(error="CouchDB battery database is empty or unreachable")
    rows = [
        {"asset_id": a, "model_ready": a in _CACHE}
        for a in ids
    ]
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


def _ground_truth_rul(asset_id: str, from_cycle: int) -> Optional[float]:
    """Compute actual RUL = (first cycle where Capacity<1.4) - from_cycle, if known."""
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


@mcp.tool()
def predict_rul(
    asset_id: str, from_cycle: int = 100
) -> Union[RULResult, ErrorResult]:
    """Predict remaining useful life (in cycles) to 30% capacity fade (1.4 Ah) for a
    lithium-ion cell. Use this when the user asks for RUL, cycles-to-failure, EOL
    timing, at-risk rankings, warranty analysis, or second-life assessment. Returns
    MAE vs ground truth when the cell has complete NASA history."""
    if not _model_available():
        return ErrorResult(
            error=(
                "Pretrained model unavailable. See battery.md (repo root) for "
                "weight/norm setup, or run scripts/setup_battery_artifacts.sh."
            )
        )
    entry = _CACHE.get(asset_id)
    if not entry:
        return ErrorResult(
            error=(
                f"No cached inference for {asset_id}. Model-ready cells: "
                f"{sorted(_CACHE.keys())}."
            )
        )
    idx = max(1, min(from_cycle, len(entry["rul_trajectory"]))) - 1
    predicted = float(entry["rul_trajectory"][idx])
    gt = _ground_truth_rul(asset_id, from_cycle)
    mae = abs(predicted - gt) if gt is not None else None
    return RULResult(
        asset_id=asset_id,
        rul_cycles=predicted,
        from_cycle=from_cycle,
        inference_ms=entry["inference_ms_per_cycle"],
        mae_cycles=mae,
    )


@mcp.tool()
def predict_voltage_curve(
    asset_id: str, cycle_index: int
) -> Union[VoltageCurveResult, ErrorResult]:
    """Predict the 500-point voltage-vs-SOC discharge curve for a lithium-ion cell at a
    given cycle. Use this when the user asks for voltage trajectories, discharge
    profiles, V-SOC curves, or end-of-discharge visualization at a specific cell age."""
    if not _model_available():
        return ErrorResult(error="Pretrained model unavailable")
    entry = _CACHE.get(asset_id)
    if not entry:
        return ErrorResult(error=f"No cached inference for {asset_id}")
    curves = entry["voltage_curves"]
    idx = max(0, min(cycle_index, len(curves) - 1))
    return VoltageCurveResult(
        asset_id=asset_id,
        cycle_index=idx,
        voltage=curves[idx].tolist(),
    )


@mcp.tool()
def predict_voltage_milestones(
    asset_id: str,
    thresholds: list[float] = [2.9, 2.8, 2.7],
) -> Union[MilestonesResult, ErrorResult]:
    """Find the cycle where a lithium-ion cell's predicted voltage first drops below each
    threshold (EOD timing). Use this when the user asks about BMS alarms, end-of-discharge
    prediction, deep-discharge events, or 'when will voltage drop below X'. Default
    thresholds target auxiliary-system cutoffs at 2.9/2.8/2.7V."""
    if not _model_available():
        return ErrorResult(error="Pretrained model unavailable")
    entry = _CACHE.get(asset_id)
    if not entry:
        return ErrorResult(error=f"No cached inference for {asset_id}")
    curves = entry["voltage_curves"]
    crossings: dict[str, int] = {}
    for th in thresholds:
        found = next(
            (i for i, curve in enumerate(curves) if curve.min() < th),
            -1,
        )
        crossings[f"{th:.2f}V"] = int(found)
    return MilestonesResult(asset_id=asset_id, crossings=crossings)


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
    # Linear fit in log space: Rct(n) = a * exp(b*n) → log|Rct| = log|a| + b*n
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


@mcp.tool()
def detect_capacity_outliers(
    asset_ids: Optional[list[str]] = None,
    window: int = 50,
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
        caps = [
            (d.get("cycle_index", 0), d.get("data", {}).get("Capacity"))
            for d in dis
        ]
        caps = [(i, c) for i, c in caps if c is not None]
        caps.sort(key=lambda x: x[0])
        if len(caps) < 2:
            continue
        # Fade rate = (Cap_0 - Cap_min(window)) / n over the first `window` cycles
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

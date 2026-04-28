"""Synthetic NASA-like battery cycle data for profiling without CouchDB.

Generates realistic charge / discharge / impedance cycles that match the
exact dict schema consumed by ``preprocessing.py`` and ``couchdb_client.py``.
Data shapes mirror real NASA B0xx measurements:
  - discharge: ~500 time-points over ~3200 s, V from 4.1 → 2.8 V
  - charge:    ~600 time-points over ~4200 s, CC→CV profile
  - impedance: scalar Rct / Re + list of complex-string Rectified_Impedance
"""
from __future__ import annotations

import math
import random
from typing import Optional


def _make_discharge_cycle(
    cycle_idx: int,
    asset_id: str,
    n_points: int = 500,
    capacity_ah: float = 1.85,
    seed: Optional[int] = None,
) -> dict:
    rng = random.Random(seed)
    t_end = 3200.0 + rng.uniform(-200.0, 200.0)
    time = [i * t_end / (n_points - 1) for i in range(n_points)]
    voltage = [
        4.1 - 1.3 * (i / (n_points - 1)) ** 1.05 + rng.gauss(0.0, 0.008)
        for i in range(n_points)
    ]
    current = [-1.95 + rng.gauss(0.0, 0.03) for _ in range(n_points)]
    temp = [
        30.0 + 10.0 * math.sin(math.pi * i / n_points) + rng.gauss(0.0, 0.4)
        for i in range(n_points)
    ]
    return {
        "asset_id": asset_id,
        "cycle_index": cycle_idx,
        "cycle_type": "discharge",
        "data": {
            "Time": time,
            "Voltage_measured": voltage,
            "Current_measured": current,
            "Temperature_measured": temp,
            "Capacity": capacity_ah,
        },
    }


def _make_charge_cycle(
    cycle_idx: int,
    asset_id: str,
    n_points: int = 600,
    seed: Optional[int] = None,
) -> dict:
    rng = random.Random(seed)
    t_end = 4200.0 + rng.uniform(-300.0, 300.0)
    time = [i * t_end / (n_points - 1) for i in range(n_points)]
    voltage = [
        3.5 + 0.7 * (i / (n_points - 1)) ** 0.75 + rng.gauss(0.0, 0.006)
        for i in range(n_points)
    ]
    current = [
        max(0.01, 1.5 - 1.3 * (i / n_points) ** 2.5 + rng.gauss(0.0, 0.02))
        for i in range(n_points)
    ]
    temp = [
        27.0 + 8.0 * math.sin(math.pi * i / n_points) + rng.gauss(0.0, 0.3)
        for i in range(n_points)
    ]
    return {
        "asset_id": asset_id,
        "cycle_index": cycle_idx,
        "cycle_type": "charge",
        "data": {
            "Time": time,
            "Voltage_measured": voltage,
            "Current_measured": current,
            "Temperature_measured": temp,
        },
    }


def _make_impedance_cycle(
    cycle_idx: int,
    asset_id: str,
    rct_base: float = 0.15,
    seed: Optional[int] = None,
) -> dict:
    rng = random.Random(seed)
    rct = rct_base * (1.0 + 0.003 * cycle_idx) + rng.gauss(0.0, 0.004)
    re = 0.08 + 0.001 * cycle_idx + rng.gauss(0.0, 0.002)
    ri_list = [f"({rct:.5f}+{re:.5f}j)"] * 5
    return {
        "asset_id": asset_id,
        "cycle_index": cycle_idx,
        "cycle_type": "impedance",
        "data": {
            "Rct": rct,
            "Re": re,
            "Rectified_Impedance": ri_list,
        },
    }


def make_cell_cycles(
    asset_id: str = "MOCK_00",
    n_cycles: int = 100,
    base_seed: int = 42,
) -> dict[str, list[dict]]:
    """Return charge / discharge / impedance cycle lists for one synthetic cell.

    The generated data is structurally identical to what CouchDB returns for a
    real NASA B0xx cell, allowing profiling without a live database.
    """
    charges, discharges, impedances = [], [], []
    for i in range(n_cycles):
        fade = max(0.70, 1.0 - 0.003 * i)
        charges.append(_make_charge_cycle(i, asset_id, seed=base_seed + i * 3))
        discharges.append(
            _make_discharge_cycle(i, asset_id, capacity_ah=1.85 * fade, seed=base_seed + i * 3 + 1)
        )
        impedances.append(_make_impedance_cycle(i, asset_id, seed=base_seed + i * 3 + 2))
    return {"charge": charges, "discharge": discharges, "impedance": impedances}


def make_fleet(
    n_cells: int = 5,
    n_cycles: int = 100,
) -> dict[str, dict[str, list[dict]]]:
    """Return a synthetic fleet keyed by asset_id."""
    return {
        f"MOCK_{i:02d}": make_cell_cycles(
            asset_id=f"MOCK_{i:02d}",
            n_cycles=n_cycles,
            base_seed=i * 317,
        )
        for i in range(n_cells)
    }


class MockCouchDBClient:
    """Drop-in replacement for CouchDBClient that serves synthetic fleet data.

    Used by the profiler so preprocessing and statistical-tool sections run
    without a live CouchDB instance.
    """

    def __init__(self, fleet: Optional[dict] = None, n_cycles: int = 100) -> None:
        self._fleet: dict = fleet or make_fleet(n_cells=5, n_cycles=n_cycles)
        self.available: bool = True

    def list_cell_ids(self) -> list[str]:
        return sorted(self._fleet.keys())

    def fetch_cycles(
        self,
        asset_id: str,
        cycle_type: Optional[str] = None,
        limit: int = 10_000,
    ) -> list[dict]:
        cell = self._fleet.get(asset_id, {})
        if cycle_type:
            return cell.get(cycle_type, [])[:limit]
        return [doc for docs in cell.values() for doc in docs][:limit]

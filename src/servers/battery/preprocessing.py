"""NASA JSON cycle → (4, 500) tensor [Q, V, I, T] for the acctouhou pretrained model.

`inp_500` is copied verbatim from acctouhou's `data_processing.ipynb`. The Coulomb
counting (cumulative_trapezoid of |I| over time) derives the Q channel from the
raw current/time arrays — NASA stores only a scalar `Capacity` per discharge,
not the full Q(t) curve Severson's pipeline assumes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

logger = logging.getLogger("battery-mcp-server")

_MIN_SAMPLES_PER_CYCLE = 100  # below this, interpolation to 500 steps is meaningless


def inp_500(x, t):
    """Linear interpolation to 500 uniform timesteps. Copied from acctouhou/data_processing.ipynb."""
    f = interp1d(t, x, kind="linear")
    t_new = np.linspace(t.min(), t.max(), num=500)
    return f(t_new)


def preprocess_cycle(data: dict) -> Optional[np.ndarray]:
    """Return (4, 500) tensor [Q, V, I, T] for one NASA cycle, or None if too short.

    Channels:
        Q = cumulative capacity (Ah), derived via Coulomb counting on |I| over time
        V = Voltage_measured
        I = Current_measured
        T = Temperature_measured
    """
    if not data:
        return None
    t = np.asarray(data.get("Time", []), dtype=float)
    if len(t) < _MIN_SAMPLES_PER_CYCLE:
        return None
    try:
        V = np.asarray(data["Voltage_measured"], dtype=float)
        I = np.asarray(data["Current_measured"], dtype=float)
        T = np.asarray(data["Temperature_measured"], dtype=float)
    except KeyError:
        return None
    if len(V) < _MIN_SAMPLES_PER_CYCLE:
        return None
    # Ensure all channels agree in length; the NASA time vector is canonical.
    n = min(len(t), len(V), len(I), len(T))
    t, V, I, T = t[:n], V[:n], I[:n], T[:n]
    # Coulomb counting: time is in seconds → divide by 3600 to get Ah
    Q = cumulative_trapezoid(np.abs(I), t, initial=0) / 3600.0
    try:
        return np.stack([inp_500(Q, t), inp_500(V, t), inp_500(I, t), inp_500(T, t)])
    except ValueError:
        # e.g. duplicate t values → interp1d fails
        return None


def preprocess_cell_from_couchdb(
    cell_id: str, client
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch a cell's cycles from CouchDB and build the (charge, discharge, summary) tensors
    the acctouhou pipeline expects.

    Returns:
        charge:    (100, 4, 500)
        discharge: (100, 4, 500)
        summary:   (100, 6) — 6 features per cycle matching summary_norm.npy shape.
            Inferred from the .npy means/scales:
              [Qd (Ah), Qc (Ah), Tavg_dis (°C), Tmin_dis (°C), Tmax_dis (°C),
               chargetime (min)]
    Raises:
        ValueError if fewer than 100 clean paired cycles can be built.
    """
    charges = client.fetch_cycles(cell_id, cycle_type="charge") or []
    discharges = client.fetch_cycles(cell_id, cycle_type="discharge") or []
    charges = sorted(charges, key=lambda c: c.get("cycle_index", 0))
    discharges = sorted(discharges, key=lambda c: c.get("cycle_index", 0))
    n = min(len(charges), len(discharges), 100)
    ch_list, dis_list, summ_list = [], [], []
    for i in range(n):
        c_data = charges[i].get("data", {})
        d_data = discharges[i].get("data", {})
        c_t = preprocess_cycle(c_data)
        d_t = preprocess_cycle(d_data)
        if c_t is None or d_t is None:
            continue
        ch_list.append(c_t)
        dis_list.append(d_t)

        # --- 6-element summary (matches summary_norm.npy shape (6,)) ---
        # Qd = scalar discharge capacity (Ah)
        Qd = float(d_data.get("Capacity", 0.0) or 0.0)
        # Qc ≈ integrated |I| over charge (Ah); approximation since NASA doesn't store scalar charge capacity
        c_time = c_data.get("Time", [])
        c_I = c_data.get("Current_measured", [])
        if c_time and c_I and len(c_time) >= 2:
            Qc = float(np.trapezoid(np.abs(np.asarray(c_I, dtype=float)), np.asarray(c_time, dtype=float)) / 3600.0)
        else:
            Qc = Qd
        d_temps = d_data.get("Temperature_measured", [])
        if d_temps:
            Tavg = float(np.mean(d_temps))
            Tmin = float(np.min(d_temps))
            Tmax = float(np.max(d_temps))
        else:
            Tavg = Tmin = Tmax = 25.0
        # chargetime in MINUTES (NASA Time is in seconds)
        chargetime_min = (float(c_time[-1]) / 60.0) if c_time else 0.0
        summ_list.append([Qd, Qc, Tavg, Tmin, Tmax, chargetime_min])

    if len(ch_list) < 100:
        raise ValueError(
            f"{cell_id}: only {len(ch_list)} clean paired cycles (need 100 for the model window)"
        )
    return (
        np.array(ch_list[:100]),
        np.array(dis_list[:100]),
        np.array(summ_list[:100]),
    )

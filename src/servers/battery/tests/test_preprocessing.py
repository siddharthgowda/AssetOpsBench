"""Unit tests for preprocessing (no CouchDB, no TF required)."""

import glob
import json
import os

import numpy as np
import pytest

from servers.battery.preprocessing import preprocess_cycle


def _load_first_discharge(cell_id: str) -> dict:
    data_dir = os.environ.get("BATTERY_DATA_DIR", "external/battery/nasa")
    path = os.path.join(data_dir, f"{cell_id}.json")
    if not os.path.exists(path):
        candidates = glob.glob(os.path.join(data_dir, "*", f"{cell_id}.json"))
        path = candidates[0] if candidates else path
    with open(path) as f:
        raw = json.load(f)
    return next(c for c in raw[cell_id]["cycle"] if c["type"] == "discharge")


def test_preprocess_cycle_shape_b0005():
    dis0 = _load_first_discharge("B0005")
    tensor = preprocess_cycle(dis0["data"])
    assert tensor is not None
    assert tensor.shape == (4, 500)
    assert not np.any(np.isnan(tensor))


def test_preprocess_cycle_shape_b0018():
    dis0 = _load_first_discharge("B0018")
    tensor = preprocess_cycle(dis0["data"])
    assert tensor is not None
    assert tensor.shape == (4, 500)


def test_short_cycles_return_none():
    fake = {
        "Time": list(range(50)),
        "Voltage_measured": [3.0] * 50,
        "Current_measured": [-2.0] * 50,
        "Temperature_measured": [25.0] * 50,
    }
    assert preprocess_cycle(fake) is None


def test_missing_fields_return_none():
    fake = {"Time": list(range(200))}
    assert preprocess_cycle(fake) is None


def test_coulomb_counting_nonnegative():
    """Derived Q channel should be monotonically non-decreasing for a discharge."""
    dis0 = _load_first_discharge("B0005")
    tensor = preprocess_cycle(dis0["data"])
    Q = tensor[0]
    # Q is cumulative integral of |I|; it should be monotonically non-decreasing
    diffs = np.diff(Q)
    assert (diffs >= -1e-6).all(), "Q channel should be non-decreasing"

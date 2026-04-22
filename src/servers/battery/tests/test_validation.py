"""Validation gate: predict_rul on B0018 should be within ±30 cycles of ground truth.

Skipped if weights or CouchDB are missing. A loud fail here = the LFP→NCA chemistry
mismatch is too large for zero-shot use; the README documents the fallback.
"""

import glob
import json
import os

import pytest

from servers.battery.main import ErrorResult, RULResult, predict_rul


def test_b0018_rul_mae_within_30_cycles(requires_couchdb, requires_weights):
    data_dir = os.environ.get("BATTERY_DATA_DIR", "external/battery/nasa")
    candidates = glob.glob(os.path.join(data_dir, "B0018.json")) + glob.glob(
        os.path.join(data_dir, "*", "B0018.json")
    )
    if not candidates:
        pytest.skip("B0018.json not found in BATTERY_DATA_DIR")
    with open(candidates[0]) as f:
        raw = json.load(f)
    caps = [
        c["data"]["Capacity"]
        for c in raw["B0018"]["cycle"]
        if c["type"] == "discharge" and "Capacity" in c["data"]
    ]
    eol_cycle = next((i for i, c in enumerate(caps) if c < 1.4), len(caps))
    gt_rul_at_100 = eol_cycle - 100

    pred = predict_rul("B0018", from_cycle=100)
    if isinstance(pred, ErrorResult):
        pytest.skip(f"B0018 not model-ready: {pred.error}")
    assert isinstance(pred, RULResult)

    # Known limitation: acctouhou was trained on Severson LFP; NASA B0xx is NCA.
    # We expect high MAE without fine-tuning. This test codifies the gate in the
    # plan: if it fails, README must document the fallback.
    assert abs(pred.rul_cycles - gt_rul_at_100) <= 30, (
        f"B0018 RUL MAE too large (LFP→NCA transfer): predicted "
        f"{pred.rul_cycles:.1f}, ground truth {gt_rul_at_100}, |diff|="
        f"{abs(pred.rul_cycles - gt_rul_at_100):.1f}"
    )

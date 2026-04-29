"""Regression checks for inference speed / precompute (needs weights)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from servers.battery import model_wrapper as mw


def test_precompute_per_cycle_ms_cap(requires_weights) -> None:
    mw._load_once()
    if not mw._MODEL_AVAILABLE:
        pytest.skip("Model failed to load")
    rng = np.random.default_rng(42)
    n = 100
    ch = rng.standard_normal((n, 4, 500), dtype=np.float32)
    dis = rng.standard_normal((n, 4, 500), dtype=np.float32)
    summ = rng.standard_normal((n, 6), dtype=np.float32)
    t0 = time.perf_counter()
    mw.precompute_cell("_pytest_speed_cell", ch, dis, summ)
    elapsed = time.perf_counter() - t0
    ms_per = mw._CACHE["_pytest_speed_cell"]["inference_ms_per_cycle"]
    assert ms_per < 30_000, f"inference_ms_per_cycle {ms_per} absurdly high"
    assert elapsed < 600.0, f"total precompute {elapsed}s exceeds 10min sanity cap"
    del mw._CACHE["_pytest_speed_cell"]

"""Unit tests for benchmark helpers (no TensorFlow import)."""

from __future__ import annotations

import sys

import numpy as np

from servers.battery import benchmark_inference as bi


def test_parse_csv_ints() -> None:
    assert bi._parse_csv_ints(None) is None
    assert bi._parse_csv_ints("") is None
    assert bi._parse_csv_ints("1, 2 ,4") == [1, 2, 4]


def test_merge_measurement_per_battery() -> None:
    blob = {"wall_ms": {"mean": 400.0, "min": 380.0, "max": 420.0, "n": 3}, "cpu_process_ms": {}}
    m = bi._merge_measurement(blob, n_batteries=4)
    assert m["wall_ms_per_battery"]["mean"] == 100.0
    assert m["wall_ms_per_battery"]["min"] == 95.0


def test_strip_tf_thread_flags() -> None:
    argv = ["--label", "x", "--intra-op-threads", "4", "--foo", "--inter-op-threads=2", "bar"]
    assert bi._strip_tf_thread_flags(argv) == ["--label", "x", "--foo", "bar"]


def test_argv_for_sweep_child() -> None:
    argv = ["--sweep-intra", "1,2", "--label", "z", "--intra-op-threads", "8"]
    assert bi._argv_for_sweep_child(argv) == ["--label", "z"]


def test_inject_label_suffix() -> None:
    assert bi._inject_label_suffix(["--label", "base"], "_s") == ["--label", "base_s"]
    assert bi._inject_label_suffix(["--n-batteries", "2"], "_s") == [
        "--n-batteries",
        "2",
        "--label",
        "inference_sweep_s",
    ]


def test_sliding_windows_matches_naive_stack() -> None:
    from servers.battery.model_wrapper import _pad_edge, build_sliding_windows

    rng = np.random.default_rng(0)
    cell_feat = rng.standard_normal((15, 12), dtype=np.float32)
    expected = np.stack(
        [_pad_edge(cell_feat[max(0, k - 49) : k + 1], 50) for k in range(len(cell_feat))]
    )
    got = build_sliding_windows(cell_feat)
    np.testing.assert_array_almost_equal(expected, got, decimal=5)


def test_module_imports_without_tensorflow() -> None:
    """Heavy ML deps are optional; this module must not import tf at import time."""
    assert "tensorflow" not in sys.modules

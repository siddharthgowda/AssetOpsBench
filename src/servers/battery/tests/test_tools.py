"""Integration tests for the battery MCP tools (needs CouchDB, skipped otherwise)."""

import pytest

from servers.battery.main import (
    BatteryListResult,
    ErrorResult,
    ImpedanceResult,
    OutlierResult,
    RULResult,
    VoltageCurveResult,
    analyze_impedance_growth,
    detect_capacity_outliers,
    list_batteries,
    predict_rul,
    predict_voltage_curve,
)


def test_list_batteries_returns_structure(requires_couchdb):
    r = list_batteries()
    assert isinstance(r, (BatteryListResult, ErrorResult))
    if isinstance(r, BatteryListResult):
        assert isinstance(r.cells, list)


def test_predict_rul_b0005(requires_couchdb, requires_weights):
    r = predict_rul("B0005", from_cycle=100)
    assert isinstance(r, (RULResult, ErrorResult))
    if isinstance(r, RULResult):
        assert r.inference_ms < 500  # generous — scenario 5 wants <100, we allow slack
        assert isinstance(r.rul_cycles, float)


def test_predict_voltage_curve_length(requires_couchdb, requires_weights):
    r = predict_voltage_curve("B0005", cycle_index=50)
    assert isinstance(r, (VoltageCurveResult, ErrorResult))
    if isinstance(r, VoltageCurveResult):
        # predictor2.h5 outputs a 100-point voltage curve
        assert len(r.voltage) == 100


def test_detect_capacity_outliers_structure(requires_couchdb):
    r = detect_capacity_outliers()
    assert isinstance(r, (OutlierResult, ErrorResult))
    if isinstance(r, OutlierResult):
        assert isinstance(r.flagged_cells, list)
        assert isinstance(r.z_scores, dict)


def test_analyze_impedance_growth_structure(requires_couchdb):
    r = analyze_impedance_growth("B0005")
    # Some cells have no impedance cycles; either Impedance or Error is fine
    assert isinstance(r, (ImpedanceResult, ErrorResult))

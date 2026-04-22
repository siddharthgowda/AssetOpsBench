"""Pytest skip markers for battery tests that need backends.

Mirrors src/servers/vibration/tests/conftest.py.
"""

import os
from pathlib import Path

import pytest


def _has_couchdb() -> bool:
    return bool(os.environ.get("COUCHDB_URL"))


def _has_weights() -> bool:
    weights = Path(os.environ.get("BATTERY_MODEL_WEIGHTS_DIR", "external/battery/acctouhou/weights"))
    return (weights / "predictor.h5").exists()


def _has_nasa_data() -> bool:
    data_dir = Path(os.environ.get("BATTERY_DATA_DIR", "external/battery/nasa"))
    return data_dir.exists() and any(data_dir.glob("B*.json"))


@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not _has_nasa_data():
        pytest.skip("NASA battery data not available at BATTERY_DATA_DIR")


@pytest.fixture
def requires_couchdb():
    if not _has_couchdb():
        pytest.skip("COUCHDB_URL not set")


@pytest.fixture
def requires_weights():
    if not _has_weights():
        pytest.skip("Pretrained weights not in BATTERY_MODEL_WEIGHTS_DIR")

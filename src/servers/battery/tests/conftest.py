"""Pytest skip marker: skip preprocessing tests if NASA cycle data isn't on disk."""

import os
from pathlib import Path

import pytest


def _has_nasa_data() -> bool:
    data_dir = Path(os.environ.get("BATTERY_DATA_DIR", "external/battery/nasa"))
    return data_dir.exists() and any(data_dir.glob("B*.json"))


@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not _has_nasa_data():
        pytest.skip("NASA battery data not available at BATTERY_DATA_DIR")

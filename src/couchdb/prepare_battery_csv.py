"""One-time script to generate a per-cycle battery metrics CSV from raw JSON.

Reads the full battery dataset and aggregates to one row per (asset_id,
cycle_index), merging metrics from discharge, charge, and impedance cycles
together. Assigns evenly-spaced timestamps so TSFM's data quality filter
treats the series as continuous.

Usage:
    python src/couchdb/prepare_battery_csv.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
_INPUT_FILE = os.path.join(_REPO_ROOT, ".data", "battery_full_data.json")
_OUTPUT_FILE = os.path.join(
    _SCRIPT_DIR,
    "..",
    "servers",
    "tsfm",
    "artifacts",
    "output",
    "tuned_datasets",
    "battery_cycle_metrics.csv",
)

# Assets with enough cycles (>= 96) for the ttm_96_28 model.
# Picked diverse cycle counts for varied RUL scenarios.
_SELECTED_ASSETS = [
    "B0029", "B0030", "B0031", "B0032",  # 96 cycles each
    "B0033", "B0034", "B0036",           # 485 cycles each (long life)
    "B0038", "B0039", "B0040",           # 121 cycles each
    "B0041",                             # 162 cycles
    "B0042", "B0043", "B0044",           # 274 cycles each
    "B0045", "B0046", "B0047", "B0048",  # 183 cycles each
    "B0053",                             # 136 cycles
    "B0054", "B0055", "B0056",           # 251-252 cycles
]


def main() -> None:
    print(f"Reading {_INPUT_FILE} ...")
    with open(_INPUT_FILE) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df = df[df["asset_id"].isin(_SELECTED_ASSETS)]
    print(f"Filtered to {len(df)} docs for {len(_SELECTED_ASSETS)} assets")

    # Aggregate per (asset_id, cycle_index, cycle_type) first — one row per
    # sub-cycle with the right aggregation for each column type.
    sub_cycle_rows = []
    for (asset_id, cycle_idx, cycle_type), group in df.groupby(
        ["asset_id", "cycle_index", "cycle_type"]
    ):
        row = {
            "asset_id": asset_id,
            "cycle_index": int(cycle_idx),
            "cycle_type": cycle_type,
            "ambient_temperature": group["ambient_temperature"].iloc[0],
        }

        if cycle_type == "discharge":
            if "Capacity" in group.columns:
                row["Capacity"] = group["Capacity"].dropna().iloc[-1] if group["Capacity"].notna().any() else None
            row["discharge_max_Temperature"] = group["Temperature_measured"].max() if "Temperature_measured" in group.columns else None
            row["discharge_avg_Voltage"] = group["Voltage_measured"].mean() if "Voltage_measured" in group.columns else None
            row["discharge_min_Voltage"] = group["Voltage_measured"].min() if "Voltage_measured" in group.columns else None
            row["discharge_avg_Current"] = group["Current_measured"].mean() if "Current_measured" in group.columns else None
            row["discharge_duration_min"] = (
                (pd.to_datetime(group["timestamp"]).max() - pd.to_datetime(group["timestamp"]).min()).total_seconds() / 60
            ) if len(group) > 1 else None

        elif cycle_type == "charge":
            row["charge_max_Temperature"] = group["Temperature_measured"].max() if "Temperature_measured" in group.columns else None
            row["charge_avg_Voltage"] = group["Voltage_measured"].mean() if "Voltage_measured" in group.columns else None
            row["charge_max_Voltage"] = group["Voltage_measured"].max() if "Voltage_measured" in group.columns else None
            row["charge_avg_Current"] = group["Current_charge"].mean() if "Current_charge" in group.columns else None
            row["charge_duration_min"] = (
                (pd.to_datetime(group["timestamp"]).max() - pd.to_datetime(group["timestamp"]).min()).total_seconds() / 60
            ) if len(group) > 1 else None

        elif cycle_type == "impedance":
            row["Re"] = group["Re"].dropna().iloc[0] if ("Re" in group.columns and group["Re"].notna().any()) else None
            row["Rct"] = group["Rct"].dropna().iloc[0] if ("Rct" in group.columns and group["Rct"].notna().any()) else None

        sub_cycle_rows.append(row)

    sub_df = pd.DataFrame(sub_cycle_rows)

    # Now merge sub-cycles into one row per (asset_id, cycle_index) by taking
    # the first non-null value of each column across cycle types.
    merged_rows = []
    for (asset_id, cycle_idx), group in sub_df.groupby(["asset_id", "cycle_index"]):
        row = {"asset_id": asset_id, "cycle_index": int(cycle_idx)}
        for col in sub_df.columns:
            if col in ("asset_id", "cycle_index", "cycle_type"):
                continue
            non_null = group[col].dropna()
            row[col] = non_null.iloc[0] if len(non_null) > 0 else None
        # Record which types were present for this cycle
        row["cycle_types_present"] = ",".join(sorted(group["cycle_type"].unique()))
        merged_rows.append(row)

    result = pd.DataFrame(merged_rows)
    result = result.sort_values(["asset_id", "cycle_index"]).reset_index(drop=True)

    # Forward-fill impedance (Re, Rct) within each asset — these are only
    # measured every N cycles, but the underlying state evolves continuously.
    for asset_id in result["asset_id"].unique():
        mask = result["asset_id"] == asset_id
        for col in ("Re", "Rct"):
            if col in result.columns:
                result.loc[mask, col] = result.loc[mask, col].ffill()

    # Drop rows with no Capacity (cycles that only had impedance/charge measurements)
    before = len(result)
    result = result.dropna(subset=["Capacity"]).reset_index(drop=True)
    dropped = before - len(result)
    if dropped:
        print(f"Dropped {dropped} rows with no Capacity measurement")

    # Assign evenly-spaced timestamps. Each asset starts in a different year
    # so timestamps are unique globally (TSFM with id_columns treats each
    # asset as a separate series, but unique timestamps avoid any ambiguity).
    interval = pd.Timedelta(hours=4)
    timestamps = []
    for i, asset_id in enumerate(result["asset_id"].unique()):
        mask = result["asset_id"] == asset_id
        n = mask.sum()
        base = pd.Timestamp(f"{2000 + i}-01-01")  # unique year per asset
        ts = pd.date_range(start=base, periods=n, freq=interval)
        timestamps.extend(ts.strftime("%Y-%m-%dT%H:%M:%S").tolist())
    result["timestamp"] = timestamps

    # Reorder columns: timestamp, asset_id, cycle_index, target, then others
    priority = ["timestamp", "asset_id", "cycle_index", "Capacity"]
    other_cols = [c for c in result.columns if c not in priority]
    result = result[priority + other_cols]

    os.makedirs(os.path.dirname(_OUTPUT_FILE), exist_ok=True)
    result.to_csv(_OUTPUT_FILE, index=False)
    print(f"\nWrote {len(result)} rows to {os.path.abspath(_OUTPUT_FILE)}")
    print(f"Columns ({len(result.columns)}): {list(result.columns)}")
    print()
    print("Per-asset summary:")
    for asset in _SELECTED_ASSETS:
        n = (result["asset_id"] == asset).sum()
        print(f"  {asset}: {n} cycles")


if __name__ == "__main__":
    main()

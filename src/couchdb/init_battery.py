"""Initialize the CouchDB battery database from raw NASA B0xx JSON files.

Pattern copied from `init_asset_data.py` (bulk-post in batches of 500). The only
adaptation is NASA-specific document transformation:
    raw[cell_id]["cycle"][i] -> one CouchDB document per cycle

One document per cycle, preserving the full within-cycle data arrays so that
preprocessing can interpolate V/I/T curves.

Usage:
    python -m couchdb.init_battery [--data-dir <path>] [--db <name>] [--drop]

Environment variables (or .env):
    COUCHDB_URL               e.g. http://localhost:5984
    COUCHDB_USERNAME          admin user
    COUCHDB_PASSWORD          admin password
    BATTERY_DBNAME            target database (default: battery)
    BATTERY_DATA_DIR          directory containing B*.json files (flat)
    BATTERY_CELL_SUBSET       comma-separated cell IDs, or "all" (default: 14-cell prototyping subset)
"""

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Resolve the default data dir relative to the repo root (two levels up from
# this script), NOT the current working directory. This way you can run the
# script from anywhere: `python -m couchdb.init_battery` or directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # src/couchdb/ -> src/ -> repo root
_DEFAULT_DATA_DIR = os.environ.get(
    "BATTERY_DATA_DIR", str(_REPO_ROOT / "external" / "battery" / "nasa")
)

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
BATTERY_DBNAME = os.environ.get("BATTERY_DBNAME", "battery")

_AUTH = (COUCHDB_USERNAME, COUCHDB_PASSWORD)

_DEFAULT_SUBSET = (
    "B0005,B0006,B0007,B0018,"
    "B0025,B0029,B0030,"
    "B0033,B0034,B0036,B0038,"
    "B0054,B0055,B0056"
)

# Index that covers the common query patterns: by asset_id, by cycle_type+asset_id, by cycle_index
_INDEXES = [
    ["asset_id", "cycle_type", "cycle_index"],
]


# ---------------------------------------------------------------------------
# Helpers (copied from init_asset_data.py)
# ---------------------------------------------------------------------------


def _db_url(db: str, *parts: str) -> str:
    return "/".join([COUCHDB_URL.rstrip("/"), db] + list(parts))


def _ensure_db(db_name: str, drop: bool) -> bool:
    """Return True if the database was freshly created, False if it already existed."""
    url = _db_url(db_name)
    resp = requests.head(url, auth=_AUTH, timeout=10)
    if resp.status_code == 200:
        if drop:
            logger.info("Dropping existing database '%s'…", db_name)
            requests.delete(url, auth=_AUTH, timeout=10).raise_for_status()
        else:
            logger.info("Database '%s' already exists — skipping.", db_name)
            return False
    logger.info("Creating database '%s'…", db_name)
    requests.put(url, auth=_AUTH, timeout=10).raise_for_status()
    return True


def _create_indexes(db_name: str) -> None:
    url = _db_url(db_name, "_index")
    for fields in _INDEXES:
        payload = {"index": {"fields": fields}, "type": "json"}
        resp = requests.post(url, json=payload, auth=_AUTH, timeout=10)
        resp.raise_for_status()
        logger.info("Index on %s: %s", fields, resp.json().get("result", "?"))


def _bulk_insert(db_name: str, docs: list, batch_size: int = 500) -> None:
    url = _db_url(db_name, "_bulk_docs")
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        resp = requests.post(url, json={"docs": batch}, auth=_AUTH, timeout=120)
        resp.raise_for_status()
        errors = [r for r in resp.json() if r.get("error")]
        if errors:
            logger.warning("%d bulk-insert errors in batch %d", len(errors), i // batch_size)
        logger.info(
            "Inserted batch %d/%d (%d docs)",
            i // batch_size + 1,
            math.ceil(total / batch_size),
            len(batch),
        )


# ---------------------------------------------------------------------------
# NASA-specific transformations
# ---------------------------------------------------------------------------


def _matlab_time_to_iso(time_vec) -> str:
    """[Y, M, D, h, m, s(float)] -> 'YYYY-MM-DDTHH:MM:SS' ISO 8601."""
    if not time_vec or len(time_vec) < 6:
        return ""
    y, mo, d, h, mi, s = time_vec
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(h):02d}:{int(mi):02d}:{int(s):02d}"


def nasa_cycle_to_docs(cell_id: str, raw: dict) -> list[dict]:
    """Convert one NASA cell's raw cycles into a list of CouchDB documents."""
    docs = []
    cycles = raw.get(cell_id, {}).get("cycle", [])
    for i, c in enumerate(cycles):
        cycle_type = c.get("type")
        data = c.get("data", {})

        # Quality filter at ingest: drop discharge cycles with too few samples
        # (NASA README notes "several discharge runs where the capacity was very low").
        if cycle_type == "discharge":
            if len(data.get("Voltage_measured", [])) < 100:
                continue

        docs.append(
            {
                "asset_id": cell_id,
                "cycle_index": i,
                "cycle_type": cycle_type,
                "ambient_temperature": c.get("ambient_temperature"),
                "timestamp": _matlab_time_to_iso(c.get("time", [])),
                "data": data,
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize CouchDB battery database from NASA B0xx JSON files."
    )
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="Directory with B*.json")
    parser.add_argument("--db", default=BATTERY_DBNAME, help="CouchDB database name")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate if exists")
    args = parser.parse_args()

    # Resolve relative data-dir paths against the repo root (the "external/..." default
    # in .env is relative, but users may invoke this script from any directory).
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (_REPO_ROOT / data_dir).resolve()

    logger.info("CouchDB URL: %s", COUCHDB_URL)
    logger.info("Database: %s", args.db)
    logger.info("Data dir: %s", data_dir)

    if not data_dir.is_dir():
        logger.error("Data dir not found: %s", data_dir)
        sys.exit(1)
    args.data_dir = str(data_dir)

    subset_env = os.environ.get("BATTERY_CELL_SUBSET", _DEFAULT_SUBSET)
    subset = None if subset_env.strip().lower() == "all" else set(
        s.strip() for s in subset_env.split(",") if s.strip()
    )
    if subset is not None:
        logger.info("Loading subset: %s", sorted(subset))
    else:
        logger.info("Loading ALL cells (no subset filter)")

    all_docs: list[dict] = []
    cells_loaded: list[str] = []
    for cell_file in sorted(glob.glob(os.path.join(args.data_dir, "B*.json"))):
        cell_id = Path(cell_file).stem
        if subset is not None and cell_id not in subset:
            continue
        with open(cell_file) as f:
            raw = json.load(f)
        docs = nasa_cycle_to_docs(cell_id, raw)
        all_docs.extend(docs)
        cells_loaded.append(cell_id)
        logger.info("Parsed %s: %d cycles", cell_id, len(docs))

    if not all_docs:
        logger.error("No documents to insert (check data dir and subset filter)")
        sys.exit(1)

    logger.info("Total %d documents from %d cells", len(all_docs), len(cells_loaded))

    _ensure_db(args.db, drop=args.drop)
    _bulk_insert(args.db, all_docs)
    _create_indexes(args.db)
    logger.info("Done. Database '%s' has %d cells loaded.", args.db, len(cells_loaded))


if __name__ == "__main__":
    main()

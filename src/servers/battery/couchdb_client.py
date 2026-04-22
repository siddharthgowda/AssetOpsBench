"""CouchDB client for the battery server.

Mirrors `src/servers/vibration/couchdb_client.py`: lazy connection, graceful
degradation when COUCHDB_URL is unset or the service is down.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import couchdb3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("battery-mcp-server")

COUCHDB_URL = os.environ.get("COUCHDB_URL")
BATTERY_DBNAME = os.environ.get("BATTERY_DBNAME", "battery")
COUCHDB_USER = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")


def _get_db() -> Optional[couchdb3.Database]:
    if not COUCHDB_URL:
        logger.warning("COUCHDB_URL not set — battery data unavailable")
        return None
    try:
        return couchdb3.Database(
            BATTERY_DBNAME,
            url=COUCHDB_URL,
            user=COUCHDB_USER,
            password=COUCHDB_PASSWORD,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("CouchDB connection failed: %s", e)
        return None


class CouchDBClient:
    """Thin wrapper around couchdb3 for fetching battery cycle documents."""

    def __init__(self):
        self._db = _get_db()

    @property
    def available(self) -> bool:
        return self._db is not None

    def list_cell_ids(self) -> list[str]:
        if self._db is None:
            return []
        try:
            res = self._db.find(
                {"asset_id": {"$exists": True}},
                fields=["asset_id"],
                limit=100000,
            )
            return sorted({d["asset_id"] for d in res.get("docs", []) if "asset_id" in d})
        except Exception as e:  # noqa: BLE001
            logger.error("list_cell_ids failed: %s", e)
            return []

    def fetch_cycles(
        self,
        asset_id: str,
        cycle_type: Optional[str] = None,
        limit: int = 10000,
    ) -> list[dict]:
        if self._db is None:
            return []
        selector: dict = {"asset_id": asset_id}
        if cycle_type:
            selector["cycle_type"] = cycle_type
        try:
            res = self._db.find(
                selector,
                limit=limit,
                sort=[
                    {"asset_id": "asc"},
                    {"cycle_type": "asc"},
                    {"cycle_index": "asc"},
                ],
            )
            docs = res.get("docs", [])
            # Sort numerically by cycle_index in case the Mango index order is insufficient
            return sorted(docs, key=lambda d: d.get("cycle_index", 0))
        except Exception as e:  # noqa: BLE001
            logger.error("fetch_cycles(%s, %s) failed: %s", asset_id, cycle_type, e)
            return []

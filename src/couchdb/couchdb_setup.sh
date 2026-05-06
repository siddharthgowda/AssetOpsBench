#!/bin/sh -xe

cat >/opt/couchdb/etc/local.ini <<EOF
[couchdb]
single_node=true

[admins]
${COUCHDB_USERNAME} = ${COUCHDB_PASSWORD}
EOF

echo "Starting CouchDB..."
/opt/couchdb/bin/couchdb &

echo "Waiting for CouchDB to be ready..."
until curl -sf -u "${COUCHDB_USERNAME}:${COUCHDB_PASSWORD}" http://localhost:5984/ >/dev/null; do
  sleep 2
done
echo "CouchDB is ready."

echo "Installing Python dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip
pip3 install -q --break-system-packages requests pandas python-dotenv

echo "Loading IoT asset data (sample of chiller dataset)..."
COUCHDB_URL="http://localhost:5984" \
  python3 /couchdb/init_asset_data.py \
    --data-file /sample_data/iot/chiller6_june2020_sensordata_couchdb.json \
    --db "${IOT_DBNAME:-chiller}" \
    --drop

echo "Loading work order data..."
COUCHDB_URL="http://localhost:5984" \
  python3 /couchdb/init_wo.py \
    --data-dir /sample_data/work_order \
    --db "${WO_DBNAME:-workorder}" \
    --drop

# Load vibration sample data (Motor_01 bearing fault) into a dedicated database
VIBRATION_FILE="/sample_data/iot/bulk_docs_vibration.json"
if [ -f "$VIBRATION_FILE" ]; then
  echo "Loading vibration data..."
  COUCHDB_URL="http://localhost:5984" \
    python3 /couchdb/init_asset_data.py \
      --data-file "$VIBRATION_FILE" \
      --db "${VIBRATION_DBNAME:-vibration}" \
      --drop
else
  echo "⚠️ $VIBRATION_FILE not found, skipping vibration data."
fi

# Load battery cycling data (NASA Li-ion B0xx cells) into a dedicated database.
# Uses init_battery.py which reads flat directory of B*.json files and applies
# subset filter via BATTERY_CELL_SUBSET env var (default: 14 cells for prototyping).
BATTERY_DATA_DIR_CONTAINER="/sample_data/battery"
if [ -d "$BATTERY_DATA_DIR_CONTAINER" ]; then
  echo "Loading battery NASA cycling data..."
  COUCHDB_URL="http://localhost:5984" \
    BATTERY_DATA_DIR="$BATTERY_DATA_DIR_CONTAINER" \
    python3 /couchdb/init_battery.py \
      --data-dir "$BATTERY_DATA_DIR_CONTAINER" \
      --db "${BATTERY_DBNAME:-battery}" \
      --drop
else
  echo "⚠️ $BATTERY_DATA_DIR_CONTAINER not found, skipping battery data."
fi

echo "✅ All databases initialised."
tail -f /dev/null

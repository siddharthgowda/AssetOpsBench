#!/usr/bin/env bash
# Verifies that all required battery artifacts are present on disk. Does NOT
# download — Google Drive requires manual OAuth.
#
# Layout convention (also referenced by BATTERY_MODEL_WEIGHTS_DIR / BATTERY_NORMS_DIR / BATTERY_DATA_DIR):
#   src/servers/battery/artifacts/weights/*.h5   (4 files, from acctouhou "pretrained/" zip)
#   src/servers/battery/artifacts/norms/*.npy    (4 files, from acctouhou "dataset/" zip)
#   external/battery/nasa/B*.json                (NASA Prognostics Center raw data, flattened)

set -e
ARTIFACTS_DIR=${ARTIFACTS_DIR:-src/servers/battery/artifacts}
NASA_DIR=${NASA_DIR:-external/battery/nasa}
REQUIRED_WEIGHTS=(feature_selector_ch.h5 feature_selector_dis.h5 predictor.h5 predictor2.h5)
REQUIRED_NORMS=(charge_norm.npy discharge_norm.npy summary_norm.npy predict_renorm.npy)

mkdir -p "$ARTIFACTS_DIR/weights" "$ARTIFACTS_DIR/norms" "$NASA_DIR"

missing=0
for f in "${REQUIRED_WEIGHTS[@]}"; do
  [ -f "$ARTIFACTS_DIR/weights/$f" ] || { echo "missing weight: $f"; missing=1; }
done
for f in "${REQUIRED_NORMS[@]}"; do
  [ -f "$ARTIFACTS_DIR/norms/$f" ] || { echo "missing norm: $f"; missing=1; }
done
n_nasa=$(ls "$NASA_DIR"/B*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_nasa" -lt 10 ]; then
  echo "only $n_nasa NASA cells found (need at least 10 for the usable subset)"
  missing=1
fi

if [ $missing -eq 1 ]; then
  cat <<EOF

Some artifacts are missing. To provision:

1. Acctouhou weights + norms (Apache 2.0):
   Download the Google Drive zip linked in the acctouhou/Prediction_of_battery README.
   The zip contains "pretrained/" (the 4 .h5 files) and "dataset/" (the 4 required .npy files + others we don't need).
   Move the 4 .h5 files to:    $ARTIFACTS_DIR/weights/
   Move only these 4 .npy to:  $ARTIFACTS_DIR/norms/
     charge_norm.npy, discharge_norm.npy, summary_norm.npy, predict_renorm.npy

2. NASA battery cycling data (US Public Domain):
   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
   Extract and flatten all B*.json into:  $NASA_DIR/

EOF
  exit 1
fi

echo "all battery artifacts present: ${#REQUIRED_WEIGHTS[@]} weights, ${#REQUIRED_NORMS[@]} norms, $n_nasa NASA cells"

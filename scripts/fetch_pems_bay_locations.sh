#!/usr/bin/env bash
# Sensor metadata (id, latitude, longitude) for PeMS-BAY, needed for the GIS layer.
# Not included in the hazdzz/dcrnn_data mirror used by fetch_pems_bay.sh, so pulled
# separately from a Hugging Face dataset mirror (verified: same 325 sensor_ids).
set -euo pipefail

DEST="data/raw/pems_bay/sensor_locations.csv"
URL="https://huggingface.co/datasets/witgaw/PEMS-BAY/resolve/main/sensor_graph/sensor_locations.csv"

curl -sL -o "$DEST" "$URL"
echo "Sensor locations saved to $DEST"

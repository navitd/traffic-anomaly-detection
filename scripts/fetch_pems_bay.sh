#!/usr/bin/env bash
# Fallback dataset fetch: PeMS-BAY, mirrored from the DCRNN paper's data release.
# Use while waiting on raw Caltrans PeMS account approval (see memory: reference-pems-dataset).
set -euo pipefail

DEST="data/raw/pems_bay"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 https://github.com/hazdzz/dcrnn_data "$TMP/dcrnn_data"
mkdir -p "$DEST"
cp -r "$TMP/dcrnn_data/pems_bay/." "$DEST/"

echo "PeMS-BAY data copied to $DEST"

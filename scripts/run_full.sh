#!/usr/bin/env bash
# Full SSL-VTC pipeline on a config. Usage: scripts/run_full.sh [config]
# Default config: configs/gulf2019.yaml (US Gulf, full Jan-Jun 2019).
set -euo pipefail

CONFIG="${1:-configs/gulf2019.yaml}"
PY="${PYTHON:-.venv/bin/python}"
SSLVTC="$PY -m sslvtc.cli -c $CONFIG"

echo ">>> config: $CONFIG"
echo ">>> [1/5] download  (~290 MB/day zipped; full Jan-Jun = ~52 GB)"
$SSLVTC download
echo ">>> [2/5] ingest    (chunked CSV -> filtered/labeled parquet)"
$SSLVTC ingest
echo ">>> [3/5] extract   (5-step trajectory extraction -> tensors)"
$SSLVTC extract
echo ">>> [4/5] experiments (Tables 2/3/4/6/7 + Fig 7 + confusion matrices)"
$SSLVTC experiment all
echo ">>> [5/5] done. results in \$(paths.results) from $CONFIG"

#!/usr/bin/env bash
# Full Danish AIS pipeline: ingest -> extract (complete+inclusive) -> tag -> consequence.
# Raw zips already at: /mnt/storage_1_10T/zezzahed/AIS_Data/external/danishais_raw/
# Run from repo root:  bash scripts/run_danish_chain.sh
set -euo pipefail

PY=".venv/bin/python"
PYTHONPATH=.
export PYTHONPATH

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Danish AIS Consequence Pipeline              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Ingest ──────────────────────────────────────────
echo "[1/5] Ingesting Danish AIS monthly zips -> interim parquets..."
echo "      (text Ship type, multi-CSV zips, DD/MM/YYYY timestamps)"
$PY scripts/ingest_danish.py configs/danishais2019.yaml
echo "      Done."
echo ""

# ── Step 2: Extract (static-complete) ───────────────────────
echo "[2/5] Extracting trajectories — static-COMPLETE cohort..."
echo "      (repartition by MMSI + 5-step extraction, require_complete_static=true)"
$PY -m sslvtc.cli -c configs/danishais2019.yaml extract
echo "      Done."
echo ""

# ── Step 3: Extract (static-inclusive) ──────────────────────
echo "[3/5] Extracting trajectories — static-INCLUSIVE cohort..."
echo "      (reuses interim shards, require_complete_static=false)"
$PY -m sslvtc.cli -c configs/danishais2019_inclusive.yaml extract
echo "      Done."
echo ""

# ── Step 4: Tag static_complete on inclusive index ───────────
echo "[4/5] Tagging static_complete flag on inclusive index..."
INCLUSIVE_PROCESSED="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019_inclusive/processed"
$PY scripts/tag_static_complete.py "$INCLUSIVE_PROCESSED"
echo "      Done."
echo ""

# ── Step 5: Consequence experiment ──────────────────────────
echo "[5/5] Running consequence experiment (3 seeds × Regime A + B)..."
echo "      Estimated: ~3h on GPU (same scale as US experiment)"
$PY scripts/consequence_danish.py
echo "      Done."
echo ""

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Pipeline complete.                                  ║"
echo "║  Results: /mnt/storage_1_10T/zezzahed/AIS_Data/     ║"
echo "║           danishais2019/results/consequence_danish.csv ║"
echo "╚══════════════════════════════════════════════════════╝"

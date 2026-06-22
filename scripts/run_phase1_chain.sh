#!/usr/bin/env bash
# Phase 1 leakage-study chain — runs automatically after table4 finishes.
#
# Order (cheapest/most-decisive first):
#   1. vessel-split        — rebuild split_vd on the re-extracted index (fast)
#   2. table3-leakage      — DECISION GATE: static ablation, temporal vs vessel-disjoint
#   3. leakage-inflation   — EXPENSIVE (16 runs incl SSL); gated behind RUN_INFLATION=1
#
# Usage:
#   CONFIG=configs/fullus2019.yaml GPU=1 WAIT_PID=<table4_pid> RUN_INFLATION=0 \
#     nohup bash scripts/run_phase1_chain.sh > /tmp/phase1_chain.log 2>&1 &
#
# Env knobs (all optional):
#   CONFIG          config yaml         (default configs/fullus2019.yaml)
#   GPU             CUDA device to pin   (default 1)
#   WAIT_PID        pid to wait for before starting (default: none)
#   RUN_INFLATION   1 to run leakage-inflation, 0 to skip (default 0)

set -u
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/fullus2019.yaml}"
GPU="${GPU:-1}"
WAIT_PID="${WAIT_PID:-}"
RUN_INFLATION="${RUN_INFLATION:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "=================================================="; echo "[$(ts)] $1"; echo "=================================================="; }

# --- 0. wait for the upstream job (table4) to finish ----------------------
if [ -n "$WAIT_PID" ]; then
  banner "WAITING for pid $WAIT_PID (table4) to finish before starting Phase 1"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(ts)] pid $WAIT_PID finished — proceeding."
fi

# --- 1. vessel-disjoint split + overlap report ----------------------------
banner "STEP 1/3  vessel-split (rebuild split_vd, report overlap)"
python -m src.sslvtc.cli -c "$CONFIG" vessel-split --report
rc=$?
echo "[$(ts)] vessel-split exit=$rc"
if [ $rc -ne 0 ]; then echo "[$(ts)] ABORT: vessel-split failed"; exit 1; fi

# --- 2. DECISION GATE: static ablation under both splits -------------------
banner "STEP 2/3  table3-leakage (DECISION GATE: temporal vs vessel-disjoint)"
python -m src.sslvtc.cli -c "$CONFIG" experiment table3-leakage
rc=$?
echo "[$(ts)] table3-leakage exit=$rc"
if [ $rc -ne 0 ]; then echo "[$(ts)] WARN: table3-leakage failed; continuing"; fi

banner "DECISION GATE RESULT (table3_leakage_comparison.csv)"
RESULTS_DIR="$(python -c "from src.sslvtc.config import load_config; print(load_config('$CONFIG').paths.results)")"
cat "$RESULTS_DIR/table3_leakage_comparison.csv" 2>/dev/null || echo "  (csv not found)"

# --- 3. inflation table (expensive; opt-in) -------------------------------
if [ "$RUN_INFLATION" = "1" ]; then
  banner "STEP 3/3  leakage-inflation (16 runs incl SSL — long)"
  python -m src.sslvtc.cli -c "$CONFIG" experiment leakage-inflation
  echo "[$(ts)] leakage-inflation exit=$?"
  cat "$RESULTS_DIR/leakage_inflation.csv" 2>/dev/null
else
  banner "STEP 3/3  leakage-inflation SKIPPED (set RUN_INFLATION=1 to enable)"
fi

banner "PHASE 1 CHAIN COMPLETE"
echo "[$(ts)] artifacts in: $RESULTS_DIR"

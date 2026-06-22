#!/usr/bin/env bash
# Master autonomous chain — runs the full leakage study end-to-end on the
# fast (RAM-packed) pipeline with the early-stop warmup fix and full metrics.
#
# Order = most-decisive-first:
#   1. vessel-split      — rebuild split_vd on the static-complete index (fast)
#   2. table3-leakage    — DECISION GATE: static ablation (acc + macro-F1 + fishing
#                          recall) temporal vs vessel-disjoint  [supervised, ~fast]
#   3. table4            — SSL vs baseline, all fractions (early-stop fix → 5% works)
#   4. leakage-inflation — Δ(temporal − vessel-disjoint) per method  [expensive]
#
# Usage: CONFIG=configs/fullus2019.yaml GPU=0 nohup bash scripts/run_master_chain.sh \
#          > /tmp/master_chain.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/fullus2019.yaml}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=.

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "============================================================"; echo "[$(ts)] $1"; echo "============================================================"; }
RES="$(python -c "from src.sslvtc.config import load_config; print(load_config('$CONFIG').paths.results)")"

banner "STEP 1/4  vessel-split (rebuild split_vd, report overlap)"
python -m src.sslvtc.cli -c "$CONFIG" vessel-split --report || { echo "ABORT: vessel-split failed"; exit 1; }

banner "STEP 2/4  table3-leakage  [DECISION GATE: acc + macro-F1 + fishing recall]"
python -m src.sslvtc.cli -c "$CONFIG" experiment table3-leakage
echo "[$(ts)] table3-leakage exit=$?"
banner "DECISION GATE RESULT"
cat "$RES/table3_leakage_comparison.csv" 2>/dev/null || cat "$RES/table3_static_ablation.csv" "$RES/table3_static_ablation_vd.csv" 2>/dev/null

banner "STEP 3/4  table4 (SSL vs baseline, all fractions, early-stop fix)"
python -m src.sslvtc.cli -c "$CONFIG" experiment table4
echo "[$(ts)] table4 exit=$?"
cat "$RES/table4_ssl_vs_baseline.csv" 2>/dev/null

banner "STEP 4/4  leakage-inflation (Δ temporal − vessel-disjoint; long)"
python -m src.sslvtc.cli -c "$CONFIG" experiment leakage-inflation
echo "[$(ts)] leakage-inflation exit=$?"
cat "$RES/leakage_inflation.csv" 2>/dev/null

banner "MASTER CHAIN COMPLETE — artifacts in $RES"

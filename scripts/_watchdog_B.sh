#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results"
CSV="$RESULTS/regime_B.csv"

seeds_done() {
    [[ -f "$CSV" ]] || { echo 0; return; }
    "$REPO/.venv/bin/python" -c "
import pandas as pd
df = pd.read_csv('$CSV')
print(df[df['regime']=='B_inclusive_all']['seed'].nunique())
" 2>/dev/null || echo 0
}

while true; do
    done=$(seeds_done)
    echo "[B $(date '+%H:%M:%S')] seeds done: $done/3"
    [[ "$done" -ge 3 ]] && echo "[B] Complete." && break
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
        "$REPO/scripts/_consequence_danish_B.py" || true
    echo "[B] Process exited. Restarting in 10s..."
    sleep 10
done

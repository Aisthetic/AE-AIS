#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results"
CSV="$RESULTS/regime_A.csv"

seeds_done() {
    [[ -f "$CSV" ]] || { echo 0; return; }
    "$REPO/.venv/bin/python" -c "
import pandas as pd
df = pd.read_csv('$CSV')
print(df[df['regime']=='A_complete']['seed'].nunique())
" 2>/dev/null || echo 0
}

while true; do
    done=$(seeds_done)
    echo "[A $(date '+%H:%M:%S')] seeds done: $done/3"
    [[ "$done" -ge 3 ]] && echo "[A] Complete." && break
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
        "$REPO/scripts/_consequence_danish_A.py" || true
    echo "[A] Process exited. Restarting in 10s..."
    sleep 10
done

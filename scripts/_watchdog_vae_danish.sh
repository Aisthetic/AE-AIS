#!/usr/bin/env bash
# Self-renewing watchdog for vae_collapse_danish.py — GPU 0
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results"
CSV="$RESULTS/vae_collapse_danish.csv"

seeds_done() {
    [[ -f "$CSV" ]] || { echo 0; return; }
    "$REPO/.venv/bin/python" -c "
import pandas as pd
df = pd.read_csv('$CSV')
print(df[df['population']=='all']['seed'].nunique())
" 2>/dev/null || echo 0
}

while true; do
    done=$(seeds_done)
    echo "[vae $(date '+%H:%M:%S')] seeds done: $done/3"
    [[ "$done" -ge 3 ]] && echo "[vae] Complete." && break
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
        "$REPO/scripts/vae_collapse_danish.py" || true
    echo "[vae] Process exited. Restarting in 10s..."
    sleep 10
done

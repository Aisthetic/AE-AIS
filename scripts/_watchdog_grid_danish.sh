#!/usr/bin/env bash
# Self-renewing watchdog for grid_perclass_danish.py
# Done when all 3 models × 3 seeds have "all" population rows = 9 "all" rows total
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results"
CSV="$RESULTS/grid_perclass_danish.csv"

combos_done() {
    [[ -f "$CSV" ]] || { echo 0; return; }
    "$REPO/.venv/bin/python" -c "
import pandas as pd
df = pd.read_csv('$CSV')
print(len(df[df['population']=='all']))
" 2>/dev/null || echo 0
}

while true; do
    done=$(combos_done)
    echo "[grid $(date '+%H:%M:%S')] model×seed combos done: $done/9"
    [[ "$done" -ge 9 ]] && echo "[grid] Complete." && break
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
        "$REPO/scripts/grid_perclass_danish.py" || true
    echo "[grid] Process exited. Restarting in 10s..."
    sleep 10
done

#!/usr/bin/env bash
# Self-renewing Danish consequence experiment.
# Launches detached watchdog processes via nohup — survive SSH disconnects and session deaths.
# Usage:
#   bash scripts/run_danish_experiment.sh          # launch
#   bash scripts/run_danish_experiment.sh status   # check progress
#   bash scripts/run_danish_experiment.sh stop     # kill watchdogs

RESULTS="/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_A="$RESULTS/watchdog_A.log"
LOG_B="$RESULTS/watchdog_B.log"
PID_A="$RESULTS/watchdog_A.pid"
PID_B="$RESULTS/watchdog_B.pid"

seeds_done() {
    local csv="$1" regime="$2"
    [[ -f "$csv" ]] || { echo 0; return; }
    python3 -c "
import pandas as pd
df = pd.read_csv('$csv')
print(df[df['regime']=='$regime']['seed'].nunique())
" 2>/dev/null || echo 0
}

_watchdog_A() {
    local csv="$RESULTS/regime_A.csv"
    while true; do
        done=$(seeds_done "$csv" "A_complete")
        echo "[A $(date '+%H:%M:%S')] seeds done: $done/3"
        if [[ "$done" -ge 3 ]]; then
            echo "[A] Complete."; break
        fi
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
            "$REPO/scripts/_consequence_danish_A.py" || true
        echo "[A] Process exited. Restarting in 10s..."
        sleep 10
    done
}

_watchdog_B() {
    local csv="$RESULTS/regime_B.csv"
    while true; do
        done=$(seeds_done "$csv" "B_inclusive_all")
        echo "[B $(date '+%H:%M:%S')] seeds done: $done/3"
        if [[ "$done" -ge 3 ]]; then
            echo "[B] Complete."; break
        fi
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$REPO/.venv/bin/python" \
            "$REPO/scripts/_consequence_danish_B.py" || true
        echo "[B] Process exited. Restarting in 10s..."
        sleep 10
    done
}

_merge() {
    echo "[merge] Both done. Merging..."
    PYTHONPATH="$REPO" "$REPO/.venv/bin/python" "$REPO/scripts/_merge_danish_results.py"
}

status() {
    echo "=== Watchdog A ==="
    [[ -f "$PID_A" ]] && kill -0 "$(cat $PID_A)" 2>/dev/null \
        && echo "  running (pid=$(cat $PID_A))" || echo "  NOT running"
    echo "  seeds done: $(seeds_done "$RESULTS/regime_A.csv" "A_complete")/3"
    [[ -f "$LOG_A" ]] && tail -3 "$LOG_A" | sed 's/^/  /'

    echo "=== Watchdog B ==="
    [[ -f "$PID_B" ]] && kill -0 "$(cat $PID_B)" 2>/dev/null \
        && echo "  running (pid=$(cat $PID_B))" || echo "  NOT running"
    echo "  seeds done: $(seeds_done "$RESULTS/regime_B.csv" "B_inclusive_all")/3"
    [[ -f "$LOG_B" ]] && tail -3 "$LOG_B" | sed 's/^/  /'
}

stop() {
    for pid_file in "$PID_A" "$PID_B"; do
        [[ -f "$pid_file" ]] && kill "$(cat $pid_file)" 2>/dev/null && echo "Killed $(cat $pid_file)"
    done
    pkill -f "_consequence_danish" 2>/dev/null || true
    echo "Stopped."
}

launch() {
    mkdir -p "$RESULTS"

    # kill stale watchdogs if any
    for pid_file in "$PID_A" "$PID_B"; do
        [[ -f "$pid_file" ]] && kill "$(cat $pid_file)" 2>/dev/null || true
    done
    pkill -f "_consequence_danish" 2>/dev/null || true
    sleep 2

    # launch A detached
    nohup bash "$REPO/scripts/_watchdog_A.sh" >> "$LOG_A" 2>&1 &
    echo $! > "$PID_A"
    echo "Watchdog A launched (pid=$(cat $PID_A)) -> $LOG_A"

    # launch B detached
    nohup bash "$REPO/scripts/_watchdog_B.sh" >> "$LOG_B" 2>&1 &
    echo $! > "$PID_B"
    echo "Watchdog B launched (pid=$(cat $PID_B)) -> $LOG_B"

    echo ""
    echo "Both running detached. Monitor:"
    echo "  bash scripts/run_danish_experiment.sh status"
    echo "  tail -f $LOG_A"
    echo "  tail -f $LOG_B"
    echo ""
    echo "Waiting for completion (safe to Ctrl+C — watchdogs keep running)..."

    # wait loop — if this session dies, watchdogs keep going
    while true; do
        a_done=$(seeds_done "$RESULTS/regime_A.csv" "A_complete")
        b_done=$(seeds_done "$RESULTS/regime_B.csv" "B_inclusive_all")
        echo "$(date '+%H:%M:%S') A=$a_done/3  B=$b_done/3"
        if [[ "$a_done" -ge 3 && "$b_done" -ge 3 ]]; then
            _merge; break
        fi
        sleep 120
    done
}

case "${1:-launch}" in
    status) status ;;
    stop)   stop ;;
    launch) launch ;;
    *)      echo "Usage: $0 [launch|status|stop]"; exit 1 ;;
esac

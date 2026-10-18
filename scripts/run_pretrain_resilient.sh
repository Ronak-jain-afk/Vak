#!/usr/bin/env bash
# Run pretraining with auto-resume so a transient GPU fault does not end a long
# unattended run. WSL2 GPU-PV can drop the CUDA context ("CUDA error: unknown
# error") and that kills the process; recovery needs a fresh process, and
# --resume continues from the last checkpoint.
# shortcut: fixed restart count and delay, no backoff; if it keeps dying, read the log.
set -u
cd "$(dirname "$0")/.."

LOG="${LOG:-logs/pretrain_55m.log}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"

trap 'echo "[resilient] stopped by user" | tee -a "$LOG"; exit 130' INT TERM

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  PYTHONPATH=. PYTHONUNBUFFERED=1 ./venv/bin/python scripts/train_pretrain.py --resume 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then
    echo "[resilient] training completed cleanly" | tee -a "$LOG"
    exit 0
  fi
  echo "[resilient] attempt $attempt exited with code $code; restarting in 20s" | tee -a "$LOG"
  sleep 20
done

echo "[resilient] gave up after $MAX_RESTARTS restarts; inspect $LOG" | tee -a "$LOG"
exit 1

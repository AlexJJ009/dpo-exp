#!/usr/bin/env bash
#
# Wait for the qwen25-3b-base tmux session to finish, then launch the
# Gemma3-4B-base resume pipeline in a new tmux session.
#
set -euo pipefail

WATCH_SESSION="qwen25-3b-base"
NEXT_SESSION="gemma3-4b-base"
NEXT_SCRIPT="/data-1/dpo-experiment/run_gemma3_4b_base_resume.sh"
NEXT_LOG="/data-1/dpo-experiment/run_gemma3_4b_base_resume.log"
QWEN_SUMMARY="/data-1/checkpoints/qwen25-3b-base-dpo/training_logs/training_summary.json"
POLL_INTERVAL=30

echo "[watcher] Waiting for tmux session '${WATCH_SESSION}' to finish..."
echo "[watcher] Will launch '${NEXT_SESSION}' when done."
echo ""

while tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; do
    sleep ${POLL_INTERVAL}
done

echo "[watcher] Session '${WATCH_SESSION}' ended at $(date)."

# Check if Qwen training succeeded via training summary
if [ -f "${QWEN_SUMMARY}" ]; then
    echo "[watcher] Qwen training summary found — proceeding."
else
    echo "[watcher] WARNING: ${QWEN_SUMMARY} not found. Qwen training may have failed."
    echo "[watcher] Launching Gemma3 pipeline anyway..."
fi

echo "[watcher] Starting tmux session '${NEXT_SESSION}'..."
tmux new-session -d -s "${NEXT_SESSION}"
tmux send-keys -t "${NEXT_SESSION}" \
    "bash ${NEXT_SCRIPT} 2>&1 | tee ${NEXT_LOG}" Enter

echo "[watcher] Session '${NEXT_SESSION}' launched. Log: ${NEXT_LOG}"
echo "[watcher] Done."

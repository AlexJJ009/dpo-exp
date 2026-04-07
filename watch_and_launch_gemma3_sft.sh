#!/usr/bin/env bash
#
# Wait for the gemma3-4b-base tmux session (B-2) to finish, then launch the
# Gemma3-4B-SFT resume pipeline (B-3) in a new tmux session.
#
set -euo pipefail

WATCH_SESSION="gemma3-4b-base"
NEXT_SESSION="gemma3-4b-sft"
NEXT_SCRIPT="/data-1/dpo-experiment/run_gemma3_4b_sft_resume.sh"
NEXT_LOG="/data-1/dpo-experiment/run_gemma3_4b_sft_resume.log"
B2_SUMMARY="/data-1/checkpoints/gemma3-4b-base-dpo/training_logs/training_summary.json"
POLL_INTERVAL=30

echo "[watcher] Waiting for tmux session '${WATCH_SESSION}' (B-2) to finish..."
echo "[watcher] Will launch '${NEXT_SESSION}' (B-3) when done."
echo ""

while tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; do
    sleep ${POLL_INTERVAL}
done

echo "[watcher] Session '${WATCH_SESSION}' ended at $(date)."

if [ -f "${B2_SUMMARY}" ]; then
    echo "[watcher] B-2 training summary found — B-2 completed successfully."
else
    echo "[watcher] WARNING: ${B2_SUMMARY} not found. B-2 may have failed."
    echo "[watcher] Launching B-3 anyway..."
fi

echo "[watcher] Starting tmux session '${NEXT_SESSION}'..."
tmux new-session -d -s "${NEXT_SESSION}"
tmux send-keys -t "${NEXT_SESSION}" \
    "bash ${NEXT_SCRIPT} 2>&1 | tee ${NEXT_LOG}" Enter

echo "[watcher] Session '${NEXT_SESSION}' launched. Log: ${NEXT_LOG}"
echo "[watcher] Done."

#!/usr/bin/env bash
#
# Queue runner with multi-layer safeguards.
#
# Safeguards between experiments:
#   1. Exit code     — pipeline script must return 0
#   2. Container cleanup — kill any leftover dpo-harness / verl-harness containers
#   3. GPU release   — wait until all GPUs are idle before starting next task
#   4. Log scan      — check for fatal patterns (OOM, NCCL, CUDA error)
#   5. Cooldown      — brief pause for system stabilization
#
# Idempotent pipelines mean re-running the queue safely skips finished steps.
#
# Usage:
#   nohup bash scripts/run_queue.sh &
#   bash scripts/run_queue.sh run_3b_med_sft_sci.sh run_4b_med_sft_sci.sh
#
# Queue log (lightweight): logs/queue/run_queue.log
# Each experiment log:     logs/<exp-slug>/<script>.log
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_DIR="${PROJECT_DIR}/experiments"
LOGS_DIR="${PROJECT_DIR}/logs"

# ==================== Configuration ====================
COOLDOWN_SECONDS="${QUEUE_COOLDOWN:-30}"        # 30s between experiments
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-300}"     # 5 min max wait for GPUs to free

# ---- Default queue (edit to add experiments) ----
# Order: small models first (faster), 8B last (needs most memory)
DEFAULT_QUEUE=(
  run_3b_med_sft_med.sh
  run_3b_sci_sft_sci.sh
  run_4b_med_sft_med.sh
  run_4b_sci_sft_sci.sh
  run_4b_code_sft_code.sh
  run_8b_med_sft_med.sh
  run_8b_sci_sft_sci.sh
)

if [ $# -gt 0 ]; then
  QUEUE=("$@")
else
  QUEUE=("${DEFAULT_QUEUE[@]}")
fi

TOTAL=${#QUEUE[@]}
PASSED=0
FAILED=0
FAILED_NAMES=()
QUEUE_LOG="${LOGS_DIR}/queue/run_queue.log"
mkdir -p "${LOGS_DIR}/queue"

# ==================== Helper functions ====================

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}"
  echo "${msg}" >> "${QUEUE_LOG}"
}

# Kill all running dpo-harness / verl-harness containers
cleanup_containers() {
  local containers
  containers=$(docker ps -q --filter "ancestor=dpo-harness" 2>/dev/null)
  containers="${containers} $(docker ps -q --filter "ancestor=verl-harness:latest" 2>/dev/null)"
  containers=$(echo "${containers}" | xargs)  # trim whitespace
  if [ -n "${containers}" ]; then
    log "CLEANUP: Killing leftover containers: ${containers}"
    docker kill ${containers} 2>/dev/null || true
    sleep 5
  fi
}

# Wait until no GPU has running compute processes
wait_gpus_idle() {
  local waited=0
  while [ ${waited} -lt ${GPU_WAIT_TIMEOUT} ]; do
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
    gpu_procs=${gpu_procs:-0}
    if [ "${gpu_procs}" -eq 0 ]; then
      if [ ${waited} -gt 0 ]; then
        log "GPUs idle after ${waited}s"
      fi
      return 0
    fi
    if [ ${waited} -eq 0 ]; then
      log "WAIT: ${gpu_procs} GPU process(es) still running, waiting for release..."
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "WARNING: GPUs not idle after ${GPU_WAIT_TIMEOUT}s, forcing container cleanup"
  cleanup_containers
  sleep 10
  return 0
}

# Scan log file for fatal error patterns
check_log_for_errors() {
  local logfile="$1"
  [ ! -f "${logfile}" ] && return 0

  local patterns=(
    "CUDA out of memory"
    "RuntimeError: CUDA"
    "NCCL error"
    "torch.cuda.OutOfMemoryError"
    "Segmentation fault"
    "Killed"
  )

  for pattern in "${patterns[@]}"; do
    if grep -qi "${pattern}" "${logfile}" 2>/dev/null; then
      log "LOG SCAN: Found '${pattern}' in ${logfile}"
      return 1
    fi
  done
  return 0
}

# ==================== Main loop ====================

log "============================================================"
log "  DPO Experiment Queue"
log "  Experiments: ${TOTAL}"
log "============================================================"
echo ""

for i in "${!QUEUE[@]}"; do
  script="${QUEUE[$i]}"
  idx=$((i + 1))
  script_path="${EXPERIMENTS_DIR}/${script}"
  # Log to logs/<exp-name>/ based on script name (e.g. run_3b_med_sft_sci -> 3b-med-sft-sci)
  exp_slug="$(echo "${script%.sh}" | sed 's/^run_//' | tr '_' '-')"
  mkdir -p "${LOGS_DIR}/${exp_slug}"
  logfile="${LOGS_DIR}/${exp_slug}/${script%.sh}.log"

  log "============================================================"
  log "  [${idx}/${TOTAL}] ${script}"
  log "============================================================"

  if [ ! -f "${script_path}" ]; then
    log "ERROR: Script not found: ${script_path}"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("${script}")
    continue
  fi

  # ---- Pre-launch: ensure GPU is clean ----
  cleanup_containers
  wait_gpus_idle

  # ---- Run ----
  log "Starting... (log: ${logfile})"
  bash "${script_path}" > "${logfile}" 2>&1
  exit_code=$?

  # ---- Post-run: log scan ----
  if [ ${exit_code} -eq 0 ]; then
    if ! check_log_for_errors "${logfile}"; then
      log "WARNING: Exit 0 but fatal errors in log — marking failed"
      exit_code=1
    fi
  fi

  if [ ${exit_code} -eq 0 ]; then
    log "[${idx}/${TOTAL}] PASSED: ${script}"
    PASSED=$((PASSED + 1))
  else
    log "[${idx}/${TOTAL}] FAILED: ${script} (exit ${exit_code})"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("${script}")
  fi

  # ---- Post-run: cleanup + cooldown ----
  cleanup_containers
  if [ ${idx} -lt ${TOTAL} ]; then
    log "Cooldown ${COOLDOWN_SECONDS}s..."
    sleep ${COOLDOWN_SECONDS}
    wait_gpus_idle
  fi

  echo ""
done

# ==================== Summary ====================
log "============================================================"
log "  Queue Complete"
log "  Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}"
if [ ${FAILED} -gt 0 ]; then
  log "  Failed:"
  for name in "${FAILED_NAMES[@]}"; do
    log "    - ${name}"
  done
fi
log "============================================================"

exit ${FAILED}

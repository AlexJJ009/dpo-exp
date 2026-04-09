#!/usr/bin/env bash
#
# Central path configuration. All scripts source this file.
#
# Auto-detects BASE_DIR from the repo location:
#   BASE_DIR = parent of the repo directory
#
# On this machine:  BASE_DIR=/data-1,       REPO=dpo-experiment
# On platform:      BASE_DIR=.../lgx,       REPO=dpo-exp
#
# All paths are overridable via environment variables.
#
# Usage (from any script):
#   source "$(dirname "$0")/../scripts/config.sh"   # from experiments/
#   source "$(dirname "$0")/config.sh"               # from scripts/

# ---- Auto-detect paths ----
_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${_CONFIG_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

# ---- Overridable base paths ----
export DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset}"
export CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/.cache}"
export CHECKPOINT_BASE="${CHECKPOINT_BASE:-${BASE_DIR}/checkpoints}"
export DPO_WORK_DIR="${DPO_WORK_DIR:-${DATASET_DIR}/dpo}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-dpo-harness}"

# ---- Run mode: docker vs native ----
# Set USE_DOCKER=0 on platforms where code runs directly inside the container.
# Default: auto-detect (if docker is available and image exists, use docker)
if [ -z "${USE_DOCKER:-}" ]; then
  if command -v docker &>/dev/null && docker image inspect "${DOCKER_IMAGE}" &>/dev/null 2>&1; then
    export USE_DOCKER=1
  else
    export USE_DOCKER=0
  fi
fi

# Helper: run a command either via docker or natively
# Usage: run_cmd python script.py --arg1 val1
run_cmd() {
  if [ "${USE_DOCKER}" = "1" ]; then
    docker run --rm --gpus all --ipc=host \
      -v "${BASE_DIR}:${BASE_DIR}" \
      -w "${REPO_DIR}" \
      "${DOCKER_IMAGE}" \
      "$@"
  else
    cd "${REPO_DIR}" && "$@"
  fi
}

# Helper: run with extra docker args (env vars etc)
# Usage: run_cmd_env "-e KEY=VAL -e KEY2=VAL2" python script.py
run_cmd_env() {
  local extra_args="$1"; shift
  if [ "${USE_DOCKER}" = "1" ]; then
    eval docker run --rm --gpus all --ipc=host \
      -v "${BASE_DIR}:${BASE_DIR}" \
      -w "${REPO_DIR}" \
      ${extra_args} \
      "${DOCKER_IMAGE}" \
      "$@"
  else
    eval export ${extra_args//-e /} 2>/dev/null || true
    cd "${REPO_DIR}" && "$@"
  fi
}

# ---- Convenience: print config (only when sourced with -v or DEBUG) ----
if [ "${1:-}" = "-v" ] || [ "${DEBUG:-}" = "1" ]; then
  echo "=== config.sh ==="
  echo "  REPO_DIR:        ${REPO_DIR}"
  echo "  BASE_DIR:        ${BASE_DIR}"
  echo "  DATASET_DIR:     ${DATASET_DIR}"
  echo "  CACHE_DIR:       ${CACHE_DIR}"
  echo "  CHECKPOINT_BASE: ${CHECKPOINT_BASE}"
  echo "  DPO_WORK_DIR:    ${DPO_WORK_DIR}"
  echo "  DOCKER_IMAGE:    ${DOCKER_IMAGE}"
  echo "  USE_DOCKER:      ${USE_DOCKER}"
  echo "================="
fi

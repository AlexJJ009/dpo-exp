#!/usr/bin/env bash
# ==============================================================================
# run_dpo.sh — Launch a dpo-harness training container
#
# Usage:
#   bash scripts/run_dpo.sh                           # interactive shell
#   bash scripts/run_dpo.sh <command> [args...]       # run a command
#
# Examples:
#   bash scripts/run_dpo.sh
#   bash scripts/run_dpo.sh python dpo_pipeline/smoke_test_dpo.py
#   GPUS='"device=0,1,2,3"' bash scripts/run_dpo.sh
# ==============================================================================

set -euo pipefail
source "$(dirname "$0")/config.sh"

IMAGE_NAME="${DOCKER_IMAGE}"
GPUS="${GPUS:-all}"

# Verify image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[run_dpo.sh] ERROR: Docker image '${IMAGE_NAME}' not found."
    exit 1
fi

# Build the docker run command
DOCKER_ARGS=(
    docker run --rm
    --gpus "$GPUS"
    --ipc=host
    -v "${BASE_DIR}:${BASE_DIR}"
    -w "${REPO_DIR}"
    -e HF_HOME="${CACHE_DIR}/huggingface"
    -e HF_HUB_CACHE="${CACHE_DIR}/huggingface"
    -e HF_HUB_OFFLINE=1
    -e PIP_CACHE_DIR="${CACHE_DIR}/pip"
    -e TORCH_HOME="${CACHE_DIR}/torch"
    -e UV_CACHE_DIR="${CACHE_DIR}/uv"
    -e TRITON_CACHE_DIR="${CACHE_DIR}/triton"
)

if [ $# -eq 0 ]; then
    # Interactive mode
    echo "[run_dpo.sh] Launching interactive container (GPUs: ${GPUS}) ..."
    exec "${DOCKER_ARGS[@]}" -it "$IMAGE_NAME" bash
else
    # Script mode
    echo "[run_dpo.sh] Running: $* (GPUs: ${GPUS})"
    exec "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "$@"
fi

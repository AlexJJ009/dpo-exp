#!/usr/bin/env bash
# ==============================================================================
# run_dpo.sh — Launch a dpo-harness training container
#
# Usage:
#   bash /data-1/dpo-experiment/run_dpo.sh                        # interactive shell
#   bash /data-1/dpo-experiment/run_dpo.sh <script.py> [args...]  # run a script
#
# Examples:
#   # Interactive shell (all GPUs)
#   bash /data-1/dpo-experiment/run_dpo.sh
#
#   # Run smoke test
#   bash /data-1/dpo-experiment/run_dpo.sh \
#       python /data-1/dpo-experiment/dpo_pipeline/smoke_test_dpo.py
#
#   # Use specific GPUs
#   GPUS='"device=0,1,2,3"' bash /data-1/dpo-experiment/run_dpo.sh
# ==============================================================================

set -euo pipefail

IMAGE_NAME="dpo-harness"
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
    -v /data-1:/data-1
    -e HF_HOME=/data-1/.cache/huggingface
    -e HF_HUB_CACHE=/data-1/.cache/huggingface
    -e HF_HUB_OFFLINE=1
    -e PIP_CACHE_DIR=/data-1/.cache/pip
    -e TORCH_HOME=/data-1/.cache/torch
    -e UV_CACHE_DIR=/data-1/.cache/uv
    -e TRITON_CACHE_DIR=/data-1/.cache/triton
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

#!/usr/bin/env bash
# ==============================================================================
# Platform entry script for DPO training
#
# This script runs inside the allocated container on the platform.
# It sets up the environment and launches the DPO pipeline.
#
# Usage: Configured in run.hope as "worker.script = bash jupyter.sh"
# ==============================================================================

export PATH=$PATH:~/.local/bin

# ---- Network (for Jupyter access) ----
IPS=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:")
array_=(${IPS})
IP=${array_[0]}

# ---- Paths: adjust this to your dolphinfs working directory ----
# Example: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/<your-username>/lgx
LGX_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/$(whoami)/lgx"

# Code repo
export REPO_DIR="${LGX_DIR}/dpo-exp"

# Data / model / checkpoint paths (auto-detected by config.sh from REPO_DIR parent)
# Override here if your data is in a different location:
# export DATASET_DIR="${LGX_DIR}/dataset"
# export CACHE_DIR="${LGX_DIR}/.cache"
# export CHECKPOINT_BASE="${LGX_DIR}/checkpoints"

# No docker inside the container
export USE_DOCKER=0

cd "${REPO_DIR}"

# ---- Optional: install missing deps at runtime (if image doesn't have them) ----
# pip install vllm==0.12.0 "trl==0.29.0" "deepspeed==0.18.9" math-verify latex2sympy2-extended 2>/dev/null

# ---- Launch Jupyter in background ----
echo "Starting Jupyter on ${IP}:8420 ..."
(python3 -m jupyter lab --ServerApp.token="oNya685" --port 8420 --ip $IP) &

# ---- Run the DPO experiment ----
# Option 1: Run a specific experiment directly
bash experiments/run_4b_code_sft_code.sh

# Option 2: Run multiple experiments via queue
# bash scripts/run_queue.sh run_4b_code_sft_code.sh

wait

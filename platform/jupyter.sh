#!/usr/bin/env bash
export PATH=$PATH:~/.local/bin

# ---- Network ----
IPS=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:")
array_=(${IPS})
IP=${array_[0]}

# ---- Paths ----
LGX_DIR="/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx"
WHEELS_DIR="${LGX_DIR}/dpo-wheels"
export REPO_DIR="${LGX_DIR}/dpo-exp"
export USE_DOCKER=0

cd "${REPO_DIR}"

# ---- Install deps from local wheels (no network needed, ~30s) ----
MARKER="${LGX_DIR}/.deps_installed"
if [ ! -f "${MARKER}" ]; then
  echo "Installing dependencies from local wheels ..."
  pip install --no-cache-dir --no-deps ${WHEELS_DIR}/*.whl 2>&1 | tail -5
  python -c "import vllm; print(f'vLLM: {vllm.__version__}'); import trl; print(f'TRL: {trl.__version__}')"
  touch "${MARKER}"
  echo "Dependencies installed."
else
  echo "Dependencies already installed (marker: ${MARKER})"
fi

# ---- Jupyter ----
echo "Starting Jupyter on ${IP}:8420 ..."
(python3 -m jupyter lab --ServerApp.token="oNya685" --port 8420 --ip $IP) &

# ---- Run DPO ----
bash experiments/run_4b_code_sft_code.sh

wait

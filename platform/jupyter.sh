#!/usr/bin/env bash
export PATH=$PATH:~/.local/bin

# ---- Network ----
IPS=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:")
array_=(${IPS})
IP=${array_[0]}

# ---- Paths ----
LGX_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/$(whoami)/lgx"
export REPO_DIR="${LGX_DIR}/dpo-exp"
export USE_DOCKER=0

cd "${REPO_DIR}"

# ---- Install deps at runtime (build env has no external PyPI access) ----
echo "Installing dependencies ..."
pip install --no-cache-dir vllm==0.12.0 "trl==0.29.0" "deepspeed==0.18.9" \
    math-verify latex2sympy2-extended pylatexenc 2>&1 | tail -5
echo "Verifying ..."
python -c "import vllm; print(f'vLLM: {vllm.__version__}'); import trl; print(f'TRL: {trl.__version__}')"

# ---- Jupyter ----
echo "Starting Jupyter on ${IP}:8420 ..."
(python3 -m jupyter lab --ServerApp.token="oNya685" --port 8420 --ip $IP) &

# ---- Run DPO ----
bash experiments/run_4b_code_sft_code.sh

wait

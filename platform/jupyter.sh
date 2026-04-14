#!/usr/bin/env bash
# Heavily instrumented launcher. When debugging platform startup issues, the priority
# is maximum visibility before any work starts. Once the pipeline is stable, strip
# the DEBUG sections below.
set -uxo pipefail
export PATH=$PATH:~/.local/bin

# ==================== Paths ====================
LGX_DIR="/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx"
WHEELS_DIR="${LGX_DIR}/dpo-wheels"
export REPO_DIR="${LGX_DIR}/dpo-exp"
export USE_DOCKER=0

# ==================== Dual logging (stdout + dolphinfs) ====================
# Try two possible dolphinfs roots; use whichever is writable.
LOG_ROOT=""
for candidate in \
    "${LGX_DIR}/logs" \
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/logs"; do
  if mkdir -p "${candidate}" 2>/dev/null && [ -w "${candidate}" ]; then
    LOG_ROOT="${candidate}"
    break
  fi
done
if [ -z "${LOG_ROOT}" ]; then
  LOG_ROOT="/tmp/dpo_logs"
  mkdir -p "${LOG_ROOT}"
  echo "WARN: dolphinfs log dirs not writable -- falling back to ${LOG_ROOT}"
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/run_${TS}_$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Save the final exit code so we can always log it even on script failure.
trap 'rc=$?; echo "=== EXIT rc=${rc} at $(date -Is) ==="; exit ${rc}' EXIT

echo "================================================================"
echo "=== jupyter.sh start @ $(date -Is)"
echo "=== LOG_FILE=${LOG_FILE}"
echo "================================================================"

# ==================== Section 1: Identity & host ====================
section() { echo; echo "---------- $1 ----------"; }

section "identity"
whoami || true
id || true
hostname || true
uname -a || true
cat /etc/os-release 2>/dev/null | head -10 || true

# ==================== Section 2: AFO / platform env ====================
section "afo/platform env"
# Everything AFO-related in the environment -- jobId, role, task index, cluster spec, image, etc.
env | grep -iE '^(AFO|HOPE|K8S|CONTAINER|HADOOP_USER|FS_CLIENT|HDFS|TF_CONFIG|LD_LIB|CUDA|NVIDIA)' | sort || true
echo "---- TF_CONFIG ----"
echo "${TF_CONFIG:-<unset>}"
echo "---- AFO_ENV_CLUSTER_SPEC ----"
echo "${AFO_ENV_CLUSTER_SPEC:-<unset>}"

# ==================== Section 3: Hardware resources ====================
section "cpu/mem"
nproc || true
echo "--- /proc/meminfo (top) ---"
head -5 /proc/meminfo || true
echo "--- cgroup mem limit ---"
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
  || cat /sys/fs/cgroup/memory.max 2>/dev/null \
  || echo "(no cgroup mem info)"
echo "--- cgroup cpu quota ---"
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null \
  || cat /sys/fs/cgroup/cpu.max 2>/dev/null \
  || echo "(no cgroup cpu info)"
echo "--- ulimits ---"
ulimit -a || true

section "gpu"
which nvidia-smi && nvidia-smi || echo "WARN: no nvidia-smi on PATH"
echo "--- CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ---"
ls /dev/nvidia* 2>/dev/null || echo "(no /dev/nvidia* devices visible)"

section "disk"
df -h / /tmp "${LGX_DIR}" "${WHEELS_DIR}" "${REPO_DIR}" 2>&1 | head -40 || true
echo "--- shm ---"
df -h /dev/shm 2>&1 || true

section "network"
hostname -I 2>/dev/null || hostname -i || true
# ip preferred over ifconfig on modern images
(ip -br addr show 2>/dev/null || ifconfig -a 2>/dev/null) | head -30 || true

# ==================== Section 4: Filesystem sanity ====================
section "dolphinfs paths"
for p in "${LGX_DIR}" "${WHEELS_DIR}" "${REPO_DIR}" \
         "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx" \
         "/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx"; do
  if [ -e "$p" ]; then
    echo "OK   $p ($(stat -c '%U:%G %a' "$p" 2>/dev/null || echo '?'))"
    ls -la "$p" 2>&1 | head -5
  else
    echo "MISS $p"
  fi
done

section "wheels inventory"
if [ -d "${WHEELS_DIR}" ]; then
  ls -la "${WHEELS_DIR}" | head -30
  echo "--- total wheel size ---"
  du -sh "${WHEELS_DIR}" 2>/dev/null || true
  echo "--- wheel count ---"
  find "${WHEELS_DIR}" -name '*.whl' | wc -l
else
  echo "ERROR: WHEELS_DIR=${WHEELS_DIR} does not exist"
fi

section "repo inventory"
if [ -d "${REPO_DIR}" ]; then
  ls -la "${REPO_DIR}" | head -30
  echo "--- git HEAD ---"
  (cd "${REPO_DIR}" && git log -1 --oneline 2>&1 || true)
else
  echo "ERROR: REPO_DIR=${REPO_DIR} does not exist"
fi

# ==================== Section 5: Python & pip ====================
section "python/pip"
which python3 || echo "WARN: no python3"
python3 --version 2>&1 || true
which pip || echo "WARN: no pip"
pip --version 2>&1 || true
echo "--- sys.path ---"
python3 -c 'import sys; [print(p) for p in sys.path]' 2>&1 || true
echo "--- already installed (grep target pkgs) ---"
pip list 2>/dev/null | grep -iE '^(vllm|trl|deepspeed|torch|transformers|accelerate|peft|datasets|flash-attn)' || echo "(none of target pkgs found yet)"

# ==================== Section 6: cd into repo (fail loudly if missing) ====================
section "cd to repo"
if [ ! -d "${REPO_DIR}" ]; then
  echo "FATAL: REPO_DIR=${REPO_DIR} not found -- cannot proceed"
  exit 10
fi
cd "${REPO_DIR}"
pwd
ls -la | head -20

# ==================== Section 7: Install wheels (with full pip output) ====================
section "install wheels"
MARKER="${LGX_DIR}/.deps_installed"
if [ ! -f "${MARKER}" ]; then
  if [ ! -d "${WHEELS_DIR}" ] || [ -z "$(ls -A ${WHEELS_DIR}/*.whl 2>/dev/null)" ]; then
    echo "FATAL: no wheels in ${WHEELS_DIR}"
    exit 11
  fi
  echo "Installing $(ls ${WHEELS_DIR}/*.whl | wc -l) wheels (no | tail, show full output) ..."
  pip install --no-cache-dir --no-deps "${WHEELS_DIR}"/*.whl
  echo "--- verifying imports ---"
  python3 -c "
import vllm, trl, deepspeed
print(f'vLLM: {vllm.__version__}')
print(f'TRL: {trl.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
"
  touch "${MARKER}"
  echo "Dependencies installed + verified; marker written to ${MARKER}"
else
  echo "Marker exists (${MARKER}); re-verifying imports only ..."
  python3 -c "import vllm, trl, deepspeed; print(vllm.__version__, trl.__version__, deepspeed.__version__)"
fi

# ==================== Section 8: Jupyter (background) ====================
section "jupyter"
IPS=$(ifconfig -a 2>/dev/null | grep inet | grep -v 127.0.0.1 | grep -v inet6 | awk '{print $2}' | tr -d "addr:")
IP=$(echo ${IPS} | awk '{print $1}')
echo "IP=${IP}"
JUPYTER_LOG="${LOG_ROOT}/jupyter_${TS}.log"
(python3 -m jupyter lab --ServerApp.token="oNya685" --port 8420 --ip "${IP}" \
    > "${JUPYTER_LOG}" 2>&1) &
JUPYTER_PID=$!
echo "Jupyter PID=${JUPYTER_PID} log=${JUPYTER_LOG}"

# ==================== Section 9: Run DPO ====================
section "run DPO pipeline"
echo "--> bash experiments/run_4b_code_sft_code.sh @ $(date -Is)"
bash experiments/run_4b_code_sft_code.sh
DPO_RC=$?
echo "--> DPO script exited with rc=${DPO_RC} @ $(date -Is)"

wait

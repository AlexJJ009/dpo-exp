#!/usr/bin/env bash
# Launcher for the DPO smoke run on MLP with the dpo_trl image.
#
# Path discipline: the script resolves LGX_DIR from its own location, NOT from
# a hardcoded absolute path. Guarantee required from the user: this file sits
# at `lgx/hope_dir/jupyter.sh`, and everything under `lgx/` has consistent
# sub-folder names (dpo-exp/, dataset/, checkpoints/, logs/, beacons/, ...).
# The absolute path up to `lgx/` may differ between machines and we don't care.
#
# This file replaces the older heavy-instrumentation version — the new dpo_trl
# image ships TRL 0.29.0 / DeepSpeed 0.18.9 / math-verify pre-baked, so we no
# longer need to install them at runtime.

set -uo pipefail
export PATH=$PATH:~/.local/bin

# ==================== Resolve paths relative to this script ====================
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" 2>/dev/null && pwd)"
if [ -z "${SCRIPT_DIR}" ]; then SCRIPT_DIR="$(pwd)"; fi
# jupyter.sh lives at lgx/hope_dir/jupyter.sh → parent = hope_dir → parent of parent = lgx
LGX_DIR="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd)"
if [ -z "${LGX_DIR}" ]; then
  echo "FATAL: could not resolve LGX_DIR from SCRIPT_DIR=${SCRIPT_DIR}"; exit 10
fi

REPO_DIR="${LGX_DIR}/dpo-exp"
LOG_ROOT="${LGX_DIR}/logs"
BEACON_ROOT="${LGX_DIR}/beacons"
export USE_DOCKER=0

mkdir -p "${LOG_ROOT}" "${BEACON_ROOT}" 2>/dev/null || true
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/run_${TS}_$$.log"
if [ -w "${LOG_ROOT}" ]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1
else
  echo "WARN: ${LOG_ROOT} not writable; log is UI-only"
fi

trap 'rc=$?; echo "=== EXIT rc=${rc} at $(date -Is) ==="; exit ${rc}' EXIT

echo "================================================================"
echo "=== jupyter.sh start @ $(date -Is)"
echo "=== LGX_DIR=${LGX_DIR}"
echo "=== REPO_DIR=${REPO_DIR}"
echo "=== LOG_FILE=${LOG_FILE}"
echo "================================================================"

# Beacon so we can confirm the launcher at least reached this point even if
# the tee log on dolphinfs is slow to surface on the login machine.
START_BEACON="${BEACON_ROOT}/launcher_start_$(hostname)_$(date +%s).txt"
{
  echo "launcher start at $(date -Is)"
  echo "host: $(hostname)"
  echo "log:  ${LOG_FILE}"
} > "${START_BEACON}" 2>/dev/null || true

# ==================== Environment snapshot ====================
section() { echo; echo "---------- $1 ----------"; }

section "identity / gpu / disk"
whoami; id; hostname; uname -a
which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "WARN: no nvidia-smi on PATH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
df -h /dev/shm / "${LGX_DIR}" 2>&1 | head -10

# ==================== Dolphinfs sanity ====================
section "dolphinfs assets"
require() {
  local p="$1"
  if [ -e "$p" ]; then
    echo "  [OK]    $p"
  else
    echo "  [MISS]  $p"
    MISSING_ASSETS=1
  fi
}
MISSING_ASSETS=0
require "${REPO_DIR}"
require "${REPO_DIR}/experiments/run_4b_code_m1_dpo_smoke.sh"
require "${LGX_DIR}/checkpoints/qwen3-4b-base-code-sft-m1"
require "${LGX_DIR}/dataset/code/code-train.jsonl"
require "${LGX_DIR}/dataset/EnsembleLLM-data-processed/HumanEval/test.jsonl"

if [ ! -d "${REPO_DIR}" ]; then
  echo "FATAL: repo not present at ${REPO_DIR} — nothing to run"; exit 11
fi

# ==================== Dep import check (NO install) ====================
# The dpo_trl image bakes in trl==0.29.0 / deepspeed==0.18.9 / math-verify etc.
# If any import fails here the image is wrong, not the script — fail loudly.
section "dep import check (image-baked)"
python3 - <<'PY' || { echo "FATAL: required deps not importable"; exit 12; }
import importlib, sys
required = [
    ("torch",          None),
    ("vllm",           None),
    ("transformers",   None),
    ("trl",            "0.29.0"),
    ("deepspeed",      "0.18.9"),
    ("accelerate",     None),
    ("datasets",       None),
    ("math_verify",    None),
]
bad = 0
for name, want in required:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        tag = f"[OK]" if (want is None or ver == want) else f"[WARN ver={want}]"
        print(f"  {tag:16s} {name} {ver}")
        if want is not None and ver != want:
            bad += 1
    except Exception as e:
        print(f"  [FAIL]           {name}: {e!r}")
        bad += 1
from trl import DPOConfig, DPOTrainer   # class-level smoke
print(f"  [OK]             DPOConfig/DPOTrainer importable")
sys.exit(0 if bad == 0 else 1)
PY

# ==================== Jupyter lab (background, optional interactive shell) ====================
section "jupyter lab (background)"
JUPYTER_LOG="${LOG_ROOT}/jupyter_${TS}.log"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "  IP=${IP:-<none>}  port=8420  token=oNya685"
(python3 -m jupyter lab --ServerApp.token="oNya685" --port 8420 --ip "${IP:-0.0.0.0}" \
    > "${JUPYTER_LOG}" 2>&1) &
JUPYTER_PID=$!
echo "  jupyter PID=${JUPYTER_PID}  log=${JUPYTER_LOG}"

# ==================== Run DPO smoke ====================
section "run DPO smoke pipeline"
if [ "${MISSING_ASSETS}" -ne 0 ]; then
  echo "WARN: some dataset/checkpoint assets are missing. run_code_dpo.sh"
  echo "      preflight will fail loudly — that's expected until the user"
  echo "      transfers the missing files into lgx/ via dolphinfs."
fi
cd "${REPO_DIR}"
echo "--> bash experiments/run_4b_code_m1_dpo_smoke.sh @ $(date -Is)"
bash experiments/run_4b_code_m1_dpo_smoke.sh
DPO_RC=$?
echo "--> smoke exited rc=${DPO_RC} @ $(date -Is)"

DONE_BEACON="${BEACON_ROOT}/launcher_done_$(hostname)_$(date +%s).txt"
echo "launcher done rc=${DPO_RC} at $(date -Is)" > "${DONE_BEACON}" 2>/dev/null || true

wait

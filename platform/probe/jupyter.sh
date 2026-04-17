#!/usr/bin/env bash
# Smoke probe for the dpo_trl image (registry-offlinebiz.sankuai.com/.../
# ai-search/training_..._dpo_trl_8071ad6f:1.0.0).
#
# Goal: validate everything the new image is *supposed* to deliver for DPO training
# and vLLM-based evaluation, WITHOUT starting heavy jobs (no model load, no engine
# spin-up, no distributed init). Failures are collected and reported together so one
# probe run surfaces every gap.
#
# Exit code:
#   0  -> all required checks passed
#   1  -> one or more required checks failed (see "SUMMARY" section below)

set -uo pipefail
set -x

LGX_DIR="/home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/yangfengkai02/lgx"
ALT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx"
REPO_DIR_GUESS="${LGX_DIR}/dpo-exp"

# Pick a writable log dir on dolphinfs; fall back to /tmp.
LOG_ROOT=""
for candidate in "${LGX_DIR}/logs" "${ALT_DIR}/logs"; do
  if mkdir -p "${candidate}" 2>/dev/null && [ -w "${candidate}" ]; then
    LOG_ROOT="${candidate}"; break
  fi
done
[ -z "${LOG_ROOT}" ] && { LOG_ROOT="/tmp/probe_logs"; mkdir -p "${LOG_ROOT}"; }
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/probe_${TS}_$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "=== PROBE START $(date -Is) on $(hostname)"
echo "=== LOG_FILE=${LOG_FILE}"
echo "================================================================"

FAILS=()
record_fail() { FAILS+=("$1"); echo "  [FAIL] $1"; }
record_ok()   { echo "  [OK]   $1"; }

# ==================== A. host & hardware ====================
section() { echo; echo "---------- $1 ----------"; }

section "A. identity / host"
whoami || true
id || true
uname -a || true
nproc || true

section "A. GPU"
if which nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || record_fail "nvidia-smi exited non-zero"
else
  record_fail "nvidia-smi not on PATH"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
ls /dev/nvidia* 2>/dev/null || record_fail "no /dev/nvidia* device nodes"

section "A. shm / disk"
df -h /dev/shm || true
df -h / /tmp 2>&1 | head -10 || true

# ==================== B. Dockerfile-baked ENV ====================
section "B. ENV vars from Dockerfile"
check_env() {
  local name="$1" expect="$2"
  local got="${!name:-<unset>}"
  if [ "${got}" = "${expect}" ]; then
    record_ok "${name}=${got}"
  else
    record_fail "${name} expected '${expect}' got '${got}'"
  fi
}
check_env HF_HUB_OFFLINE     1
check_env VLLM_USE_V1        1
check_env VLLM_NO_USAGE_STATS 1
check_env PYTHONUNBUFFERED   1

# ==================== C/D/E/F. Python-side checks ====================
# Run all Python checks in ONE subprocess so we get a single stdout block and can
# correlate failures. The script prints [OK]/[FAIL] lines that bash then counts.
section "C-F. Python dependency matrix + functional smokes"
PY_REPORT="${LOG_ROOT}/probe_${TS}_py.txt"
python3 - <<'PY' | tee "${PY_REPORT}"
import sys, os, importlib, traceback

def line_ok(tag):   print(f"  [OK]   {tag}")
def line_fail(tag): print(f"  [FAIL] {tag}")

def import_version(mod, want=None, op="=="):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        if want is None:
            line_ok(f"{mod} imported ({ver})")
            return True
        if op == "==":
            ok = (ver == want)
        elif op == ">=":
            ok = tuple(map(int, (ver.split('+')[0].split('.') + ['0','0','0'])[:3])) >= \
                 tuple(map(int, (want.split('.') + ['0','0','0'])[:3]))
        else:
            ok = False
        (line_ok if ok else line_fail)(f"{mod} {op} {want} (got {ver})")
        return ok
    except Exception as e:
        line_fail(f"{mod} import error: {e!r}")
        return False

print("---- C. versions ----")
print(f"  python: {sys.version.split()[0]}  exec: {sys.executable}")

# Required by DPO pipeline:
import_version("torch")
import_version("vllm")
import_version("transformers")
import_version("trl",         "0.29.0", "==")
import_version("deepspeed",   "0.18.9", "==")
import_version("accelerate",  "1.4.0",  ">=")
import_version("datasets",    "3.0.0",  ">=")
import_version("peft")
import_version("math_verify")
import_version("latex2sympy2_extended")
import_version("pylatexenc")

print("---- D. CUDA runtime smoke ----")
try:
    import torch
    if not torch.cuda.is_available():
        line_fail("torch.cuda.is_available() is False")
    else:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        cap  = torch.cuda.get_device_capability(0)
        line_ok(f"torch.cuda: {n} dev(s), dev0={name} sm={cap[0]}.{cap[1]}")
        # Actually execute a kernel (catches broken driver/runtime without a full model)
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        line_ok(f"cuda matmul smoke: sum={y:.3e}")
        torch.cuda.synchronize()
except Exception as e:
    line_fail(f"cuda smoke failed: {e!r}")
    traceback.print_exc()

print("---- E. DPO class imports ----")
try:
    from trl import DPOConfig, DPOTrainer
    # Construct DPOConfig with a tmp output_dir — verifies the dataclass signature
    # we rely on (kwargs added in TRL >=0.28). No training started.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = DPOConfig(output_dir=d, beta=0.1, loss_type="sigmoid")
        line_ok(f"DPOConfig instantiated (beta={cfg.beta}, loss_type={cfg.loss_type})")
    line_ok("DPOTrainer class importable")
except Exception as e:
    line_fail(f"DPO class check failed: {e!r}")
    traceback.print_exc()

print("---- F. math_verify functional smoke ----")
try:
    from math_verify import parse, verify
    gold = parse("$\\frac{1}{2}$")
    pred = parse("0.5")
    ok = bool(verify(gold, pred))
    (line_ok if ok else line_fail)(f"math_verify 1/2 == 0.5 -> {ok}")
except Exception as e:
    line_fail(f"math_verify smoke failed: {e!r}")
    traceback.print_exc()

print("---- F. vLLM import-only smoke (no engine) ----")
try:
    from vllm import LLM, SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    line_ok(f"vllm.LLM / SamplingParams importable (sp.max_tokens={sp.max_tokens})")
except Exception as e:
    line_fail(f"vllm import failed: {e!r}")
    traceback.print_exc()

# Summarize for bash to parse
PY
PY_RC=$?
if [ "${PY_RC}" -ne 0 ]; then
  record_fail "python probe block exited rc=${PY_RC}"
fi
# Count [FAIL] lines the Python block emitted.
PY_FAILS=$(grep -c "^  \[FAIL\] " "${PY_REPORT}" || true)
if [ "${PY_FAILS}" -gt 0 ]; then
  record_fail "python checks: ${PY_FAILS} failure(s) — see ${PY_REPORT}"
fi

# ==================== G. dolphinfs assets (informational) ====================
section "G. dolphinfs assets (informational — missing items reported, non-fatal)"
check_path() {
  local p="$1" required="$2"
  if [ -e "$p" ]; then
    record_ok "exists: $p"
  else
    if [ "${required}" = "required" ]; then
      record_fail "missing (required): $p"
    else
      echo "  [warn] missing (optional): $p"
    fi
  fi
}
# Repo + dataset — needed for real runs, not for proving the image.
check_path "${REPO_DIR_GUESS}"                                               optional
check_path "${REPO_DIR_GUESS}/experiments/run_4b_code_m1_dpo.sh"             optional
check_path "${LGX_DIR}/dataset/code/code-train.jsonl"                        optional
check_path "${LGX_DIR}/dataset/EnsembleLLM-data-processed/HumanEval/test.jsonl"    optional
check_path "${LGX_DIR}/dataset/EnsembleLLM-data-processed/MBPP/test.jsonl"         optional
check_path "${LGX_DIR}/dataset/EnsembleLLM-data-processed/BigCodeBench/test.jsonl" optional
check_path "${LGX_DIR}/dataset/EnsembleLLM-data-processed/LiveCodeBench/test.jsonl" optional
check_path "${LGX_DIR}/checkpoints/qwen3-4b-base-code-sft-m1"                optional

# If a ckpt dir exists, try an HF_HUB_OFFLINE=1 tokenizer load — this is the single
# most common silent failure when a build bakes HF_HUB_OFFLINE in.
CKPT_PROBE="${LGX_DIR}/checkpoints/qwen3-4b-base-code-sft-m1"
if [ -d "${CKPT_PROBE}" ]; then
  section "G. offline tokenizer load (HF_HUB_OFFLINE=1)"
  python3 - <<PY || record_fail "offline tokenizer load failed for ${CKPT_PROBE}"
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${CKPT_PROBE}")
print(f"  [OK]   AutoTokenizer: vocab_size={tok.vocab_size} class={type(tok).__name__}")
PY
fi

# ==================== H. dolphinfs beacon ====================
section "H. beacon"
for D in "${LGX_DIR}" "${ALT_DIR}"; do
  if [ -d "$D" ]; then
    mkdir -p "$D/beacons" 2>/dev/null || true
    BEACON="$D/beacons/probe_$(hostname)_$(date +%s).txt"
    {
      echo "probe alive at $(date -Is)"
      echo "image: dpo_trl_8071ad6f:1.0.0"
      echo "host:  $(hostname)"
      echo "fails: ${#FAILS[@]}"
      echo "log:   ${LOG_FILE}"
    } > "$BEACON" 2>&1 && record_ok "beacon written: $BEACON"
    break
  fi
done

# ==================== SUMMARY ====================
section "SUMMARY"
if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "ALL PROBE CHECKS PASSED"
  PROBE_RC=0
else
  echo "PROBE FAILURES (${#FAILS[@]}):"
  for f in "${FAILS[@]}"; do echo "  - $f"; done
  PROBE_RC=1
fi

# Stay alive for UI inspection, but only after the report is already on disk.
echo "=== sleeping 300s for UI inspection ==="
sleep 300
echo "=== PROBE END $(date -Is) rc=${PROBE_RC} ==="
exit ${PROBE_RC}

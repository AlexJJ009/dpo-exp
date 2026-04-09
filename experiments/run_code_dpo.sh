#!/usr/bin/env bash
#
# Generic Code DPO pipeline.
#
# This script is NOT called directly -- it is sourced by experiment-specific
# wrapper scripts that set the required variables before calling run_pipeline.
#
# Required variables (set by caller):
#   MODEL           - Path to SFT model checkpoint
#   TRAIN_DATA      - Path to code train JSONL (prompt/reference_answer/chosen/test_case/source)
#   EXP_NAME        - Experiment name, used for directory/file naming
#   CHECKPOINT_DIR  - Where to save the DPO checkpoint
#   EVAL_TEST_FILES - Space-separated list of code test JSONL files (HumanEval, MBPP, etc.)
#
# Optional variables (have defaults):
#   NUM_ROLLOUTS    - Rollouts per prompt (default: 16)
#   MAX_TOKENS      - Max tokens per rollout (default: 2048)
#   TEMPERATURE     - Sampling temperature (default: 0.7)
#   TP_SIZE         - Tensor parallel size per instance (default: 1)
#   DP_SIZE         - Data parallel: number of independent GPU instances (default: 8)
#   GPU_MEM_UTIL    - GPU memory utilization (default: 0.9)
#   EVAL_TP         - Eval tensor parallel (default: 1)
#
# Usage (from a wrapper):
#   source run_code_dpo.sh
#   run_pipeline
#
set -euo pipefail

# ==================== Defaults ====================
NUM_ROLLOUTS="${NUM_ROLLOUTS:-16}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# Incremental rollout settings
ROLLOUT_BATCH="${ROLLOUT_BATCH:-2000}"
TARGET_PAIRS="${TARGET_PAIRS:-8000}"
MAX_ROLLOUT_BATCHES="${MAX_ROLLOUT_BATCHES:-10}"

# Eval settings
EVAL_TP="${EVAL_TP:-1}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
EVAL_REPEAT="${EVAL_REPEAT:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-15}"

# Training params
DPO_PER_DEVICE_BATCH="${DPO_PER_DEVICE_BATCH:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-2}"
DPO_PRECOMPUTE_REF="${DPO_PRECOMPUTE_REF:-}"
DPO_OPTIM="${DPO_OPTIM:-adamw_torch}"

# ==================== Derived paths ====================
DATA_DIR="${DPO_WORK_DIR}/${EXP_NAME}"
ROLLOUTS_PATH="${DATA_DIR}/${EXP_NAME}-rollouts.jsonl"
PAIRS_PATH="${DATA_DIR}/${EXP_NAME}-pairs.jsonl"
TRAINING_SUMMARY="${CHECKPOINT_DIR}/training_logs/training_summary.json"

# ==================== Pre-flight ====================
preflight() {
  echo "============================================================"
  echo "  Pre-flight checks: ${EXP_NAME}"
  echo "============================================================"
  echo ""

  local ok=true

  if docker image inspect dpo-harness &>/dev/null; then
    echo "  [OK] Docker image 'dpo-harness'"
  else
    echo "  [FAIL] Docker image 'dpo-harness' not found"
    ok=false
  fi

  echo ""
  if docker run --rm dpo-harness python -c "
import trl; print(f'  [OK] trl=={trl.__version__}')
import deepspeed; print(f'  [OK] deepspeed=={deepspeed.__version__}')
" 2>/dev/null; then true; else
    echo "  [FAIL] dpo-harness dependency check"
    ok=false
  fi

  echo ""
  if [ -f "${MODEL}/config.json" ]; then
    echo "  [OK] Model: ${MODEL}"
  else
    echo "  [FAIL] Model not found: ${MODEL}"
    ok=false
  fi

  if [ -f "${TRAIN_DATA}" ]; then
    local count
    count=$(wc -l < "${TRAIN_DATA}")
    echo "  [OK] Train data: ${count} samples"
  else
    echo "  [FAIL] Train data not found: ${TRAIN_DATA}"
    ok=false
  fi

  for tf in ${EVAL_TEST_FILES}; do
    if [ -f "${tf}" ]; then
      echo "  [OK] Test file: $(basename ${tf})"
    else
      echo "  [FAIL] Test file not found: ${tf}"
      ok=false
    fi
  done

  if [ "${ok}" = false ]; then
    echo ""
    echo "FATAL: Pre-flight checks failed."
    return 1
  fi

  echo ""
  echo ">>> Pre-flight checks passed"
  return 0
}

# ==================== Pipeline steps ====================
run_pipeline() {
  echo "============================================================"
  echo "  Code DPO Pipeline: ${EXP_NAME}"
  echo "============================================================"
  echo ""
  echo "  Model:       ${MODEL}"
  echo "  Train data:  ${TRAIN_DATA}"
  echo "  Checkpoint:  ${CHECKPOINT_DIR}"
  echo "  Eval files:  ${EVAL_TEST_FILES}"
  echo "  Rollouts:    ${NUM_ROLLOUTS}/prompt"
  echo ""

  preflight || exit 1

  mkdir -p "${DATA_DIR}"

  # ---- Step 1+2: Incremental rollout + pair building ----
  echo ""
  echo "============================================================"
  echo "  STEP 1-2/4: Incremental rollout + pair building (code mode)"
  echo "  Target: ${TARGET_PAIRS} pairs, batch size: ${ROLLOUT_BATCH} prompts"
  echo "============================================================"
  echo ""

  local pair_count=0
  if [ -f "${PAIRS_PATH}" ]; then
    pair_count=$(wc -l < "${PAIRS_PATH}")
  fi

  if [ "${pair_count}" -ge "${TARGET_PAIRS}" ]; then
    echo ">>> SKIP: Already have ${pair_count} pairs (target: ${TARGET_PAIRS})"
  else
    local offset=0
    local batch_num=0

    if [ -f "${ROLLOUTS_PATH}" ]; then
      offset=$(wc -l < "${ROLLOUTS_PATH}")
      echo ">>> Resuming from offset ${offset} (${pair_count} pairs so far)"
    fi

    while [ "${pair_count}" -lt "${TARGET_PAIRS}" ] && [ "${batch_num}" -lt "${MAX_ROLLOUT_BATCHES}" ]; do
      batch_num=$((batch_num + 1))
      echo ""
      echo "--- Batch ${batch_num}: rollout prompts[${offset}:$((offset + ROLLOUT_BATCH))] ---"

      # Rollout this batch
      docker run --rm --gpus all --ipc=host \
        -v "${BASE_DIR}:${BASE_DIR}" \
        -w "${REPO_DIR}" \
        dpo-harness \
        python dpo_pipeline/batch_rollout.py \
          --input "${TRAIN_DATA}" \
          --output "${ROLLOUTS_PATH}" \
          --model "${MODEL}" \
          --num-rollouts ${NUM_ROLLOUTS} \
          --max-tokens ${MAX_TOKENS} \
          --temperature ${TEMPERATURE} \
          --tensor-parallel-size ${TP_SIZE} \
          --data-parallel-size ${DP_SIZE} \
          --gpu-memory-utilization ${GPU_MEM_UTIL} \
          --chat-template \
          --offset ${offset} \
          --limit ${ROLLOUT_BATCH} \
          --append

      # Build pairs with code verification
      docker run --rm --gpus all --ipc=host \
        -v "${BASE_DIR}:${BASE_DIR}" \
        -w "${REPO_DIR}" \
        dpo-harness \
        python dpo_pipeline/build_pairs.py \
          --input "${ROLLOUTS_PATH}" \
          --output "${PAIRS_PATH}" \
          --code

      offset=$((offset + ROLLOUT_BATCH))
      pair_count=$(wc -l < "${PAIRS_PATH}")
      echo "--- Batch ${batch_num} done: ${pair_count} total pairs (target: ${TARGET_PAIRS}) ---"
    done

    if [ "${pair_count}" -lt 100 ]; then
      echo "ERROR: Only ${pair_count} pairs after ${batch_num} batches. Expected >= 100."
      exit 1
    fi
    echo ""
    echo ">>> Step 1-2 complete: ${pair_count} pairs from ${offset} prompts (${batch_num} batches)"
  fi

  local pair_count
  pair_count=$(wc -l < "${PAIRS_PATH}")
  echo ">>> Total pairs: ${pair_count}"

  # ---- Step 3: DPO Training ----
  echo ""
  echo "============================================================"
  echo "  STEP 3/4: DPO Training (DeepSpeed ZeRO 2, 8 GPUs)"
  echo "============================================================"
  echo ""

  if [ -f "${TRAINING_SUMMARY}" ]; then
    echo ">>> SKIP: Training already completed"
  else
    docker run --rm --gpus all --ipc=host \
      -v "${BASE_DIR}:${BASE_DIR}" \
      -w "${REPO_DIR}" \
      -e DPO_MODEL_NAME="${MODEL}" \
      -e DPO_DATASET_PATH="${PAIRS_PATH}" \
      -e DPO_OUTPUT_DIR="${CHECKPOINT_DIR}" \
      -e DPO_PER_DEVICE_BATCH="${DPO_PER_DEVICE_BATCH}" \
      -e DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM}" \
      -e DPO_PRECOMPUTE_REF="${DPO_PRECOMPUTE_REF}" \
      -e DPO_OPTIM="${DPO_OPTIM}" \
      dpo-harness \
      accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
        dpo_pipeline/train_dpo_mcq.py

    echo ">>> Step 3 complete"
  fi

  if [ -f "${TRAINING_SUMMARY}" ]; then
    echo ""
    python3 -c "
import json
with open('${TRAINING_SUMMARY}') as f:
    s = json.load(f)
r = s['results']
print(f\"  Dataset:  {s['dataset_size']} pairs\")
print(f\"  Steps:    {r['total_steps']}\")
print(f\"  Loss:     {r['first_step_loss']:.4f} -> {r['final_step_loss']:.4f}\")
print(f\"  Margins:  {r['first_step_margins']:.4f} -> {r['final_step_margins']:.4f}\")
print(f\"  Runtime:  {r['training_runtime_seconds']:.0f}s\")
"
  fi

  # ---- Step 4: Evaluation (code execution) ----
  echo ""
  echo "============================================================"
  echo "  STEP 4/4: Code Evaluation"
  echo "============================================================"
  echo ""

  local eval_done=true
  for tf in ${EVAL_TEST_FILES}; do
    local ds_name
    ds_name=$(basename "${tf}" .jsonl)
    ds_name=$(basename "${ds_name}" .json)
    local eval_summary="${CHECKPOINT_DIR}/eval_code/${ds_name}/eval_summary.json"
    if [ ! -f "${eval_summary}" ]; then
      eval_done=false
      break
    fi
  done

  if [ "${eval_done}" = true ]; then
    echo ">>> SKIP: Evaluation already completed"
  else
    for tf in ${EVAL_TEST_FILES}; do
      local ds_name
      ds_name=$(basename "${tf}" .jsonl)
      ds_name=$(basename "${ds_name}" .json)
      local eval_out="${CHECKPOINT_DIR}/eval_code/${ds_name}"
      local eval_summary="${eval_out}/eval_summary.json"

      if [ -f "${eval_summary}" ]; then
        echo ">>> SKIP: ${ds_name} already evaluated"
        continue
      fi

      echo "--- Evaluating: ${ds_name} ---"
      docker run --rm --gpus all --ipc=host \
        -v "${BASE_DIR}:${BASE_DIR}" \
        -w "${REPO_DIR}" \
        dpo-harness \
        python dpo_pipeline/eval_vllm_code.py \
          --dataset "${tf}" \
          --model "${CHECKPOINT_DIR}" \
          --tp ${EVAL_TP} \
          --max_model_len ${EVAL_MAX_TOKENS} \
          --timeout ${EVAL_TIMEOUT} \
          --repeat ${EVAL_REPEAT} \
          --out_dir "${eval_out}"
    done

    echo ">>> Step 4 complete"
  fi

  # ---- Summary ----
  echo ""
  echo "============================================================"
  echo "  PIPELINE COMPLETE: ${EXP_NAME}"
  echo "============================================================"
  echo ""
  echo "  Pairs:      ${PAIRS_PATH}"
  echo "  Checkpoint:  ${CHECKPOINT_DIR}"
  echo ""

  for tf in ${EVAL_TEST_FILES}; do
    local ds_name
    ds_name=$(basename "${tf}" .jsonl)
    ds_name=$(basename "${ds_name}" .json)
    local eval_summary="${CHECKPOINT_DIR}/eval_code/${ds_name}/eval_summary.json"
    if [ -f "${eval_summary}" ]; then
      python3 -c "
import json
with open('${eval_summary}') as f:
    s = json.load(f)
print(f\"  {s.get('dataset_path','?'):40s}  acc={s['acc']:.4f}  pass@k={s.get('pass_at_k', 'N/A')}\")
"
    fi
  done
}

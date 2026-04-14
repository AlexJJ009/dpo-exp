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

# Code tasks have high error rates (~40-60% of rollouts are wrong),
# so fewer prompts are needed to reach TARGET_PAIRS.
# 800 prompts × 16 rollouts × ~40% error ≈ 5000 pairs per batch.
ROLLOUT_BATCH="${ROLLOUT_BATCH:-800}"
TARGET_PAIRS="${TARGET_PAIRS:-8000}"
MAX_ROLLOUT_BATCHES="${MAX_ROLLOUT_BATCHES:-10}"

EVAL_TP="${EVAL_TP:-1}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
EVAL_REPEAT="${EVAL_REPEAT:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-15}"

DPO_PER_DEVICE_BATCH="${DPO_PER_DEVICE_BATCH:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-2}"
DPO_PRECOMPUTE_REF="${DPO_PRECOMPUTE_REF:-}"
DPO_OPTIM="${DPO_OPTIM:-adamw_torch}"
DPO_MAX_PAIRS="${DPO_MAX_PAIRS:-${TARGET_PAIRS}}"

# ==================== Derived paths ====================
DATA_DIR="${DPO_WORK_DIR}/${EXP_NAME}"
ROLLOUTS_PATH="${DATA_DIR}/${EXP_NAME}-rollouts.jsonl"
PAIRS_PATH="${DATA_DIR}/${EXP_NAME}-pairs.jsonl"
TRAINING_SUMMARY="${CHECKPOINT_DIR}/training_logs/training_summary.json"

# ==================== Pre-flight ====================
preflight() {
  echo "============================================================"
  echo "  Pre-flight checks: ${EXP_NAME}"
  echo "  Mode: $([ "${USE_DOCKER}" = "1" ] && echo "Docker (${DOCKER_IMAGE})" || echo "Native")"
  echo "============================================================"

  local ok=true

  # Check dependencies
  run_cmd python -c "
import trl; print(f'  [OK] trl=={trl.__version__}')
import deepspeed; print(f'  [OK] deepspeed=={deepspeed.__version__}')
" 2>/dev/null || { echo "  [FAIL] dependency check"; ok=false; }

  [ -f "${MODEL}/config.json" ] && echo "  [OK] Model: ${MODEL}" \
    || { echo "  [FAIL] Model not found: ${MODEL}"; ok=false; }

  if [ -f "${TRAIN_DATA}" ]; then
    echo "  [OK] Train data: $(wc -l < "${TRAIN_DATA}") samples"
  else
    echo "  [FAIL] Train data not found: ${TRAIN_DATA}"; ok=false
  fi

  for tf in ${EVAL_TEST_FILES}; do
    [ -f "${tf}" ] && echo "  [OK] Test: $(basename ${tf})" \
      || { echo "  [FAIL] Not found: ${tf}"; ok=false; }
  done

  [ "${ok}" = false ] && { echo "FATAL: Pre-flight failed."; return 1; }
  echo ">>> Pre-flight passed"
}

# ==================== Pipeline ====================
run_pipeline() {
  echo "============================================================"
  echo "  Code DPO Pipeline: ${EXP_NAME}"
  echo "============================================================"
  echo "  Model:       ${MODEL}"
  echo "  Train data:  ${TRAIN_DATA}"
  echo "  Checkpoint:  ${CHECKPOINT_DIR}"
  echo "  Eval files:  ${EVAL_TEST_FILES}"
  echo "  Rollouts:    ${NUM_ROLLOUTS}/prompt"

  preflight || exit 1
  mkdir -p "${DATA_DIR}"

  # ---- Step 1+2: Incremental rollout + pair building ----
  echo ""
  echo "=== STEP 1-2/4: Rollout + pair building (code mode) ==="

  local pair_count=0
  [ -f "${PAIRS_PATH}" ] && pair_count=$(wc -l < "${PAIRS_PATH}")

  if [ "${pair_count}" -ge "${TARGET_PAIRS}" ]; then
    echo ">>> SKIP: Already have ${pair_count} pairs (target: ${TARGET_PAIRS})"
  else
    local offset=0 batch_num=0
    [ -f "${ROLLOUTS_PATH}" ] && offset=$(wc -l < "${ROLLOUTS_PATH}") \
      && echo ">>> Resuming from offset ${offset} (${pair_count} pairs so far)"

    while [ "${pair_count}" -lt "${TARGET_PAIRS}" ] && [ "${batch_num}" -lt "${MAX_ROLLOUT_BATCHES}" ]; do
      batch_num=$((batch_num + 1))
      echo "--- Batch ${batch_num}: prompts[${offset}:$((offset + ROLLOUT_BATCH))] ---"

      run_cmd python dpo_pipeline/batch_rollout.py \
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

      run_cmd python dpo_pipeline/build_pairs.py \
        --input "${ROLLOUTS_PATH}" \
        --output "${PAIRS_PATH}" \
        --code

      offset=$((offset + ROLLOUT_BATCH))
      pair_count=$(wc -l < "${PAIRS_PATH}")
      echo "--- Batch ${batch_num}: ${pair_count} pairs (target: ${TARGET_PAIRS}) ---"
    done

    [ "${pair_count}" -lt 100 ] && { echo "ERROR: Only ${pair_count} pairs."; exit 1; }
    echo ">>> Step 1-2 complete: ${pair_count} pairs"
  fi

  pair_count=$(wc -l < "${PAIRS_PATH}")
  echo ">>> Total pairs: ${pair_count}"

  # ---- Step 3: DPO Training ----
  echo ""
  echo "=== STEP 3/4: DPO Training ==="

  if [ -f "${TRAINING_SUMMARY}" ]; then
    echo ">>> SKIP: Training already completed"
  else
    run_cmd_env \
      "-e DPO_MODEL_NAME=${MODEL} -e DPO_DATASET_PATH=${PAIRS_PATH} -e DPO_OUTPUT_DIR=${CHECKPOINT_DIR} -e DPO_PER_DEVICE_BATCH=${DPO_PER_DEVICE_BATCH} -e DPO_GRAD_ACCUM=${DPO_GRAD_ACCUM} -e DPO_PRECOMPUTE_REF=${DPO_PRECOMPUTE_REF} -e DPO_OPTIM=${DPO_OPTIM} -e DPO_MAX_PAIRS=${DPO_MAX_PAIRS}" \
      accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
        dpo_pipeline/train_dpo_mcq.py
    echo ">>> Step 3 complete"
  fi

  if [ -f "${TRAINING_SUMMARY}" ]; then
    python3 -c "
import json
with open('${TRAINING_SUMMARY}') as f:
    s = json.load(f)
r = s['results']
print(f\"  Dataset: {s['dataset_size']} | Steps: {r['total_steps']} | Loss: {r['first_step_loss']:.4f}->{r['final_step_loss']:.4f} | Runtime: {r['training_runtime_seconds']:.0f}s\")
"
  fi

  # ---- Step 4: Code Evaluation ----
  echo ""
  echo "=== STEP 4/4: Code Evaluation ==="

  local eval_done=true
  for tf in ${EVAL_TEST_FILES}; do
    local ds_name; ds_name=$(basename "${tf}" .jsonl); ds_name=$(basename "${ds_name}" .json)
    [ ! -f "${CHECKPOINT_DIR}/eval_code/${ds_name}/eval_summary.json" ] && eval_done=false && break
  done

  if [ "${eval_done}" = true ]; then
    echo ">>> SKIP: Evaluation already completed"
  else
    for tf in ${EVAL_TEST_FILES}; do
      local ds_name; ds_name=$(basename "${tf}" .jsonl); ds_name=$(basename "${ds_name}" .json)
      local eval_out="${CHECKPOINT_DIR}/eval_code/${ds_name}"
      [ -f "${eval_out}/eval_summary.json" ] && echo ">>> SKIP: ${ds_name}" && continue

      echo "--- Evaluating: ${ds_name} ---"
      run_cmd python dpo_pipeline/eval_vllm_code.py \
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
  echo "=== PIPELINE COMPLETE: ${EXP_NAME} ==="
  echo "  Pairs:      ${PAIRS_PATH}"
  echo "  Checkpoint:  ${CHECKPOINT_DIR}"
  for tf in ${EVAL_TEST_FILES}; do
    local ds_name; ds_name=$(basename "${tf}" .jsonl); ds_name=$(basename "${ds_name}" .json)
    local summary="${CHECKPOINT_DIR}/eval_code/${ds_name}/eval_summary.json"
    [ -f "${summary}" ] && python3 -c "
import json
with open('${summary}') as f: s=json.load(f)
print(f\"  {s.get('dataset_path','?'):40s} acc={s['acc']:.4f} pass@k={s.get('pass_at_k','N/A')}\")
"
  done
}

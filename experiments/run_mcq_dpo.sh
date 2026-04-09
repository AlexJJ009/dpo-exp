#!/usr/bin/env bash
#
# Generic MCQ cross-domain DPO pipeline.
#
# This script is NOT called directly — it is sourced by experiment-specific
# wrapper scripts that set the required variables before calling run_pipeline.
#
# Required variables (set by caller):
#   MODEL           - Path to SFT model checkpoint
#   TRAIN_DATA      - Path to unified train JSONL (prompt/reference_answer/chosen)
#   EXP_NAME        - Experiment name, used for directory/file naming
#   CHECKPOINT_DIR  - Where to save the DPO checkpoint
#   EVAL_TEST_FILES - Space-separated list of test parquet files
#
# Optional variables (have defaults):
#   NUM_ROLLOUTS    - Rollouts per prompt (default: 16)
#   MAX_TOKENS      - Max tokens per rollout (default: 2048)
#   TEMPERATURE     - Sampling temperature (default: 0.7)
#   TP_SIZE         - Tensor parallel size per instance (default: 1)
#   DP_SIZE         - Data parallel: number of independent GPU instances (default: 8)
#   GPU_MEM_UTIL    - GPU memory utilization (default: 0.9)
#   EVAL_N          - Eval responses per prompt (default: 3)
#   EVAL_MAX_TOKENS - Max tokens for eval (default: 2048)
#
# Usage (from a wrapper):
#   source run_mcq_dpo.sh
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
ROLLOUT_BATCH="${ROLLOUT_BATCH:-2000}"          # prompts per batch
TARGET_PAIRS="${TARGET_PAIRS:-8000}"            # stop when we have this many pairs
MAX_ROLLOUT_BATCHES="${MAX_ROLLOUT_BATCHES:-10}" # safety cap on iterations
EVAL_N="${EVAL_N:-3}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-2048}"
EVAL_TP="${EVAL_TP:-1}"
EVAL_DP="${EVAL_DP:-8}"

# Training params (per model size, set by wrapper)
DPO_PER_DEVICE_BATCH="${DPO_PER_DEVICE_BATCH:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-2}"
DPO_PRECOMPUTE_REF="${DPO_PRECOMPUTE_REF:-}"
DPO_OPTIM="${DPO_OPTIM:-adamw_torch}"

# ==================== Derived paths ====================
DATA_DIR="${DPO_WORK_DIR}/${EXP_NAME}"
ROLLOUTS_PATH="${DATA_DIR}/${EXP_NAME}-rollouts.jsonl"
PAIRS_PATH="${DATA_DIR}/${EXP_NAME}-pairs.jsonl"
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/inference_n${EVAL_N}"
TRAINING_SUMMARY="${CHECKPOINT_DIR}/training_logs/training_summary.json"
EVAL_METRICS="${EVAL_OUTPUT_DIR}/eval_metrics.json"

# ==================== Pre-flight ====================
preflight() {
  echo "============================================================"
  echo "  Pre-flight checks: ${EXP_NAME}"
  echo "============================================================"
  echo ""

  local ok=true

  if docker image inspect "${DOCKER_IMAGE}" &>/dev/null; then
    echo "  [OK] Docker image '${DOCKER_IMAGE}'"
  else
    echo "  [FAIL] Docker image '${DOCKER_IMAGE}' not found"
    ok=false
  fi

  echo ""
  if docker run --rm "${DOCKER_IMAGE}" python -c "
import trl; print(f'  [OK] trl=={trl.__version__}')
import deepspeed; print(f'  [OK] deepspeed=={deepspeed.__version__}')
" 2>/dev/null; then true; else
    echo "  [FAIL] ${DOCKER_IMAGE} dependency check"
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
  echo "  MCQ DPO Pipeline: ${EXP_NAME}"
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
  echo "  STEP 1-2/4: Incremental rollout + pair building"
  echo "  Target: ${TARGET_PAIRS} pairs, batch size: ${ROLLOUT_BATCH} prompts"
  echo "============================================================"
  echo ""

  # Check if we already have enough pairs
  local pair_count=0
  if [ -f "${PAIRS_PATH}" ]; then
    pair_count=$(wc -l < "${PAIRS_PATH}")
  fi

  if [ "${pair_count}" -ge "${TARGET_PAIRS}" ]; then
    echo ">>> SKIP: Already have ${pair_count} pairs (target: ${TARGET_PAIRS})"
  else
    local offset=0
    local batch_num=0

    # If rollouts file exists, count how many prompts were already processed
    if [ -f "${ROLLOUTS_PATH}" ]; then
      offset=$(wc -l < "${ROLLOUTS_PATH}")
      echo ">>> Resuming from offset ${offset} (${pair_count} pairs so far)"
    fi

    while [ "${pair_count}" -lt "${TARGET_PAIRS}" ] && [ "${batch_num}" -lt "${MAX_ROLLOUT_BATCHES}" ]; do
      batch_num=$((batch_num + 1))
      echo ""
      echo "--- Batch ${batch_num}: rollout prompts[${offset}:$((offset + ROLLOUT_BATCH))] ---"

      # Rollout this batch (append to rollouts file)
      docker run --rm --gpus all --ipc=host \
        -v "${BASE_DIR}:${BASE_DIR}" \
        -w "${REPO_DIR}" \
        "${DOCKER_IMAGE}" \
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
          --no-think-seed \
          --offset ${offset} \
          --limit ${ROLLOUT_BATCH} \
          --append

      # Build pairs from ALL accumulated rollouts (with dedup)
      docker run --rm --gpus all --ipc=host \
        -v "${BASE_DIR}:${BASE_DIR}" \
        -w "${REPO_DIR}" \
        "${DOCKER_IMAGE}" \
        python dpo_pipeline/build_pairs.py \
          --input "${ROLLOUTS_PATH}" \
          --output "${PAIRS_PATH}" \
          --strict \
          --skip-think-filter

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
      "${DOCKER_IMAGE}" \
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

  # ---- Step 4: Evaluation ----
  echo ""
  echo "============================================================"
  echo "  STEP 4/4: Evaluation (n=${EVAL_N})"
  echo "============================================================"
  echo ""

  if [ -f "${EVAL_METRICS}" ]; then
    echo ">>> SKIP: Evaluation already completed"
  else
    # Build --test_files arguments
    local test_args=""
    for tf in ${EVAL_TEST_FILES}; do
      test_args="${test_args} ${tf}"
    done

    docker run --rm --gpus all --ipc=host \
      -v "${BASE_DIR}:${BASE_DIR}" \
      -w "${REPO_DIR}" \
      "${DOCKER_IMAGE}" \
      python dpo_pipeline/eval/offline_eval.py \
          --model_path "${CHECKPOINT_DIR}" \
          --test_files ${test_args} \
          --n ${EVAL_N} \
          --tensor_parallel ${EVAL_TP} \
          --data_parallel ${EVAL_DP} \
          --temperature ${EVAL_TEMPERATURE} \
          --top_p ${EVAL_TOP_P} \
          --max_tokens ${EVAL_MAX_TOKENS} \
          --gpu_memory_utilization 0.85 \
          --output_dir "${EVAL_OUTPUT_DIR}"

    echo ">>> Step 4 complete"
  fi

  # ---- Summary ----
  echo ""
  echo "============================================================"
  echo "  PIPELINE COMPLETE: ${EXP_NAME}"
  echo "============================================================"
  echo ""
  echo "  Pairs:     ${PAIRS_PATH}"
  echo "  Checkpoint: ${CHECKPOINT_DIR}"
  echo "  Eval:       ${EVAL_METRICS}"
  echo ""

  if [ -f "${EVAL_METRICS}" ]; then
    python3 -c "
import json
with open('${EVAL_METRICS}') as f:
    data = json.load(f)
metrics = data.get('metrics', data)
print(f\"{'Benchmark':<25} {'mean@${EVAL_N}':>8} {'pass@1':>8} {'maj@${EVAL_N}':>8}\")
print('-' * 55)
for name, m in metrics.items():
    mean = m.get('mean@${EVAL_N}', m.get('mean@3', 'N/A'))
    p1 = m.get('pass@1', 'N/A')
    maj = m.get('maj@${EVAL_N}', m.get('maj@3', 'N/A'))
    if isinstance(mean, float):
        print(f'{name:<25} {mean:>7.1%} {p1:>7.1%} {maj:>7.1%}')
"
  fi
}

#!/usr/bin/env bash
#
# Full pipeline for Qwen2.5-3B (base model) DPO
#
# Steps:
#   0. Pre-flight checks (Docker images, dependencies, model files)
#   1. Generate preference pairs with --strict  [skips if output exists]
#   2. Clean pairs (re-verify rejected responses) [skips if already cleaned]
#   3. Train DPO with DeepSpeed ZeRO 2 (8 GPUs)  [skips if checkpoint exists]
#   4. Evaluate on 7 math benchmarks (n=3)        [skips if eval exists]
#
# Idempotent: re-running will skip completed steps automatically.
# No separate resume scripts needed.
#
# Usage:
#   bash run_qwen25_3b_base_pipeline.sh 2>&1 | tee run_qwen25_3b_base_pipeline.log
#
set -euo pipefail

# ==================== Configuration ====================
OUTPUT_PREFIX="dpo-qwen25-3b-base"
MODEL="Qwen/Qwen2.5-3B"
CHECKPOINT_DIR="/data-1/checkpoints/qwen25-3b-base-dpo"
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/inference_n3"
TRAIN_SCRIPT="dpo_pipeline/train_dpo_qwen25_3b_base.py"

# Data generation parameters
NUM_ROLLOUTS=16
LIMIT=1200
MAX_TOKENS=4096
TEMPERATURE=0.7
TP_SIZE=8
GPU_MEM_UTIL=0.9

# Evaluation parameters
EVAL_N=3
EVAL_TEMPERATURE=1.0
EVAL_TOP_P=0.95
EVAL_MAX_TOKENS=4096
EVAL_TP=8

# Derived paths
PAIRS_PATH="/data-1/dataset/dpo/${OUTPUT_PREFIX}/${OUTPUT_PREFIX}-pairs.jsonl"
EXTRACTED_PATH="/data-1/dataset/dpo/${OUTPUT_PREFIX}/${OUTPUT_PREFIX}-extracted.jsonl"
TRAINING_SUMMARY="${CHECKPOINT_DIR}/training_logs/training_summary.json"
EVAL_METRICS="${EVAL_OUTPUT_DIR}/eval_metrics.json"

echo "============================================================"
echo "  Qwen2.5-3B DPO Full Pipeline (DeepSpeed ZeRO 2)"
echo "============================================================"
echo ""
echo "  Model:            ${MODEL}"
echo "  Output prefix:    ${OUTPUT_PREFIX}"
echo "  Checkpoint:       ${CHECKPOINT_DIR}"
echo "  Strict filtering: YES"
echo "  Rollouts/prompt:  ${NUM_ROLLOUTS}"
echo "  Prompt limit:     ${LIMIT}"
echo "  TP size:          ${TP_SIZE}"
echo "  Training:         DeepSpeed ZeRO 2, 8 GPUs"
echo ""

# ==================== Step 0: Pre-flight Checks ====================
echo "============================================================"
echo "  STEP 0: Pre-flight checks"
echo "============================================================"
echo ""

PREFLIGHT_OK=true

# Check Docker images exist
for img in dpo-harness verl-harness:latest; do
  if docker image inspect "${img}" &>/dev/null; then
    echo "  [OK] Docker image '${img}' found"
  else
    echo "  [FAIL] Docker image '${img}' not found"
    PREFLIGHT_OK=false
  fi
done

# Check critical dependencies inside dpo-harness
echo ""
echo "  Checking dpo-harness dependencies..."
if docker run --rm dpo-harness python -c "
import trl; print(f'  [OK] trl=={trl.__version__}')
import deepspeed; print(f'  [OK] deepspeed=={deepspeed.__version__}')
import torch; print(f'  [OK] torch=={torch.__version__}, CUDA={torch.cuda.is_available()}')
import accelerate; print(f'  [OK] accelerate=={accelerate.__version__}')
"; then
  true
else
  echo "  [FAIL] Dependency check failed inside dpo-harness"
  PREFLIGHT_OK=false
fi

# Check verl-harness has the eval module accessible
echo ""
echo "  Checking verl-harness eval module..."
if docker run --rm \
  -v /data-1:/data-1 \
  -e PYTHONPATH=/data-1/verl07/verl \
  verl-harness:latest \
  python -c "import verl; print('  [OK] verl module importable')"; then
  true
else
  echo "  [FAIL] verl module not importable in verl-harness"
  PREFLIGHT_OK=false
fi

# Check GPUs available
echo ""
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "${GPU_COUNT}" -ge 8 ]; then
  echo "  [OK] ${GPU_COUNT} GPUs detected"
else
  echo "  [WARN] Only ${GPU_COUNT} GPUs detected (expected 8)"
fi

if [ "${PREFLIGHT_OK}" = false ]; then
  echo ""
  echo "FATAL: Pre-flight checks failed. Fix the issues above before running the pipeline."
  exit 1
fi

echo ""
echo ">>> Pre-flight checks passed"
echo ""

# ==================== Step 1: Generate preference pairs ====================
echo "============================================================"
echo "  STEP 1/4: Generate preference pairs (--strict)"
echo "============================================================"
echo ""

if [ -f "${PAIRS_PATH}" ]; then
  EXISTING_COUNT=$(wc -l < "${PAIRS_PATH}")
  echo ">>> SKIP: Pairs file already exists at ${PAIRS_PATH} (${EXISTING_COUNT} pairs)"
else
  docker run --rm --gpus all --ipc=host \
    -v /data-1:/data-1 \
    -w /data-1/dpo-experiment \
    dpo-harness \
    python dpo_pipeline/run_pipeline.py \
      --model "${MODEL}" \
      --num-rollouts ${NUM_ROLLOUTS} \
      --limit ${LIMIT} \
      --output-prefix "${OUTPUT_PREFIX}" \
      --max-tokens ${MAX_TOKENS} \
      --temperature ${TEMPERATURE} \
      --tensor-parallel-size ${TP_SIZE} \
      --gpu-memory-utilization ${GPU_MEM_UTIL} \
      --strict

  echo ""
  echo ">>> Step 1 complete: preference pairs generated at ${PAIRS_PATH}"
fi

# ==================== Step 2: Clean pairs ====================
echo ""
echo "============================================================"
echo "  STEP 2/4: Clean pairs (re-verify rejected responses)"
echo "============================================================"
echo ""

CLEAN_MARKER="/data-1/dataset/dpo/${OUTPUT_PREFIX}/.cleaned"

if [ -f "${CLEAN_MARKER}" ]; then
  PAIR_COUNT=$(wc -l < "${PAIRS_PATH}")
  echo ">>> SKIP: Pairs already cleaned (${PAIR_COUNT} pairs)"
else
  docker run --rm --gpus all --ipc=host \
    -v /data-1:/data-1 \
    -w /data-1/dpo-experiment \
    dpo-harness \
    python dpo_pipeline/clean_pairs.py \
      --pairs "${PAIRS_PATH}" \
      --extracted "${EXTRACTED_PATH}" \
      --output "${PAIRS_PATH}"

  touch "${CLEAN_MARKER}"
  echo ""
  echo ">>> Step 2 complete: pairs cleaned in-place at ${PAIRS_PATH}"
fi

PAIR_COUNT=$(wc -l < "${PAIRS_PATH}")
echo ">>> Total pairs after cleaning: ${PAIR_COUNT}"

if [ "${PAIR_COUNT}" -lt 1000 ]; then
  echo "ERROR: Only ${PAIR_COUNT} pairs generated. Expected at least 1000."
  echo "Consider increasing --limit or checking model rollout quality."
  exit 1
fi

# ==================== Step 3: DPO Training (DeepSpeed ZeRO 2) ====================
echo ""
echo "============================================================"
echo "  STEP 3/4: DPO Training (DeepSpeed ZeRO 2, 8 GPUs)"
echo "============================================================"
echo ""
echo "  Hyperparameters:"
echo "    beta=0.1, lr=5e-7, epochs=1"
echo "    per_device_batch=1, grad_accum=2, 8 GPUs (effective=16)"
echo "    max_length=2048, warmup=0.1, scheduler=cosine"
echo ""

if [ -f "${TRAINING_SUMMARY}" ]; then
  echo ">>> SKIP: Training already completed (summary at ${TRAINING_SUMMARY})"
else
  docker run --rm --gpus all --ipc=host \
    -v /data-1:/data-1 \
    -w /data-1/dpo-experiment \
    dpo-harness \
    accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
      "${TRAIN_SCRIPT}"

  echo ""
  echo ">>> Step 3 complete: checkpoint saved to ${CHECKPOINT_DIR}"
fi

echo ""

if [ -f "${TRAINING_SUMMARY}" ]; then
  echo "Training summary:"
  python3 -c "
import json
with open('${TRAINING_SUMMARY}') as f:
    s = json.load(f)
r = s['results']
print(f\"  Dataset size:     {s['dataset_size']}\")
print(f\"  DeepSpeed:        {s.get('deepspeed', 'N/A')}\")
print(f\"  Num GPUs:         {s.get('num_gpus', 'N/A')}\")
print(f\"  Final loss:       {r['final_step_loss']}\")
print(f\"  Loss trend:       {r['first_step_loss']:.4f} -> {r['final_step_loss']:.4f}\")
print(f\"  Final margins:    {r['final_step_margins']:.4f}\")
print(f\"  Margin trend:     {r['first_step_margins']:.4f} -> {r['final_step_margins']:.4f}\")
print(f\"  Chosen reward:    {r['final_rewards_chosen']:.4f}\")
print(f\"  Rejected reward:  {r['final_rewards_rejected']:.4f}\")
print(f\"  Runtime:          {r['training_runtime_seconds']:.0f}s\")
"
fi

# ==================== Step 4: Evaluation ====================
echo ""
echo "============================================================"
echo "  STEP 4/4: Evaluation on 7 math benchmarks (n=${EVAL_N})"
echo "============================================================"
echo ""

if [ -f "${EVAL_METRICS}" ]; then
  echo ">>> SKIP: Evaluation already completed (metrics at ${EVAL_METRICS})"
else
  docker run --rm --gpus all --ipc=host \
    -v /data-1:/data-1 \
    -e PYTHONPATH=/data-1/verl07/verl \
    -w /data-1/verl07/verl \
    verl-harness:latest \
    python recipe/joint_training/offline_eval.py \
        --model_path ${CHECKPOINT_DIR} \
        --test_files \
          /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
          /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
          /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
          /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
          /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
          /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
          /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet \
        --n ${EVAL_N} \
        --tensor_parallel ${EVAL_TP} \
        --temperature ${EVAL_TEMPERATURE} \
        --top_p ${EVAL_TOP_P} \
        --max_tokens ${EVAL_MAX_TOKENS} \
        --gpu_memory_utilization 0.85 \
        --output_dir ${EVAL_OUTPUT_DIR}

  echo ""
  echo ">>> Step 4 complete: evaluation results saved to ${EVAL_OUTPUT_DIR}"
fi

echo ""

# ==================== Final Summary ====================
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Artifacts:"
echo "    Preference pairs:  ${PAIRS_PATH}"
echo "    Checkpoint:        ${CHECKPOINT_DIR}"
echo "    Training summary:  ${TRAINING_SUMMARY}"
echo "    Eval metrics:      ${EVAL_METRICS}"
echo "    Eval details:      ${EVAL_OUTPUT_DIR}/eval_details.parquet"
echo ""

if [ -f "${EVAL_METRICS}" ]; then
  echo "Evaluation results:"
  python3 -c "
import json
with open('${EVAL_METRICS}') as f:
    data = json.load(f)
metrics = data.get('metrics', data)
print(f\"{'Benchmark':<30} {'mean@${EVAL_N}':>8} {'pass@1':>8} {'maj@${EVAL_N}':>8}\")
print('-' * 60)
for name, m in metrics.items():
    mean = m.get('mean@${EVAL_N}', m.get('mean@3', 'N/A'))
    p1 = m.get('pass@1', 'N/A')
    maj = m.get('maj@${EVAL_N}', m.get('maj@3', 'N/A'))
    if isinstance(mean, float):
        print(f'{name:<30} {mean:>8.4f} {p1:>8.4f} {maj:>8.4f}')
    else:
        print(f'{name:<30} {mean:>8} {p1:>8} {maj:>8}')
"
fi

echo ""
echo "Done!"

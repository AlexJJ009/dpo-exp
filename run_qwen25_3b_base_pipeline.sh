#!/usr/bin/env bash
#
# Full pipeline for Qwen2.5-3B (base model) DPO
#
# Steps:
#   1. Generate preference pairs with --strict
#   2. Clean pairs (re-verify rejected responses)
#   3. Train DPO with DeepSpeed ZeRO 2 (8 GPUs)
#   4. Evaluate on 7 math benchmarks (n=3)
#
# Usage:
#   bash run_qwen25_3b_base_pipeline.sh
#
set -euo pipefail

# ==================== Configuration ====================
OUTPUT_PREFIX="dpo-qwen25-3b-base"
MODEL="Qwen/Qwen2.5-3B"
CHECKPOINT_DIR="/data-1/checkpoints/qwen25-3b-base-dpo"
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/inference_n3"

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

# ==================== Step 1: Generate preference pairs ====================
echo ""
echo "============================================================"
echo "  STEP 1/4: Generate preference pairs (--strict)"
echo "============================================================"
echo ""

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

# ==================== Step 2: Clean pairs ====================
echo ""
echo "============================================================"
echo "  STEP 2/4: Clean pairs (re-verify rejected responses)"
echo "============================================================"
echo ""

docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  -w /data-1/dpo-experiment \
  dpo-harness \
  python dpo_pipeline/clean_pairs.py \
    --pairs "${PAIRS_PATH}" \
    --extracted "${EXTRACTED_PATH}" \
    --output "${PAIRS_PATH}"

echo ""
echo ">>> Step 2 complete: pairs cleaned in-place at ${PAIRS_PATH}"

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

docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  -w /data-1/dpo-experiment \
  dpo-harness \
  accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
    dpo_pipeline/train_dpo_qwen25_3b_base.py

echo ""
echo ">>> Step 3 complete: checkpoint saved to ${CHECKPOINT_DIR}"
echo ""

if [ -f "${CHECKPOINT_DIR}/training_logs/training_summary.json" ]; then
  echo "Training summary:"
  python3 -c "
import json
with open('${CHECKPOINT_DIR}/training_logs/training_summary.json') as f:
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
echo ""

# ==================== Final Summary ====================
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Artifacts:"
echo "    Preference pairs:  ${PAIRS_PATH}"
echo "    Checkpoint:        ${CHECKPOINT_DIR}"
echo "    Training summary:  ${CHECKPOINT_DIR}/training_logs/training_summary.json"
echo "    Eval metrics:      ${EVAL_OUTPUT_DIR}/eval_metrics.json"
echo "    Eval details:      ${EVAL_OUTPUT_DIR}/eval_details.parquet"
echo ""

if [ -f "${EVAL_OUTPUT_DIR}/eval_metrics.json" ]; then
  echo "Evaluation results:"
  python3 -c "
import json
with open('${EVAL_OUTPUT_DIR}/eval_metrics.json') as f:
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

#!/usr/bin/env bash
#
# Continuation pipeline for Gemma3-4B-SFT DPO
#
# Context:
#   The original pipeline (run_gemma3_4b_sft_pipeline.sh) completed Steps 1-2
#   but produced only 3781 pairs (< 5000 target). Step 3 then failed due to
#   DeepSpeed not being installed in dpo-harness (now fixed).
#
# This script:
#   1. Generate extra preference pairs (offset=1200, new 500 prompts)
#   2. Clean the extra pairs
#   3. Merge extra pairs into existing pairs file (target: 5000+)
#   4. Train DPO with DeepSpeed ZeRO 2 (8 GPUs)
#   5. Evaluate on 7 math benchmarks (n=3)
#
# Usage:
#   bash run_gemma3_4b_sft_continue.sh
#
set -euo pipefail

# ==================== Configuration ====================
OUTPUT_PREFIX="dpo-gemma3-4b-sft"
EXTRA_PREFIX="dpo-gemma3-4b-sft-extra"
MODEL="/data-1/.cache/gemma3-4b-base-sft-stage-1"
CHECKPOINT_DIR="/data-1/checkpoints/gemma3-4b-sft-dpo"
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/inference_n3"

# Data generation parameters (same as original, offset to skip already processed)
EXTRA_OFFSET=1200
EXTRA_LIMIT=500
NUM_ROLLOUTS=16
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

# Paths
PAIRS_PATH="/data-1/dataset/dpo/${OUTPUT_PREFIX}/${OUTPUT_PREFIX}-pairs.jsonl"
EXTRA_PAIRS_PATH="/data-1/dataset/dpo/${EXTRA_PREFIX}/${EXTRA_PREFIX}-pairs.jsonl"
EXTRA_EXTRACTED_PATH="/data-1/dataset/dpo/${EXTRA_PREFIX}/${EXTRA_PREFIX}-extracted.jsonl"

TARGET_PAIRS=5000

echo "============================================================"
echo "  Gemma3-4B-SFT DPO Continuation Pipeline"
echo "============================================================"
echo ""
echo "  Model:            ${MODEL}"
echo "  Existing pairs:   ${PAIRS_PATH}"
echo "  Extra offset:     ${EXTRA_OFFSET} (skip already-processed prompts)"
echo "  Extra limit:      ${EXTRA_LIMIT} new prompts"
echo "  Target pairs:     ${TARGET_PAIRS}+"
echo "  Checkpoint:       ${CHECKPOINT_DIR}"
echo ""

# ==================== Step 1: Generate extra preference pairs ====================
echo "============================================================"
echo "  STEP 1/5: Generate extra preference pairs (offset=${EXTRA_OFFSET})"
echo "============================================================"
echo ""

docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  -w /data-1/dpo-experiment \
  dpo-harness \
  python dpo_pipeline/run_pipeline.py \
    --model "${MODEL}" \
    --num-rollouts ${NUM_ROLLOUTS} \
    --limit ${EXTRA_LIMIT} \
    --offset ${EXTRA_OFFSET} \
    --output-prefix "${EXTRA_PREFIX}" \
    --max-tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --strict

echo ""
echo ">>> Step 1 complete: extra pairs at ${EXTRA_PAIRS_PATH}"

# ==================== Step 2: Clean extra pairs ====================
echo ""
echo "============================================================"
echo "  STEP 2/5: Clean extra pairs"
echo "============================================================"
echo ""

docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  -w /data-1/dpo-experiment \
  dpo-harness \
  python dpo_pipeline/clean_pairs.py \
    --pairs "${EXTRA_PAIRS_PATH}" \
    --extracted "${EXTRA_EXTRACTED_PATH}" \
    --output "${EXTRA_PAIRS_PATH}"

EXTRA_COUNT=$(wc -l < "${EXTRA_PAIRS_PATH}")
echo ""
echo ">>> Step 2 complete: ${EXTRA_COUNT} extra pairs after cleaning"

# ==================== Step 3: Merge pairs ====================
echo ""
echo "============================================================"
echo "  STEP 3/5: Merge extra pairs into main dataset"
echo "============================================================"
echo ""

EXISTING_COUNT=$(wc -l < "${PAIRS_PATH}")
echo "  Existing pairs: ${EXISTING_COUNT}"
echo "  Extra pairs:    ${EXTRA_COUNT}"

cat "${EXTRA_PAIRS_PATH}" >> "${PAIRS_PATH}"

TOTAL_COUNT=$(wc -l < "${PAIRS_PATH}")
echo "  Total pairs:    ${TOTAL_COUNT}"
echo ""

if [ "${TOTAL_COUNT}" -lt "${TARGET_PAIRS}" ]; then
  echo "WARNING: Only ${TOTAL_COUNT} pairs total (target: ${TARGET_PAIRS})."
  echo "Proceeding anyway — increase EXTRA_LIMIT and rerun Step 1-3 if needed."
  echo ""
fi

echo ">>> Step 3 complete: ${TOTAL_COUNT} pairs ready for training"

# ==================== Step 4: DPO Training (DeepSpeed ZeRO 2) ====================
echo ""
echo "============================================================"
echo "  STEP 4/5: DPO Training (DeepSpeed ZeRO 2, 8 GPUs)"
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
    dpo_pipeline/train_dpo_gemma3_4b_sft.py

echo ""
echo ">>> Step 4 complete: checkpoint saved to ${CHECKPOINT_DIR}"
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
print(f\"  Loss trend:       {r['first_step_loss']:.4f} -> {r['final_step_loss']:.4f}\")
print(f\"  Margin trend:     {r['first_step_margins']:.4f} -> {r['final_step_margins']:.4f}\")
print(f\"  Chosen reward:    {r['final_rewards_chosen']:.4f}\")
print(f\"  Rejected reward:  {r['final_rewards_rejected']:.4f}\")
print(f\"  Runtime:          {r['training_runtime_seconds']:.0f}s\")
"
fi

# ==================== Step 5: Evaluation ====================
echo ""
echo "============================================================"
echo "  STEP 5/5: Evaluation on 7 math benchmarks (n=${EVAL_N})"
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
echo ">>> Step 5 complete: evaluation results saved to ${EVAL_OUTPUT_DIR}"
echo ""

# ==================== Final Summary ====================
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Artifacts:"
echo "    Preference pairs:  ${PAIRS_PATH} (${TOTAL_COUNT} pairs)"
echo "    Extra pairs:       ${EXTRA_PAIRS_PATH}"
echo "    Checkpoint:        ${CHECKPOINT_DIR}"
echo "    Training summary:  ${CHECKPOINT_DIR}/training_logs/training_summary.json"
echo "    Eval metrics:      ${EVAL_OUTPUT_DIR}/eval_metrics.json"
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

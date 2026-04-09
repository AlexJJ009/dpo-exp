#!/usr/bin/env bash
#
# Gemma3-4B-SFT DPO pipeline (training + eval only)
#
# Dataset already merged on Server A and synced here:
#   /data-1/dataset/dpo/dpo-gemma3-4b-sft/dpo-gemma3-4b-sft-pairs.jsonl (5202 pairs)
#
# DO NOT run run_gemma3_4b_sft_continue.sh — it would regenerate extra pairs
# and overwrite the already-merged dataset.
#
# Idempotent: re-running will skip completed steps automatically.
#
# Usage:
#   bash run_gemma3_4b_sft_resume.sh 2>&1 | tee run_gemma3_4b_sft_resume.log
#
set -euo pipefail

CHECKPOINT_DIR="/data-1/checkpoints/gemma3-4b-sft-dpo"
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/inference_n3"
PAIRS_PATH="/data-1/dataset/dpo/dpo-gemma3-4b-sft/dpo-gemma3-4b-sft-pairs.jsonl"
TRAIN_SCRIPT="dpo_pipeline/train_dpo_gemma3_4b_sft.py"
TRAINING_SUMMARY="${CHECKPOINT_DIR}/training_logs/training_summary.json"
EVAL_METRICS="${EVAL_OUTPUT_DIR}/eval_metrics.json"
EVAL_N=3
EVAL_TEMPERATURE=1.0
EVAL_TOP_P=0.95
EVAL_MAX_TOKENS=4096
EVAL_TP=8

PAIR_COUNT=$(wc -l < "${PAIRS_PATH}")
echo "============================================================"
echo "  Gemma3-4B-SFT DPO Pipeline (Training + Eval)"
echo "============================================================"
echo ""
echo "  Dataset:          ${PAIRS_PATH}"
echo "  Pairs:            ${PAIR_COUNT}"
echo "  Checkpoint:       ${CHECKPOINT_DIR}"
echo ""

# ==================== Pre-flight Checks ====================
echo "============================================================"
echo "  Pre-flight checks"
echo "============================================================"
echo ""

PREFLIGHT_OK=true

for img in dpo-harness verl-harness:latest; do
  if docker image inspect "${img}" &>/dev/null; then
    echo "  [OK] Docker image '${img}' found"
  else
    echo "  [FAIL] Docker image '${img}' not found"
    PREFLIGHT_OK=false
  fi
done

echo ""
echo "  Checking dpo-harness dependencies..."
if docker run --rm dpo-harness python -c "
import trl; print(f'  [OK] trl=={trl.__version__}')
import deepspeed; print(f'  [OK] deepspeed=={deepspeed.__version__}')
import torch; print(f'  [OK] torch=={torch.__version__}, CUDA={torch.cuda.is_available()}')
"; then
  true
else
  echo "  [FAIL] Dependency check failed inside dpo-harness"
  PREFLIGHT_OK=false
fi

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

if [ "${PREFLIGHT_OK}" = false ]; then
  echo ""
  echo "FATAL: Pre-flight checks failed. Fix the issues above before running."
  exit 1
fi

echo ""
echo ">>> Pre-flight checks passed"
echo ""

# ==================== DPO Training (DeepSpeed ZeRO 2) ====================
echo "============================================================"
echo "  STEP 1/2: DPO Training (DeepSpeed ZeRO 2, 8 GPUs)"
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
  echo ">>> Step 1 complete: checkpoint saved to ${CHECKPOINT_DIR}"
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

# ==================== Evaluation ====================
echo ""
echo "============================================================"
echo "  STEP 2/2: Evaluation on 7 math benchmarks (n=${EVAL_N})"
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
  echo ">>> Step 2 complete: evaluation results saved to ${EVAL_OUTPUT_DIR}"
fi

echo ""

# ==================== Final Summary ====================
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Artifacts:"
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

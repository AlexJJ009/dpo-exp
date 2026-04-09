#!/usr/bin/env bash
# Qwen3-8B Med-SFT + ScienceQA (cross-domain)
source "$(dirname "$0")/../scripts/config.sh"
MODEL="${CACHE_DIR}/Qwen3-8B-Base-Med-SFT/checkpoint-134"
TRAIN_DATA="${DATASET_DIR}/medmcqa-dpo/medmcqa-train.jsonl"
EXP_NAME="dpo-8b-med-sft-med"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/qwen3-8b-med-sft-dpo-med"
EVAL_TEST_FILES="${DATASET_DIR}/medmcqa-dpo/medmcqa-test.parquet ${DATASET_DIR}/scienceqa-dpo/scienceqa-test.parquet"
# 8B: batch=1, accum=2, 8 GPUs -> effective 16 + memory optimizations
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
DPO_PRECOMPUTE_REF=true
DPO_OPTIM=adafactor
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

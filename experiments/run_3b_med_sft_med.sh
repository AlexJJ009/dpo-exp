#!/usr/bin/env bash
# Qwen2.5-3B Med-SFT + ScienceQA (cross-domain)
source "$(dirname "$0")/../scripts/config.sh"
MODEL="${CACHE_DIR}/Qwen2.5-3B-Med-SFT/checkpoint-134"
TRAIN_DATA="${DATASET_DIR}/medmcqa-dpo/medmcqa-train.jsonl"
EXP_NAME="dpo-3b-med-sft-med"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/qwen25-3b-med-sft-dpo-med"
EVAL_TEST_FILES="${DATASET_DIR}/medmcqa-dpo/medmcqa-test.parquet ${DATASET_DIR}/scienceqa-dpo/scienceqa-test.parquet"
# 3B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

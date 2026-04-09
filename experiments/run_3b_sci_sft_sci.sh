#!/usr/bin/env bash
# Qwen2.5-3B Sci-SFT + MedMCQA (cross-domain)
source "$(dirname "$0")/../scripts/config.sh"
MODEL="${CACHE_DIR}/Qwen2.5-3B-Sci-SFT/checkpoint-109"
TRAIN_DATA="${DATASET_DIR}/scienceqa-dpo/scienceqa-train.jsonl"
EXP_NAME="dpo-3b-sci-sft-sci"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/qwen25-3b-sci-sft-dpo-sci"
EVAL_TEST_FILES="${DATASET_DIR}/medmcqa-dpo/medmcqa-test.parquet ${DATASET_DIR}/scienceqa-dpo/scienceqa-test.parquet"
# 3B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

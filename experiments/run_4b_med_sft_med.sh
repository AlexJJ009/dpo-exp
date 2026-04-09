#!/usr/bin/env bash
# Qwen3-4B Med-SFT + ScienceQA (cross-domain)
MODEL="/data-1/.cache/Qwen3-4B-Base-Med-SFT/checkpoint-134"
TRAIN_DATA="/data-1/dataset/dpo/medmcqa/medmcqa-train.jsonl"
EXP_NAME="dpo-4b-med-sft-med"
CHECKPOINT_DIR="/data-1/checkpoints/qwen3-4b-med-sft-dpo-med"
EVAL_TEST_FILES="/data-1/dataset/dpo/medmcqa/medmcqa-test.parquet /data-1/dataset/dpo/scienceqa/scienceqa-test.parquet"
# 4B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

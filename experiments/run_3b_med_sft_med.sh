#!/usr/bin/env bash
# Qwen2.5-3B Med-SFT + ScienceQA (cross-domain)
MODEL="/data-1/.cache/Qwen2.5-3B-Med-SFT/checkpoint-134"
TRAIN_DATA="/data-1/dataset/medmcqa-dpo/medmcqa-train.jsonl"
EXP_NAME="dpo-3b-med-sft-med"
CHECKPOINT_DIR="/data-1/checkpoints/qwen25-3b-med-sft-dpo-med"
EVAL_TEST_FILES="/data-1/dataset/medmcqa-dpo/medmcqa-test.parquet /data-1/dataset/scienceqa-dpo/scienceqa-test.parquet"
# 3B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

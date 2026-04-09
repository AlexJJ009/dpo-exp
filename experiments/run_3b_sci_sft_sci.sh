#!/usr/bin/env bash
# Qwen2.5-3B Sci-SFT + MedMCQA (cross-domain)
MODEL="/data-1/.cache/Qwen2.5-3B-Sci-SFT/checkpoint-109"
TRAIN_DATA="/data-1/dataset/dpo/scienceqa/scienceqa-train.jsonl"
EXP_NAME="dpo-3b-sci-sft-sci"
CHECKPOINT_DIR="/data-1/checkpoints/qwen25-3b-sci-sft-dpo-sci"
EVAL_TEST_FILES="/data-1/dataset/dpo/medmcqa/medmcqa-test.parquet /data-1/dataset/dpo/scienceqa/scienceqa-test.parquet"
# 3B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

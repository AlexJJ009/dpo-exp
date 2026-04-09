#!/usr/bin/env bash
# Qwen3-8B Med-SFT + ScienceQA (cross-domain)
MODEL="/data-1/.cache/Qwen3-8B-Base-Med-SFT/checkpoint-134"
TRAIN_DATA="/data-1/dataset/medmcqa-dpo/medmcqa-train.jsonl"
EXP_NAME="dpo-8b-med-sft-med"
CHECKPOINT_DIR="/data-1/checkpoints/qwen3-8b-med-sft-dpo-med"
EVAL_TEST_FILES="/data-1/dataset/medmcqa-dpo/medmcqa-test.parquet /data-1/dataset/scienceqa-dpo/scienceqa-test.parquet"
# 8B: batch=1, accum=2, 8 GPUs -> effective 16 + memory optimizations
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
DPO_PRECOMPUTE_REF=true
DPO_OPTIM=adafactor
source "$(dirname "$0")/run_mcq_dpo.sh"
run_pipeline

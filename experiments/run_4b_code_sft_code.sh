#!/usr/bin/env bash
# Qwen3-4B Code-SFT + Code DPO
source "$(dirname "$0")/../scripts/config.sh"
MODEL="${CACHE_DIR}/Qwen3-4B-Base-Code-SFT/checkpoint-38"
TRAIN_DATA="${DATASET_DIR}/code/code-train.jsonl"
EXP_NAME="dpo-4b-code-sft-code"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/qwen3-4b-code-sft-dpo-code"
EVAL_TEST_FILES="${DATASET_DIR}/EnsembleLLM-data-processed/HumanEval/test.jsonl ${DATASET_DIR}/EnsembleLLM-data-processed/MBPP/test.jsonl ${DATASET_DIR}/EnsembleLLM-data-processed/BigCodeBench/test.jsonl ${DATASET_DIR}/EnsembleLLM-data-processed/LiveCodeBench/test.jsonl"
# 4B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
TARGET_PAIRS=5000
source "$(dirname "$0")/run_code_dpo.sh"
run_pipeline

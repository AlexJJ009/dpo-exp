#!/usr/bin/env bash
# Qwen3-4B Code-SFT + Code DPO
MODEL="/data-1/.cache/Qwen3-4B-Base-Code-SFT/checkpoint-38"
TRAIN_DATA="/data-1/dataset/code/code-train.jsonl"
EXP_NAME="dpo-4b-code-sft-code"
CHECKPOINT_DIR="/data-1/checkpoints/qwen3-4b-code-sft-dpo-code"
EVAL_TEST_FILES="/data-1/dataset/EnsembleLLM-data-processed/HumanEval/test.jsonl /data-1/dataset/EnsembleLLM-data-processed/MBPP/test.jsonl /data-1/dataset/EnsembleLLM-data-processed/BigCodeBench/test.jsonl"
# 4B: batch=1, accum=2, 8 GPUs -> effective 16
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2
source "$(dirname "$0")/run_code_dpo.sh"
run_pipeline

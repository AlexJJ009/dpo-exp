#!/usr/bin/env bash
# Smoke test of the Code DPO pipeline on the new dpo_trl image.
#
# Goal: prove the TRL 0.29.0 + DeepSpeed 0.18.9 + vLLM stack baked into the
# image can run the pipeline end-to-end (rollout -> pair build -> DPO train
# -> eval) using the existing run_4b_code_m1 experiment, but capped to tiny
# sample counts so the whole thing finishes in ~10-30 min on 2 GPUs.
#
# Relationship to run_4b_code_m1_dpo.sh: same base model (M1, lr=1e-5, ckpt-39),
# same train data, same chosen/rejected pairing logic — just smaller numbers
# and isolated EXP_NAME so rollout/pair files never collide with real runs.
#
# Prereqs on the target container (all under lgx/ — paths auto-detected by
# scripts/config.sh from the repo location):
#   lgx/dpo-exp/                                           (this repo)
#   lgx/checkpoints/qwen3-4b-base-code-sft-m1/             (SFT-M1 base)
#   lgx/dataset/code/code-train.jsonl                      (train prompts)
#   lgx/dataset/EnsembleLLM-data-processed/HumanEval/test.jsonl   (eval)
#
# If any is missing, run_code_dpo.sh preflight fails loudly and exits non-zero
# — that's the correct behavior: smoke is meaningful only with real assets.

source "$(dirname "$0")/../scripts/config.sh"

MODEL="${CHECKPOINT_BASE}/qwen3-4b-base-code-sft-m1"
TRAIN_DATA="${DATASET_DIR}/code/code-train.jsonl"
EXP_NAME="dpo-4b-code-m1-smoke"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/qwen3-4b-code-m1-dpo-code-smoke"
EVAL_TEST_FILES="${DATASET_DIR}/EnsembleLLM-data-processed/HumanEval/test.jsonl"

# Fewer GPUs (2) → easier to schedule. Tiny sample counts → fast validation.
# These env vars are consumed by run_code_dpo.sh defaults.
export DP_SIZE=2
export TP_SIZE=1
export GPU_MEM_UTIL=0.85
export TARGET_PAIRS=200
export ROLLOUT_BATCH=100
export MAX_ROLLOUT_BATCHES=2
export DPO_MAX_PAIRS=100
export EVAL_TP=2
export EVAL_REPEAT=1
# Only run humaneval during smoke — the other benchmarks' test.jsonls may not
# have been synced to the container yet and BCB/LCB eval paths hardcode DATA_DIR.
export EVAL_BENCHMARKS="humaneval"

# 2 GPU × per_device=1 × grad_accum=2 → effective batch=4 (smoke-appropriate).
DPO_PER_DEVICE_BATCH=1
DPO_GRAD_ACCUM=2

source "$(dirname "$0")/run_code_dpo.sh"
run_pipeline

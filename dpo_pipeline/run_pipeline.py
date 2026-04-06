#!/usr/bin/env python3
"""
End-to-end DPO preference data pipeline.

Orchestrates:
  1. Prompt extraction from source dataset
  2. Batch rollout generation with vLLM
  3. Answer verification and preference pair assembly

Usage:
    python run_pipeline.py \
        --model Qwen/Qwen3-4B-Base \
        --num-rollouts 16 \
        --limit 100 \
        --output-prefix dpo-4b

    For a small-scale test:
    python run_pipeline.py \
        --model Qwen/Qwen3-4B-Base \
        --num-rollouts 2 \
        --limit 10 \
        --output-prefix dpo-test
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "/data-1/dataset"
DPO_DIR = os.path.join(DATASET_DIR, "dpo")
SOURCE_DATA = os.path.join(DATASET_DIR, "EnsembleLLM-data", "am_deepseek_r1_filtered_ad.jsonl")


def run_step(description: str, cmd: list[str]):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: Step failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="End-to-end DPO preference data pipeline")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base", help="Model for rollout")
    parser.add_argument("--num-rollouts", type=int, default=16, help="Rollouts per prompt")
    parser.add_argument("--limit", type=int, default=None, help="Max prompts to process")
    parser.add_argument("--output-prefix", default="dpo-4b", help="Prefix for output files")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per rollout")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU mem util")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict rejected response filtering (think tags, answer completeness)")
    parser.add_argument("--offset", type=int, default=None,
                        help="Offset for prompt extraction (skip N valid samples)")
    parser.add_argument("--chat-template", action="store_true",
                        help="Use model's tokenizer chat template for prompt formatting (for SFT/chat models)")
    args = parser.parse_args()

    output_dir = os.path.join(DPO_DIR, args.output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    extracted_path = os.path.join(output_dir, f"{args.output_prefix}-extracted.jsonl")
    rollouts_path = os.path.join(output_dir, f"{args.output_prefix}-rollouts.jsonl")
    pairs_path = os.path.join(output_dir, f"{args.output_prefix}-pairs.jsonl")

    start = time.time()

    # Step 1: Extract prompts
    if args.offset is not None:
        extract_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "extract_prompts_offset.py"),
            "--input", SOURCE_DATA,
            "--output", extracted_path,
            "--offset", str(args.offset),
        ]
    else:
        extract_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "extract_prompts.py"),
            "--input", SOURCE_DATA,
            "--output", extracted_path,
        ]
    if args.limit is not None:
        extract_cmd.extend(["--limit", str(args.limit)])
    run_step("Extract prompts from source dataset", extract_cmd)

    # Step 2: Batch rollout
    rollout_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "batch_rollout.py"),
        "--input", extracted_path,
        "--output", rollouts_path,
        "--model", args.model,
        "--num-rollouts", str(args.num_rollouts),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if args.chat_template:
        rollout_cmd.append("--chat-template")
    run_step("Generate rollouts with vLLM", rollout_cmd)

    # Step 3: Build preference pairs
    pairs_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "build_pairs.py"),
        "--input", rollouts_path,
        "--output", pairs_path,
    ]
    if args.strict:
        pairs_cmd.append("--strict")
    run_step("Build preference pairs", pairs_cmd)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Extracted prompts: {extracted_path}")
    print(f"  Rollouts:          {rollouts_path}")
    print(f"  Preference pairs:  {pairs_path}")
    print(f"{'='*60}")

    # Print summary stats
    pair_count = 0
    with open(pairs_path, "r") as f:
        for line in f:
            if line.strip():
                pair_count += 1
    print(f"\nTotal preference pairs generated: {pair_count}")


if __name__ == "__main__":
    main()

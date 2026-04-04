#!/usr/bin/env python3
"""
Build DPO preference pairs from rollout results.

Reads rollout outputs, verifies each rollout against the reference answer
using dual-layer verification, and assembles preference pairs where:
  - chosen = original dataset response (verified correct)
  - rejected = rollout response (verified incorrect)

Output format is compatible with TRL's DPOTrainer:
  {"prompt": [...messages...], "chosen": [...messages...], "rejected": [...messages...]}

Usage:
    python build_pairs.py \
        --input /data-1/dataset/rollouts.jsonl \
        --output /data-1/dataset/dpo-4b-pairs.jsonl
"""

import argparse
import json
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_verify import verify_answer


def build_preference_pairs(rollouts_path: str, output_path: str) -> dict:
    """
    Build preference pairs from rollout results.

    Returns statistics dict.
    """
    stats = {
        "total_prompts": 0,
        "total_rollouts": 0,
        "rollouts_correct": 0,
        "rollouts_incorrect": 0,
        "rollouts_no_answer": 0,
        "pairs_generated": 0,
        "prompts_with_pairs": 0,
    }

    seen_pairs = set()

    with open(rollouts_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            stats["total_prompts"] += 1

            prompt_messages = sample["prompt"]
            reference_answer = sample["reference_answer"]
            chosen_response = sample["chosen"]
            rollouts = sample["rollouts"]

            prompt_had_pair = False

            for rollout in rollouts:
                stats["total_rollouts"] += 1

                result = verify_answer(rollout, reference_answer)

                if result["correct"]:
                    stats["rollouts_correct"] += 1
                    continue  # Skip correct rollouts - we want incorrect ones as rejected

                if result["extracted_answer"] is None:
                    stats["rollouts_no_answer"] += 1
                    # Still use as rejected - no answer is still wrong

                stats["rollouts_incorrect"] += 1

                # Build the preference pair in TRL chat format
                # prompt: the conversation up to the assistant turn
                # chosen: the assistant's correct response (from dataset)
                # rejected: the rollout's incorrect response

                # Create chosen and rejected as assistant message lists
                chosen_messages = [{"role": "assistant", "content": chosen_response}]
                rejected_messages = [{"role": "assistant", "content": rollout}]

                # Dedup check
                pair_key = (
                    json.dumps(prompt_messages, sort_keys=True),
                    chosen_response[:200],
                    rollout[:200],
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                pair = {
                    "prompt": prompt_messages,
                    "chosen": chosen_messages,
                    "rejected": rejected_messages,
                }

                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                stats["pairs_generated"] += 1
                prompt_had_pair = True

            if prompt_had_pair:
                stats["prompts_with_pairs"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument("--input", required=True, help="Path to rollouts JSONL")
    parser.add_argument("--output", required=True, help="Path to output pairs JSONL")
    args = parser.parse_args()

    stats = build_preference_pairs(args.input, args.output)

    print("=== Preference Pair Generation Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()

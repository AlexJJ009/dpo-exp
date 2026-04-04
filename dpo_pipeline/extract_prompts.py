#!/usr/bin/env python3
"""
Extract prompts and reference answers from the source dataset.

Reads the EnsembleLLM JSONL dataset, extracts:
  - user prompt (math problem)
  - reference answer (ground truth)
  - chosen response (assistant's full chain-of-thought response)

Applies a chat template matching the AIME-2025 format:
  system message + user message with the math problem.

Usage:
    python extract_prompts.py \
        --input /data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl \
        --output /data-1/dataset/extracted_prompts.jsonl \
        --limit 100
"""

import argparse
import json
import sys

SYSTEM_PROMPT = (
    "You are a helpful assistant. To answer the user's question, you first think "
    "about the reasoning process and then provide the user with the answer. The "
    "reasoning process and answer are enclosed within <think> and <answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer "
    "here </answer>."
)

USER_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def extract_from_sample(sample: dict) -> dict | None:
    """Extract prompt, reference_answer, and chosen response from a dataset sample."""
    messages = sample.get("messages", [])
    if len(messages) < 2:
        return None

    user_msg = messages[0]
    assistant_msg = messages[1]

    if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
        return None

    user_content = user_msg.get("content", "").strip()
    if not user_content:
        return None

    # Get reference answer from user message info
    info = user_msg.get("info", {})
    reference_answer = info.get("reference_answer", "")
    if not reference_answer or reference_answer.strip() == "":
        return None

    # Get the chosen (assistant) response
    chosen_response = assistant_msg.get("content", "").strip()
    if not chosen_response:
        return None

    # Build chat template matching AIME-2025 format
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content + USER_SUFFIX},
    ]

    return {
        "prompt": prompt,
        "reference_answer": reference_answer.strip(),
        "chosen": chosen_response,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract prompts from EnsembleLLM dataset")
    parser.add_argument(
        "--input",
        default="/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl",
        help="Path to source JSONL file",
    )
    parser.add_argument(
        "--output",
        default="/data-1/dataset/extracted_prompts.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to process (default: all)",
    )
    args = parser.parse_args()

    count = 0
    skipped = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line_num, line in enumerate(fin):
            if args.limit is not None and count >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = extract_from_sample(sample)
            if result is None:
                skipped += 1
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    print(f"Extracted {count} prompts, skipped {skipped} samples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

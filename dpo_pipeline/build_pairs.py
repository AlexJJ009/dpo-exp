#!/usr/bin/env python3
"""
Build DPO preference pairs from rollout results.

Reads rollout outputs, verifies each rollout against the reference answer
using dual-layer verification, and assembles preference pairs where:
  - chosen = original dataset response (verified correct)
  - rejected = rollout response (verified incorrect)

Strict filtering mode (--strict) enforces three conditions on rejected responses:
  1. Complete <think>...</think> tags with non-empty reasoning
  2. Complete, extractable answer (\boxed{...} or <answer>...</answer>)
  3. Answer verified as incorrect by dual-layer verification

Output format is compatible with TRL's DPOTrainer:
  {"prompt": [...messages...], "chosen": [...messages...], "rejected": [...messages...]}

Usage:
    python build_pairs.py \
        --input /data-1/dataset/dpo/dpo-8b/dpo-8b-rollouts.jsonl \
        --output /data-1/dataset/dpo/dpo-8b/dpo-8b-pairs.jsonl \
        --strict
"""

import argparse
import json
import re
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_verify import (verify_answer, verify_code_answer, extract_python_code,
                           extract_boxed, extract_answer_tag, extract_mcq_letter)


def normalize_think_tags(text: str) -> str:
    """
    Normalize response format to ensure proper <think>...</think> tags.

    Base models often produce responses without </think>. Two patterns:
    1. <think>...<answer> — insert </think> before <answer>
    2. <think>...\boxed{...} (no <answer> or </think>) — insert </think> before
       the last \boxed{} occurrence to close the think block
    """
    if "<think>" in text and "</think>" not in text:
        if "<answer>" in text:
            text = text.replace("<answer>", "</think>\n<answer>", 1)
        else:
            # Find the last \boxed{...} and insert </think> before it
            matches = list(re.finditer(r"\\boxed\s*\{", text))
            if matches:
                insert_pos = matches[-1].start()
                text = text[:insert_pos] + "</think>\n" + text[insert_pos:]
    return text


def has_complete_think_tags(text: str) -> bool:
    """Check if response has complete <think>...</think> tags with non-empty content."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match is None:
        return False
    content = match.group(1).strip()
    return len(content) > 0


def has_complete_answer(text: str, code_mode: bool = False) -> bool:
    """Check if response has a complete, extractable answer."""
    if code_mode:
        code = extract_python_code(text)
        return code is not None and len(code.strip()) > 0

    answer = extract_boxed(text)
    if answer is not None and answer.strip():
        return True
    answer = extract_answer_tag(text)
    if answer is not None and answer.strip():
        return True
    answer = extract_mcq_letter(text)
    if answer is not None:
        return True
    return False


_MCQ_FORMAT_RE = re.compile(r"[Tt]he\s+answer\s+is\s*\(?([A-Ea-e])\)?")


def _has_mcq_format(text: str) -> bool:
    """Check if response contains 'The answer is (X).' pattern."""
    return _MCQ_FORMAT_RE.search(text) is not None


def build_preference_pairs(rollouts_path: str, output_path: str, strict: bool = False,
                           skip_think_filter: bool = False, append: bool = False,
                           code_mode: bool = False) -> dict:
    """
    Build preference pairs from rollout results.

    Args:
        strict: If True, enforce structural completeness filters on rejected responses.
        code_mode: If True, use code execution verification instead of math/MCQ.

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

    if strict:
        stats["filtered_no_think_tags"] = 0
        stats["filtered_no_complete_answer"] = 0
        stats["filtered_no_mcq_format"] = 0
        stats["filtered_answer_not_verified_incorrect"] = 0

    seen_pairs = set()

    # In append mode, load existing pairs for dedup and count
    existing_pairs = 0
    if append and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_pairs += 1
                pair = json.loads(line)
                pair_key = (
                    json.dumps(pair["prompt"], sort_keys=True),
                    pair["chosen"][0]["content"][:200],
                    pair["rejected"][0]["content"][:200],
                )
                seen_pairs.add(pair_key)
        print(f"  Append mode: loaded {existing_pairs} existing pairs for dedup")

    file_mode = "a" if append else "w"
    with open(rollouts_path, "r") as fin, open(output_path, file_mode) as fout:
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
            # Code-mode fields (present only in code datasets)
            test_case = sample.get("test_case")
            source = sample.get("source", "unknown")

            prompt_had_pair = False

            for rollout in rollouts:
                stats["total_rollouts"] += 1

                if code_mode and test_case:
                    # ---- Code verification path ----
                    # Filter: must contain extractable code
                    if not has_complete_answer(rollout, code_mode=True):
                        stats["filtered_no_complete_answer"] += 1
                        continue

                    result = verify_code_answer(rollout, test_case, source)
                    if result["correct"]:
                        stats["rollouts_correct"] += 1
                        continue
                    if result["extracted_answer"] is None:
                        stats["rollouts_no_answer"] += 1
                        continue
                    stats["rollouts_incorrect"] += 1

                elif strict:
                    if not skip_think_filter:
                        # Normalize think tags: insert </think> before <answer>
                        # when base model produces <think>...<answer> without </think>
                        rollout = normalize_think_tags(rollout)

                        # Filter 1: Complete <think>...</think> tags
                        if not has_complete_think_tags(rollout):
                            stats["filtered_no_think_tags"] += 1
                            continue

                    # Filter 2: Complete, extractable answer
                    #   For MCQ: require "The answer is (X)." format specifically
                    if not has_complete_answer(rollout):
                        stats["filtered_no_complete_answer"] += 1
                        continue
                    if skip_think_filter and not _has_mcq_format(rollout):
                        stats["filtered_no_mcq_format"] += 1
                        continue

                    # Filter 3: Answer verified as incorrect
                    result = verify_answer(rollout, reference_answer)
                    if result["correct"]:
                        stats["rollouts_correct"] += 1
                        continue
                    # With strict mode, we require extracted_answer to be non-None
                    # (already ensured by filter 2, but double-check via verification)
                    if result["extracted_answer"] is None:
                        stats["filtered_answer_not_verified_incorrect"] += 1
                        continue

                    stats["rollouts_incorrect"] += 1
                else:
                    # Legacy mode (used for 4B)
                    result = verify_answer(rollout, reference_answer)

                    if result["correct"]:
                        stats["rollouts_correct"] += 1
                        continue

                    if result["extracted_answer"] is None:
                        stats["rollouts_no_answer"] += 1

                    stats["rollouts_incorrect"] += 1

                # Build the preference pair in TRL chat format
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

    stats["existing_pairs"] = existing_pairs
    stats["total_pairs"] = existing_pairs + stats["pairs_generated"]
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument("--input", required=True, help="Path to rollouts JSONL")
    parser.add_argument("--output", required=True, help="Path to output pairs JSONL")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict filtering: require complete think tags, "
                             "extractable answer, and verified-incorrect answer")
    parser.add_argument("--skip-think-filter", action="store_true",
                        help="In strict mode, skip the think tag completeness check "
                             "(for MCQ responses without <think> tags)")
    parser.add_argument("--append", action="store_true",
                        help="Append new pairs to existing output file (with dedup)")
    parser.add_argument("--code", action="store_true",
                        help="Code mode: verify rollouts by executing code against "
                             "test cases instead of math/MCQ verification")
    args = parser.parse_args()

    stats = build_preference_pairs(args.input, args.output, strict=args.strict,
                                   skip_think_filter=args.skip_think_filter,
                                   append=args.append, code_mode=args.code)

    print("=== Preference Pair Generation Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()

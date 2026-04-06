#!/usr/bin/env python3
"""
Post-processing cleanup for DPO preference pairs.

Re-verifies all rejected responses using both math_verify AND regex independently.
If EITHER layer says the rejected response is correct, the pair is removed.

This catches cases where math_verify gives a false negative (says incorrect when
the answer is actually correct) by also checking regex, and vice versa.

Usage:
    python clean_pairs.py \
        --pairs /data-1/dataset/dpo/dpo-4b/dpo-4b-pairs.jsonl \
        --extracted /data-1/dataset/dpo/dpo-4b/dpo-4b-extracted.jsonl \
        --output /data-1/dataset/dpo/dpo-4b/dpo-4b-pairs.jsonl
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_verify import verify_with_math_verify, verify_with_regex


def main():
    parser = argparse.ArgumentParser(description="Clean DPO pairs by re-verifying rejected responses")
    parser.add_argument("--pairs", required=True, help="Path to pairs JSONL")
    parser.add_argument("--extracted", required=True, help="Path to extracted prompts JSONL")
    parser.add_argument("--output", required=True, help="Path to output cleaned pairs JSONL")
    args = parser.parse_args()

    # Build lookup: prompt_key -> reference_answer
    ref_lookup = {}
    with open(args.extracted, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = json.dumps(d["prompt"], sort_keys=True)
            ref_lookup[key] = d["reference_answer"]

    # Load and re-verify pairs
    total = 0
    removed = 0
    kept = 0
    removed_by_math = 0
    removed_by_regex = 0

    with open(args.pairs, "r") as fin, open(args.output + ".tmp", "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            pair = json.loads(line)

            prompt_key = json.dumps(pair["prompt"], sort_keys=True)
            ref_answer = ref_lookup.get(prompt_key)
            if ref_answer is None:
                # Can't verify, keep the pair
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                kept += 1
                continue

            rejected_content = pair["rejected"][0]["content"]

            # Check with math_verify
            math_result = verify_with_math_verify(rejected_content, ref_answer)
            if math_result is True:
                removed += 1
                removed_by_math += 1
                continue

            # Check with regex
            regex_correct, _ = verify_with_regex(rejected_content, ref_answer)
            if regex_correct:
                removed += 1
                removed_by_regex += 1
                continue

            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            kept += 1

    # Replace original with cleaned
    os.replace(args.output + ".tmp", args.output)

    print("=== Pair Cleaning Statistics ===")
    print(f"  Total pairs input: {total}")
    print(f"  Removed (rejected actually correct): {removed}")
    print(f"    - by math_verify: {removed_by_math}")
    print(f"    - by regex: {removed_by_regex}")
    print(f"  Kept: {kept}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract prompts with offset support for supplementary batches.

Usage:
    python extract_prompts_offset.py \
        --input /data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl \
        --output /data-1/dataset/dpo-4b-extra-extracted.jsonl \
        --offset 1000 --limit 200
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_prompts import extract_from_sample


def main():
    parser = argparse.ArgumentParser(description="Extract prompts with offset")
    parser.add_argument("--input", default="/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0, help="Number of valid samples to skip")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to extract after offset")
    args = parser.parse_args()

    count = 0
    skipped_invalid = 0
    skipped_offset = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            result = extract_from_sample(sample)
            if result is None:
                skipped_invalid += 1
                continue

            if skipped_offset < args.offset:
                skipped_offset += 1
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if args.limit is not None and count >= args.limit:
                break

    print(f"Extracted {count} prompts (offset={args.offset}, skipped_invalid={skipped_invalid})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

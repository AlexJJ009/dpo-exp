#!/usr/bin/env python3
"""
Prepare MCQ datasets (MedMCQA, ScienceQA) into a unified format
compatible with the DPO pipeline.

Unified train format (one JSONL record per sample):
  {
    "prompt": [{"role": "user", "content": "..."}],
    "reference_answer": "A",          # single letter
    "chosen": "reasoning...\nThe answer is (A)."  # assistant response
  }

Unified test format (parquet, compatible with offline_eval.py):
  columns: question, answer, task

Filters:
  - MedMCQA: only keeps 'medmcqa' task (drops 'medical_o1' which lacks extractable answers)
  - ScienceQA: keeps all records (all have explicit answer field)

Usage:
    python prepare_mcq_dataset.py --dataset medmcqa
    python prepare_mcq_dataset.py --dataset scienceqa
    python prepare_mcq_dataset.py --dataset all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path("/data-1/dataset/EnsembleLLM-data-processed")
OUTPUT_DIR = Path("/data-1/dataset/dpo")

# Format instruction appended to every user prompt so model outputs
# a structured final answer that the pipeline can reliably extract.
MCQ_FORMAT_SUFFIX = (
    '\n\nPlease reason step by step, and put your final answer within '
    '"The answer is (X)." where X is the correct letter choice.'
)

DATASETS = {
    "medmcqa": {
        "source_dir": SOURCE_DIR / "MedMCQA",
        "output_prefix": "medmcqa",
    },
    "scienceqa": {
        "source_dir": SOURCE_DIR / "ScienceQA",
        "output_prefix": "scienceqa",
    },
}


def extract_answer_letter(text: str) -> str | None:
    """Extract the answer letter from an assistant response."""
    # Pattern: "The answer is (X)." or "The answer is X."
    match = re.search(r"[Tt]he\s+answer\s+is\s*\(?([A-Ea-e])\)?", text[-300:])
    if match:
        return match.group(1).upper()
    return None


def process_medmcqa(source_dir: Path, output_dir: Path) -> dict:
    """Process MedMCQA: filter to 'medmcqa' task only, extract answer letters."""
    train_path = source_dir / "train.jsonl"
    test_path = source_dir / "test.jsonl"
    out_train = output_dir / "medmcqa-train.jsonl"
    out_test = output_dir / "medmcqa-test.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "skipped_task": 0, "skipped_no_answer": 0}

    with open(train_path) as fin, open(out_train, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            stats["total"] += 1

            # Only keep 'medmcqa' task (drop 'medical_o1')
            if sample.get("task") != "medmcqa":
                stats["skipped_task"] += 1
                continue

            messages = sample["messages"]
            user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]

            # Extract ground truth letter from assistant response
            answer = extract_answer_letter(assistant_content)
            if answer is None:
                stats["skipped_no_answer"] += 1
                continue

            record = {
                "prompt": [{"role": "user", "content": user_content + MCQ_FORMAT_SUFFIX}],
                "reference_answer": answer,
                "chosen": assistant_content,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    # Process test set
    test_records = []
    with open(test_path) as fin, open(out_test, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            record = {
                "question": sample["input"] + MCQ_FORMAT_SUFFIX,
                "answer": sample["target"],
                "task": sample.get("task", "medmcqa"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            test_records.append(record)

    # Generate parquet test file (for offline_eval.py)
    out_parquet = output_dir / "medmcqa-test.parquet"
    parquet_rows = []
    for rec in test_records:
        parquet_rows.append({
            "data_source": "medmcqa",
            "ability": "mcq",
            "reward_model": {"ground_truth": rec["answer"], "style": "rule"},
            "prompt": [{"role": "user", "content": rec["question"]}],
            "split": "test",
            "extra_info": {"source": "medmcqa"},
        })
    pd.DataFrame(parquet_rows).to_parquet(str(out_parquet))
    stats["test_parquet"] = str(out_parquet)

    stats["test_count"] = len(test_records)
    return stats


def process_scienceqa(source_dir: Path, output_dir: Path) -> dict:
    """Process ScienceQA: all records have explicit answer field."""
    train_path = source_dir / "train.jsonl"
    test_path = source_dir / "test.jsonl"
    out_train = output_dir / "scienceqa-train.jsonl"
    out_test = output_dir / "scienceqa-test.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "skipped_no_answer": 0}

    with open(train_path) as fin, open(out_train, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            stats["total"] += 1

            messages = sample["messages"]
            user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]
            answer = sample.get("answer", "").strip().upper()

            if not answer or not re.match(r"^[A-E]$", answer):
                stats["skipped_no_answer"] += 1
                continue

            record = {
                "prompt": [{"role": "user", "content": user_content + MCQ_FORMAT_SUFFIX}],
                "reference_answer": answer,
                "chosen": assistant_content,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    # Process test set
    test_records = []
    with open(test_path) as fin, open(out_test, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            record = {
                "question": sample["input"] + MCQ_FORMAT_SUFFIX,
                "answer": sample["target"],
                "task": sample.get("task", "scienceqa"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            test_records.append(record)

    # Generate parquet test file (for offline_eval.py)
    out_parquet = output_dir / "scienceqa-test.parquet"
    parquet_rows = []
    for rec in test_records:
        parquet_rows.append({
            "data_source": "scienceqa",
            "ability": "mcq",
            "reward_model": {"ground_truth": rec["answer"], "style": "rule"},
            "prompt": [{"role": "user", "content": rec["question"]}],
            "split": "test",
            "extra_info": {"source": "scienceqa"},
        })
    pd.DataFrame(parquet_rows).to_parquet(str(out_parquet))
    stats["test_parquet"] = str(out_parquet)

    stats["test_count"] = len(test_records)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare MCQ datasets for DPO pipeline")
    parser.add_argument(
        "--dataset",
        choices=["medmcqa", "scienceqa", "all"],
        default="all",
        help="Which dataset to process",
    )
    args = parser.parse_args()

    targets = ["medmcqa", "scienceqa"] if args.dataset == "all" else [args.dataset]

    for name in targets:
        cfg = DATASETS[name]
        source_dir = cfg["source_dir"]
        output_dir = OUTPUT_DIR / cfg["output_prefix"]

        print(f"\n{'='*60}")
        print(f"  Processing: {name}")
        print(f"  Source: {source_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

        if name == "medmcqa":
            stats = process_medmcqa(source_dir, output_dir)
        else:
            stats = process_scienceqa(source_dir, output_dir)

        print(f"\n  Train statistics:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

        # Verify output
        out_train = output_dir / f"{cfg['output_prefix']}-train.jsonl"
        with open(out_train) as f:
            first = json.loads(f.readline())
        print(f"\n  Output fields: {list(first.keys())}")
        print(f"  Sample reference_answer: {first['reference_answer']}")
        print(f"  Sample prompt (first 100 chars): {first['prompt'][0]['content'][:100]}...")
        print(f"  Output: {out_train}")


if __name__ == "__main__":
    main()

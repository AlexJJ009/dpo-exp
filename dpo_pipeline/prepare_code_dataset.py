#!/usr/bin/env python3
"""
Prepare code datasets for the DPO pipeline.

Converts the AM-DeepSeekR1-Code training data into the unified format
expected by batch_rollout.py / build_pairs.py:

  {
    "prompt": [{"role": "user", "content": "..."}],
    "reference_answer": "<verified Python code>",
    "chosen": "<full assistant response with reasoning + code>",
    "test_case": "<JSON string of test cases>",
    "source": "codeio"
  }

Also generates test parquet files from HumanEval / MBPP / BigCodeBench /
LiveCodeBench in the offline_eval.py schema:

  columns: data_source, ability, reward_model, prompt, split, extra_info

Usage:
    python prepare_code_dataset.py --dataset code
    python prepare_code_dataset.py --dataset code --limit 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path("/data-1/dataset/EnsembleLLM-data-processed")
OUTPUT_DIR = Path("/data-1/dataset/dpo")

# Format instruction appended to every user prompt
CODE_FORMAT_SUFFIX = (
    "\n\nPlease reason step by step, and put your final solution code "
    "in a Python code block (```python ... ```)."
)


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def process_code_train(source_path: Path, output_dir: Path, limit: int | None = None) -> dict:
    """Process AM-DeepSeekR1-Code into DPO pipeline training format."""
    out_train = output_dir / "code-train.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "skipped_no_code": 0, "skipped_no_test": 0}

    with open(source_path) as fin, open(out_train, "w") as fout:
        for line in fin:
            if limit and stats["kept"] >= limit:
                break
            stats["total"] += 1

            sample = json.loads(line)
            messages = sample.get("messages", [])
            if len(messages) < 2:
                continue

            user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]
            extracted_code = sample.get("extracted_code")
            test_case = sample.get("test_case")
            source = messages[0].get("info", {}).get("source", "unknown")

            if not extracted_code:
                stats["skipped_no_code"] += 1
                continue
            if not test_case:
                stats["skipped_no_test"] += 1
                continue

            record = {
                "prompt": [{"role": "user", "content": user_content + CODE_FORMAT_SUFFIX}],
                "reference_answer": extracted_code,
                "chosen": assistant_content,
                "test_case": test_case if isinstance(test_case, str) else json.dumps(test_case),
                "source": source,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def create_example_json(source_path: Path, output_dir: Path):
    """Create example.json showing the data format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read first record to build example
    with open(source_path) as f:
        sample = json.loads(f.readline())

    messages = sample["messages"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]
    source = messages[0].get("info", {}).get("source", "unknown")

    example = {
        "dataset": "AM-DeepSeekR1-Code",
        "summary": {
            "train_total": 19457,
            "train_sources": {
                "codeio": 15953,
                "OpenCoder": 3409,
                "OpenCoderStage2": 88,
                "prime": 7,
            },
            "train_format": "prompt (messages), reference_answer (code), chosen (assistant response), test_case, source",
            "test_benchmarks": ["HumanEval", "MBPP", "BigCodeBench", "LiveCodeBench"],
            "answer_format": {
                "verification": "Code execution against test cases (subprocess)",
                "extraction": "```python ... ``` code blocks from model responses",
            },
        },
        "train_example": {
            "prompt": [{"role": "user", "content": user_content + CODE_FORMAT_SUFFIX}],
            "reference_answer": sample.get("extracted_code", ""),
            "chosen": assistant_content,
            "test_case": sample.get("test_case", ""),
            "source": source,
        },
    }

    with open(output_dir / "example.json", "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Test data preparation (parquet for offline_eval.py)
# ---------------------------------------------------------------------------

def _build_livecodebench_test_code(public_test_cases) -> str:
    """Convert LiveCodeBench public_test_cases to executable test code."""
    cases = json.loads(public_test_cases) if isinstance(public_test_cases, str) else public_test_cases
    if not cases:
        return ""
    lines = ["import sys", "from io import StringIO", ""]
    for idx, tc in enumerate(cases):
        inp = tc.get("input", "")
        out = tc.get("output", "")
        lines += [
            f"# Test case {idx}",
            f"_si, _so = sys.stdin, sys.stdout",
            f"sys.stdin = StringIO({repr(inp)})",
            f"sys.stdout = _cap = StringIO()",
            f"try:",
            f"    exec(open(__file__).read().split('# === TESTS ===')[0])",
            f"finally:",
            f"    sys.stdin, sys.stdout = _si, _so",
            f"assert _cap.getvalue().strip() == {repr(out.strip())}, "
            f"f'Test {idx} failed'",
            "",
        ]
    return "\n".join(lines)


def process_test_benchmarks() -> dict:
    """Create test parquet files for code benchmarks.

    Each benchmark gets its own directory under OUTPUT_DIR, e.g.:
      /data-1/dataset/dpo/humaneval/humaneval-test.parquet
      /data-1/dataset/dpo/mbpp/mbpp-test.parquet
    """
    stats = {}

    benchmarks = {
        "humaneval": {
            "path": SOURCE_DIR / "HumanEval" / "test.jsonl",
            "data_source": "humaneval",
        },
        "mbpp": {
            "path": SOURCE_DIR / "MBPP" / "test.jsonl",
            "data_source": "mbpp",
        },
        "bigcodebench": {
            "path": SOURCE_DIR / "BigCodeBench" / "test.jsonl",
            "data_source": "bigcodebench",
        },
        "livecodebench": {
            "path": SOURCE_DIR / "LiveCodeBench" / "test.jsonl",
            "data_source": "livecodebench",
        },
    }

    for name, cfg in benchmarks.items():
        path = cfg["path"]
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue

        bench_dir = OUTPUT_DIR / name
        bench_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)

                if name == "livecodebench":
                    prompt_text = ex["question_content"]
                    if ex.get("starter_code"):
                        prompt_text += f"\n\n{ex['starter_code']}"
                    test_code = _build_livecodebench_test_code(ex.get("public_test_cases", "[]"))
                    task_id = ex.get("question_id", "")
                else:
                    prompt_text = ex.get("prompt", "")
                    test_code = ex.get("test_code", "")
                    task_id = str(ex.get("task_id", ""))

                if not prompt_text.strip() or not test_code.strip():
                    continue

                prompt_text += CODE_FORMAT_SUFFIX

                rows.append({
                    "data_source": cfg["data_source"],
                    "ability": "code",
                    "reward_model": {
                        "ground_truth": test_code,
                        "style": "code_execution",
                        "task_id": task_id,
                    },
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "split": "test",
                    "extra_info": {"source": cfg["data_source"], "task_id": task_id},
                })

        out_parquet = bench_dir / f"{name}-test.parquet"
        pd.DataFrame(rows).to_parquet(str(out_parquet))
        stats[name] = {"count": len(rows), "parquet": str(out_parquet)}
        print(f"  [OK] {name}: {len(rows)} tasks -> {out_parquet}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare code datasets for DPO pipeline")
    parser.add_argument("--dataset", choices=["code"], default="code")
    parser.add_argument("--limit", type=int, default=None, help="Max training samples")
    args = parser.parse_args()

    source_path = SOURCE_DIR / "AM-DeepSeekR1-Code" / "am_deepseek_r1_filtered_code_verified.jsonl"
    output_dir = OUTPUT_DIR / "code"

    print(f"\n{'='*60}")
    print(f"  Processing: AM-DeepSeekR1-Code")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # 1. Training data
    print("\n--- Training data ---")
    train_stats = process_code_train(source_path, output_dir, limit=args.limit)
    for k, v in train_stats.items():
        print(f"  {k}: {v}")

    # 2. Example JSON
    create_example_json(source_path, output_dir)
    print(f"\n  example.json written")

    # 3. Verify output
    out_train = output_dir / "code-train.jsonl"
    with open(out_train) as f:
        first = json.loads(f.readline())
    print(f"\n  Output fields: {list(first.keys())}")
    print(f"  Sample source: {first['source']}")
    print(f"  Sample prompt (first 100 chars): {first['prompt'][0]['content'][:100]}...")
    print(f"  Output: {out_train}")

    # 4. Test benchmarks (each in its own directory under OUTPUT_DIR)
    print(f"\n--- Test benchmarks ---")
    test_stats = process_test_benchmarks()

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for code datasets (HumanEval / MBPP) using vLLM.

This mirrors the math evaluator but runs unit tests provided in each sample.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# System prompt for code generation tasks
SYSTEM_PROMPT = (
    "You are a helpful assistant. To answer the user's question, you first think "
    "about the reasoning process and then provide the user with the answer. The "
    "reasoning process and answer are enclosed within <think> and <answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer "
    "here </answer>."
)


# -----------------------------
# vLLM Backend
# -----------------------------
class VLLMBackend:
    def __init__(self, model, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9, seed=None):
        from vllm import LLM, SamplingParams

        self.LLM = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_model_len,
            stop=None,
            seed=seed,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        outs = self.LLM.generate(prompts, self.sp)
        return [o.outputs[0].text for o in outs]


# -----------------------------
# Prompt helper
# -----------------------------
def apply_chat_template(tokenizer, prompt: str, thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


# -----------------------------
# Data loading
# -----------------------------
def _build_livecodebench_test_code(public_test_cases: str) -> str:
    """Convert LiveCodeBench public_test_cases JSON to executable test code.

    LiveCodeBench problems use stdin/stdout. The generated test code
    re-runs the solution for each test case by mocking stdin and capturing stdout.
    """
    cases = json.loads(public_test_cases) if isinstance(public_test_cases, str) else public_test_cases
    if not cases:
        return ""

    lines = [
        "import sys",
        "from io import StringIO",
        "",
    ]
    for idx, tc in enumerate(cases):
        inp = tc.get("input", "")
        out = tc.get("output", "")
        lines.append(f"# --- Test case {idx} ---")
        lines.append(f"_orig_stdin, _orig_stdout = sys.stdin, sys.stdout")
        lines.append(f"sys.stdin = StringIO({repr(inp)})")
        lines.append(f"sys.stdout = _captured = StringIO()")
        lines.append(f"try:")
        lines.append(f"    exec(open(__file__).read().split('# === TESTS ===')[0])")
        lines.append(f"finally:")
        lines.append(f"    sys.stdin, sys.stdout = _orig_stdin, _orig_stdout")
        lines.append(f"_actual = _captured.getvalue().strip()")
        lines.append(f"_expected = {repr(out.strip())}")
        lines.append(f"assert _actual == _expected, "
                      f"f'Test {idx}: expected {{repr(_expected)}}, got {{repr(_actual)}}'")
        lines.append("")
    return "\n".join(lines)


def load_code_dataset(path: str) -> List[Dict[str, Any]]:
    """Load code dataset saved as prompt/test_code (e.g., humaneval.jsonl, mbpp.jsonl).

    Also handles LiveCodeBench format (question_content + public_test_cases).
    """
    data: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict) and "data" in obj:
                data = obj["data"]
            else:
                raise ValueError("JSON format not recognized.")
    else:
        raise ValueError("dataset must be .json or .jsonl")

    norm = []
    for i, ex in enumerate(data):
        # Detect LiveCodeBench format
        if "question_content" in ex and "public_test_cases" in ex:
            task_id = ex.get("question_id") or ex.get("question_title") or f"lcb_{i}"
            prompt = ex["question_content"].strip()
            if ex.get("starter_code"):
                prompt += f"\n\n{ex['starter_code']}"
            test_code = _build_livecodebench_test_code(ex["public_test_cases"])
            if prompt and test_code:
                norm.append({
                    "task_id": str(task_id),
                    "prompt": prompt,
                    "test_code": test_code,
                    "test_type": "stdin",
                })
            continue

        # Standard format (HumanEval / MBPP / BigCodeBench)
        prompt = ex.get("prompt") or ""
        test_code = ex.get("test_code") or ""

        # 兼容旧字段
        if not test_code:
            test_code = ex.get("test") or ex.get("tests") or ex.get("test_case") or ""
        if not test_code and isinstance(ex.get("test_list"), list):
            test_code = "\n".join(ex["test_list"])

        task_id = ex.get("task_id") or ex.get("id") or f"task_{i}"

        if not prompt.strip() or not test_code.strip():
            continue

        norm.append(
            {
                "task_id": str(task_id),
                "prompt": prompt.strip(),
                "test_code": test_code,
            }
        )
    return norm


# -----------------------------
# Code extraction
# -----------------------------
def extract_code(text: str) -> str:
    """Extract python code from model output."""
    # Priority 1: Extract code between [BEGIN] and [DONE] markers (for MBPP format)
    begin_match = re.search(r"\[BEGIN\]\s*(.*?)\[DONE\]", text, flags=re.DOTALL)
    if begin_match:
        code = begin_match.group(1).strip()
        # Remove any test assertions that might have been included
        lines = code.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('assert ')]
        return '\n'.join(filtered_lines).strip()
    
    # Priority 2: Fenced python code blocks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if code_blocks:
        code = "\n\n".join(code_blocks).strip()
        # Remove test assertions
        lines = code.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('assert ')]
        return '\n'.join(filtered_lines).strip()

    # Priority 3: Content inside <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if answer_match:
        code = answer_match.group(1).strip()
        # Remove test assertions
        lines = code.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('assert ')]
        return '\n'.join(filtered_lines).strip()

    # Last resort: return raw text but remove test assertions
    lines = text.strip().split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith('assert ')]
    return '\n'.join(filtered_lines).strip()


# -----------------------------
# Test runner
# -----------------------------
def run_python_test(code: str, test_code: str, timeout: float = 15.0,
                    test_type: str = "function") -> Tuple[bool, str]:
    """Run provided tests against generated code in an isolated subprocess.

    Args:
        test_type: "function" for HumanEval/MBPP/BigCodeBench (code + test in same file),
                   "stdin" for LiveCodeBench (stdin/stdout, solution run via exec per test case).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if test_type == "stdin":
            # LiveCodeBench: solution code goes first, then a separator, then test code
            # Test code uses exec() to re-run the solution part with mocked stdin
            combined_code = f"""{code}

# === TESTS ===
{test_code}
"""
        else:
            # Standard: solution + test asserts in the same namespace
            combined_code = f"""{code}

{test_code}
"""
        test_file = os.path.join(tmpdir, "test_solution.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(combined_code)

        try:
            result = subprocess.run(
                ["python3", test_file],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, "passed"
            output = (result.stdout + result.stderr)[-500:]
            return False, output if output else "No output"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)


# -----------------------------
# Evaluator
# -----------------------------
def evaluate(
    dataset_path: str,
    model_path: str,
    tp: int = 1,
    out_csv: str = "results/code_eval_detail.csv",
    out_json: str = "results/code_eval_summary.json",
    max_model_len: int = 4096,
    thinking: bool = False,
    timeout: float = 15.0,
    repeat: int = 1,
    seed: Optional[int] = 2,
):
    base_samples = load_code_dataset(dataset_path)
    print(f"📊 Loaded {len(base_samples)} tasks from {dataset_path}")

    # Expand samples for pass@k
    samples: List[Dict[str, Any]] = []
    for ex in base_samples:
        for k in range(repeat):
            ex_copy = ex.copy()
            ex_copy["task_id"] = f"{ex['task_id']}_rep{k+1}"
            ex_copy["orig_task_id"] = ex["task_id"]
            samples.append(ex_copy)
    print(f"🚀 Expanded to {len(samples)} samples (repeat={repeat})")

    engine = VLLMBackend(model_path, tensor_parallel_size=tp, max_model_len=max_model_len)
    tokenizer = engine.LLM.get_tokenizer()

    prompts = [apply_chat_template(tokenizer, s["prompt"], thinking=thinking) for s in samples]

    preds, latencies = [], []
    B = 500
    for i in range(0, len(prompts), B):
        batch = prompts[i : i + B]
        t0 = time.time()
        texts = engine.generate(batch)
        dt = time.time() - t0
        per_item = dt / len(batch)
        latencies.extend([per_item] * len(batch))
        preds.extend(texts)

    rows, passed = [], 0
    for ex, text, lat in zip(samples, preds, latencies):
        code = extract_code(text)
        ok, msg = run_python_test(code, ex["test_code"], timeout=timeout,
                                  test_type=ex.get("test_type", "function"))
        passed += int(ok)
        rows.append(
            {
                "task_id": ex["task_id"],
                "orig_task_id": ex.get("orig_task_id", ex["task_id"]),
                "prompt": ex["prompt"],
                "pred_text": text.strip(),
                "code": code,
                "passed": int(ok),
                "test_msg": msg,
                "latency_s": round(lat, 4),
            }
        )

    total = len(samples)
    acc = passed / total if total else 0.0

    # Pass@repeat on original tasks
    grouped: Dict[str, List[int]] = {}
    for r in rows:
        oid = r["orig_task_id"]
        grouped.setdefault(oid, []).append(r["passed"])
    pass_total = len(grouped)
    pass_success = sum(1 for vals in grouped.values() if any(vals))
    pass_at_k = pass_success / pass_total if pass_total else 0.0

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("task_id,passed,latency_s,code,test_msg\n")
        for r in rows:
            f.write(f"{r['task_id']},{r['passed']},{r['latency_s']},{json.dumps(r['code'])},{json.dumps(r['test_msg'])}\n")

    summary = {
        "acc": acc,
        "passed": passed,
        "total": total,
        "repeat": repeat,
        "pass_at_k": pass_at_k,
        "pass_success": pass_success,
        "orig_total": pass_total,
        "dataset_path": dataset_path,
        "model_path": model_path,
        "thinking": thinking,
        "timeout": timeout,
        "csv_file": out_csv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Acc: {acc:.4f} ({passed}/{total})")
    print(f"💾 Detail CSV: {out_csv}")
    print(f"💾 Summary JSON: {out_json}")
    return acc


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str, help="Path to HumanEval/MBPP json/jsonl")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="results/code_eval")
    p.add_argument("--repeat", type=int, default=1, help="Repeat sampling times per task for pass@k")
    args = p.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "eval_detail.csv")
    out_json = os.path.join(out_dir, "eval_summary.json")

    evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        tp=args.tp,
        out_csv=out_csv,
        out_json=out_json,
        max_model_len=args.max_model_len,
        thinking=args.thinking,
        timeout=args.timeout,
        repeat=args.repeat,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


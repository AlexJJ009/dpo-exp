#!/usr/bin/env python3
"""
Multi-layer answer verification module.

Supports three answer types:
  - MCQ: single letter A-E extraction
  - Math: LaTeX semantic/regex matching
  - Code: subprocess execution against test cases

Usage as module:
    from answer_verify import verify_answer, verify_code_answer
    result = verify_answer(response_text, ground_truth)
    result = verify_code_answer(response_text, test_case, source)
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: math_verify semantic checking
# ──────────────────────────────────────────────────────────────────────────────

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


def verify_with_math_verify(response: str, ground_truth: str) -> Optional[bool]:
    """
    Semantic LaTeX verification using math_verify.

    Returns True/False if verification succeeds, None if it cannot determine.
    """
    if not MATH_VERIFY_AVAILABLE:
        return None

    try:
        gold_str = "$" + ground_truth.strip().strip("$") + "$"
        gold_parsed = parse(gold_str, extraction_mode="first_match")

        if len(gold_parsed) == 0:
            return None

        # First try parsing the full response (looks for \boxed{} etc.)
        answer_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="all",
        )

        if len(answer_parsed) > 0:
            return bool(verify(gold_parsed, answer_parsed))

        # If no boxed content found, try extracting from <answer> tags
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_str = "$" + answer_text.strip("$") + "$"
            answer_parsed = parse(answer_str, extraction_mode="first_match")
            if len(answer_parsed) > 0:
                return bool(verify(gold_parsed, answer_parsed))

        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: Regex-based LaTeX matching
# ──────────────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    # Find all \boxed{ positions
    pattern = r"\\boxed\s*\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Use the last match
    match = matches[-1]
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3: MCQ letter-choice extraction
# ──────────────────────────────────────────────────────────────────────────────

# Patterns ordered from most specific to least specific.
# All are searched against the full response; the earliest match wins.
_MCQ_PATTERNS = [
    # "The (correct/right/best) answer is (A)"
    re.compile(
        r"(?:the\s+)?(?:correct|right|best|most\s+appropriate\s+)answer\s+is\s*[:\-]?\s*\(?([A-Ea-e])\)?",
        re.IGNORECASE,
    ),
    # "The answer is (A)" / "answer is B"
    re.compile(r"\banswer\s+is\s+\(?([A-Ea-e])\)?", re.IGNORECASE),
    # "Answer: A" / "Answer- A"
    re.compile(r"\banswer\s*[:\-]\s*\(?([A-Ea-e])\)?", re.IGNORECASE),
    # "Ans. is 'a'" / "Ans is A" / "Ans: B" / "Ans. D"
    re.compile(r"\bans\.?\s*(?:is\s*)?[:\-\s]*['\"\(]?\s*([A-Ea-e])\b", re.IGNORECASE),
    # "Option A is correct/right"
    re.compile(
        r"\boption\s+\(?([A-Ea-e])\)?\s+is\s+(?:correct|right|the\s+(?:correct\s+)?answer)",
        re.IGNORECASE,
    ),
]
# Bare letter at the very start of the response: "A\n", "C. explanation"
_MCQ_BARE_START = re.compile(r"\A\s*\(?([A-Ea-e])\)?\s*[\.\)\:\n\r]")


def extract_mcq_letter(text: str) -> Optional[str]:
    """Extract a single letter answer (A-E) from MCQ-style responses.

    Searches the full text with all patterns and returns the earliest match.
    Falls back to a bare letter at position 0.
    """
    best_letter = None
    best_pos = len(text) + 1

    for pattern in _MCQ_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < best_pos:
            best_letter = match.group(1).upper()
            best_pos = match.start()

    if best_letter is not None:
        return best_letter

    # Fallback: bare letter at position 0
    match = _MCQ_BARE_START.match(text)
    if match:
        return match.group(1).upper()

    return None


def verify_mcq(response: str, ground_truth: str) -> Optional[dict]:
    """Verify MCQ letter-choice answers.

    Only activates when ground_truth is a single letter A-E.
    Returns None if ground_truth is not a single letter (falls through to math layers).
    """
    gt = ground_truth.strip().upper()
    if not re.match(r"^[A-E]$", gt):
        return None

    extracted = extract_mcq_letter(response)
    if extracted is None:
        return {"correct": False, "method": "mcq", "extracted_answer": None}

    return {
        "correct": extracted == gt,
        "method": "mcq",
        "extracted_answer": extracted,
    }


def normalize_latex(s: str) -> str:
    """Normalize a LaTeX string for comparison."""
    s = s.strip()
    # Remove surrounding $
    s = s.strip("$")
    # Remove \text{}, \mathrm{}, \textbf{} wrappers
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\s*\{([^}]*)\}", r"\1", s)
    # Remove \left and \right
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Remove display math delimiters
    s = s.replace("\\[", "").replace("\\]", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing period or comma
    s = s.rstrip(".,;")
    # Normalize common LaTeX
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("{,}", ",")
    # Remove ° and \circ (for degree answers)
    s = s.replace("°", "").replace("\\circ", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    # Remove \displaystyle
    s = s.replace("\\displaystyle", "")
    # Normalize fractions: \frac{a}{b} → a/b for simple cases
    s = re.sub(r"\\dfrac", r"\\frac", s)
    return s.strip()


def verify_with_regex(response: str, ground_truth: str) -> tuple[bool, Optional[str]]:
    """
    Regex-based answer verification.

    Extracts the answer from \\boxed{} or <answer> tags, normalizes, and compares.

    Returns (is_correct, extracted_answer).
    """
    # Try to extract answer from response
    extracted = extract_boxed(response)
    if extracted is None:
        extracted = extract_answer_tag(response)
    if extracted is None:
        return False, None

    # Normalize both
    norm_pred = normalize_latex(extracted)
    norm_gold = normalize_latex(ground_truth)

    if not norm_pred or not norm_gold:
        return False, extracted

    # Direct comparison after normalization
    if norm_pred == norm_gold:
        return True, extracted

    # Try numeric comparison
    try:
        val_pred = float(norm_pred.replace(",", ""))
        val_gold = float(norm_gold.replace(",", ""))
        if abs(val_pred - val_gold) < 1e-6:
            return True, extracted
    except (ValueError, OverflowError):
        pass

    return False, extracted


# ──────────────────────────────────────────────────────────────────────────────
# Layer 4: Code execution verification
# ──────────────────────────────────────────────────────────────────────────────

def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from a model response.

    Priority: ```python blocks > ``` blocks > <answer> tag > raw text with def/class.
    """
    # Remove <think> content
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Try ```python ... ``` blocks
    blocks = re.findall(r"```python\s*(.*?)```", cleaned, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())

    # Try generic ``` blocks
    blocks = re.findall(r"```\s*(.*?)```", cleaned, re.DOTALL)
    if blocks:
        code_blocks = [b.strip() for b in blocks if b.strip() and ("def " in b or "import " in b or "class " in b)]
        if code_blocks:
            return "\n\n".join(code_blocks)

    # Try <answer> tag
    match = re.search(r"<answer>(.*?)</answer>", cleaned, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        inner_blocks = re.findall(r"```(?:python)?\s*(.*?)```", inner, re.DOTALL)
        if inner_blocks:
            return "\n\n".join(b.strip() for b in inner_blocks if b.strip())
        if "def " in inner or "import " in inner:
            return inner

    # Last resort: look for function definitions in the cleaned text
    lines = cleaned.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("import "):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)
    return None


def _extract_function_name(code: str) -> Optional[str]:
    """Extract the first function name from code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None


def _build_test_code(code: str, test_case: str, source: str) -> Optional[str]:
    """Convert test_case + source into executable test code.

    Adapted from am_deepseek_r1_code_verified.py.
    """
    try:
        tc = json.loads(test_case) if isinstance(test_case, str) else test_case
    except (json.JSONDecodeError, TypeError):
        return None

    if source == "codeio":
        if not isinstance(tc, dict) or "inputs" not in tc or "outputs" not in tc:
            return None
        func_name = _extract_function_name(code)
        if not func_name:
            return None

        lines = ["from solution import *", ""]
        inputs, outputs = tc["inputs"], tc["outputs"]
        if len(inputs) != len(outputs):
            return None

        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            if not isinstance(inp, dict) or not isinstance(out, dict):
                return None
            output_value = out.get("output", list(out.values())[0] if out else None)
            if len(inp) == 1:
                param_value = list(inp.values())[0]
                lines.append(f"result = {func_name}({repr(param_value)})")
            else:
                kwargs = ", ".join(f"{k}={repr(v)}" for k, v in inp.items())
                lines.append(f"result = {func_name}({kwargs})")

            if isinstance(output_value, float):
                lines.append(f"if abs({repr(output_value)}) > 1e-10:")
                lines.append(f"    assert abs((result - {repr(output_value)}) / {repr(output_value)}) < 1e-5, "
                             f"f'Expected {repr(output_value)}, got {{result}}'")
                lines.append(f"else:")
                lines.append(f"    assert abs(result - {repr(output_value)}) < 1e-6, "
                             f"f'Expected {repr(output_value)}, got {{result}}'")
            else:
                lines.append(f"assert result == {repr(output_value)}, "
                             f"f'Expected {repr(output_value)}, got {{result}}'")
        return "\n".join(lines)

    elif source in ("OpenCoder", "OpenCoderStage2"):
        if isinstance(tc, dict) and "testcase" in tc:
            tc = tc["testcase"]
        if not isinstance(tc, list):
            return None
        lines = ["from solution import *", ""]
        for assertion in tc:
            if isinstance(assertion, str):
                assertion = assertion.split("#")[0].strip()
                if assertion:
                    lines.append(assertion)
        return "\n".join(lines)

    elif source == "prime":
        if not isinstance(tc, dict) or "ground_truth" not in tc:
            return None
        gt = tc["ground_truth"]
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(gt, dict) or "inputs" not in gt or "outputs" not in gt:
            return None
        inputs, outputs = gt["inputs"], gt["outputs"]
        lines = ["import sys", "from io import StringIO", ""]
        lines.append(f"_solution_code = open('solution.py').read()")
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            if not isinstance(inp, str) or not isinstance(out, str):
                continue
            lines.append(f"sys.stdin = StringIO({repr(inp)})")
            lines.append(f"sys.stdout = _cap = StringIO()")
            lines.append(f"try:")
            lines.append(f"    exec(_solution_code)")
            lines.append(f"finally:")
            lines.append(f"    sys.stdin, sys.stdout = sys.__stdin__, sys.__stdout__")
            lines.append(f"assert _cap.getvalue().rstrip() == {repr(out.rstrip())}, "
                         f"f'Test {i} failed'")
        return "\n".join(lines)

    return None


def verify_code_answer(response: str, test_case: str, source: str,
                       timeout: int = 10) -> dict:
    """Verify code answer by extracting code and running against test cases.

    Args:
        response: Full model response text
        test_case: JSON string of test cases
        source: Data source type (codeio, OpenCoder, etc.)
        timeout: Execution timeout in seconds

    Returns:
        dict with keys: correct, method, extracted_answer
    """
    code = extract_python_code(response)
    if code is None:
        return {"correct": False, "method": "code_exec", "extracted_answer": None}

    test_code = _build_test_code(code, test_case, source)
    if test_code is None:
        return {"correct": False, "method": "code_exec", "extracted_answer": code[:200]}

    with tempfile.TemporaryDirectory() as tmpdir:
        solution_file = os.path.join(tmpdir, "solution.py")
        test_file = os.path.join(tmpdir, "test_solution.py")

        with open(solution_file, "w") as f:
            f.write(code)
        with open(test_file, "w") as f:
            f.write(test_code)

        try:
            result = subprocess.run(
                ["python", test_file],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = result.returncode == 0
            return {
                "correct": passed,
                "method": "code_exec",
                "extracted_answer": code[:200],
            }
        except subprocess.TimeoutExpired:
            return {"correct": False, "method": "code_exec_timeout", "extracted_answer": code[:200]}
        except Exception as e:
            return {"correct": False, "method": "code_exec_error", "extracted_answer": str(e)[:200]}


# ──────────────────────────────────────────────────────────────────────────────
# Combined verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_answer(response: str, ground_truth: str) -> dict:
    """
    Verify whether a model response contains the correct answer.

    Uses three-layer verification:
      0. MCQ letter-choice matching (if ground_truth is a single letter A-E)
      1. math_verify semantic checking (primary, for math expressions)
      2. regex-based LaTeX matching (fallback)

    Args:
        response: Full model response text
        ground_truth: Expected answer string

    Returns:
        dict with keys:
            correct (bool): Whether the answer is correct
            method (str): Which verification method was used
            extracted_answer (str|None): The extracted answer from the response
    """
    # Layer 0: MCQ letter-choice (activates only when ground_truth is a single letter)
    mcq_result = verify_mcq(response, ground_truth)
    if mcq_result is not None:
        return mcq_result

    # Layer 1: math_verify (for math expressions)
    extracted = extract_boxed(response)
    if extracted is None:
        extracted = extract_answer_tag(response)

    math_result = verify_with_math_verify(response, ground_truth)
    if math_result is not None:
        return {
            "correct": math_result,
            "method": "math_verify",
            "extracted_answer": extracted,
        }

    # Layer 2: regex fallback
    regex_correct, regex_extracted = verify_with_regex(response, ground_truth)
    return {
        "correct": regex_correct,
        "method": "regex",
        "extracted_answer": regex_extracted if regex_extracted is not None else extracted,
    }


if __name__ == "__main__":
    # Quick self-test
    tests = [
        # Math (existing)
        ("The answer is \\boxed{42}", "42", True),
        ("The answer is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
        ("The answer is \\boxed{3}", "5", False),
        ("<think>work</think><answer>120</answer>", "120", True),
        ("The answer is \\boxed{x = 1}", "x = 1", True),
        ("No answer here", "42", False),
        # MCQ letter-choice (new)
        ("Long reasoning...\nThe answer is (A).", "A", True),
        ("Long reasoning...\nThe answer is (A).", "B", False),
        ("The answer is B.", "B", True),
        ("The answer is (C)", "C", True),
        ("analysis...\nThe answer is (d).", "D", True),
        ("No clear answer letter", "A", False),
        ("The answer is (A).", "42", False),  # MCQ should not activate for non-letter GT
    ]
    for response, gt, expected in tests:
        result = verify_answer(response, gt)
        status = "PASS" if result["correct"] == expected else "FAIL"
        print(f"[{status}] correct={result['correct']}, method={result['method']}, "
              f"extracted={result['extracted_answer']}, gt={gt!r}, expected={expected}")

    # Code verification tests
    print("\n--- Code verification tests ---")
    code_tests = [
        # Correct code
        (
            "```python\ndef add(a, b):\n    return a + b\n```",
            '{"inputs": [{"a": 1, "b": 2}], "outputs": [{"output": 3}]}',
            "codeio",
            True,
        ),
        # Wrong code
        (
            "```python\ndef add(a, b):\n    return a - b\n```",
            '{"inputs": [{"a": 1, "b": 2}], "outputs": [{"output": 3}]}',
            "codeio",
            False,
        ),
        # OpenCoder assert-based
        (
            "```python\ndef double(x):\n    return x * 2\n```",
            '["assert double(3) == 6", "assert double(0) == 0"]',
            "OpenCoder",
            True,
        ),
    ]
    for response, tc, source, expected in code_tests:
        result = verify_code_answer(response, tc, source)
        status = "PASS" if result["correct"] == expected else "FAIL"
        print(f"[{status}] correct={result['correct']}, method={result['method']}, expected={expected}")

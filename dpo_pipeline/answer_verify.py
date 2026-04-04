#!/usr/bin/env python3
"""
Dual-layer answer verification module.

Layer 1: math_verify + latex2sympy2_extended for semantic LaTeX equivalence.
Layer 2: Regex-based LaTeX string matching as fallback.

Adapted from:
  /data-1/verl07/verl/recipe/joint_training/custom_reward_function_latex_verify.py

Usage as module:
    from answer_verify import verify_answer
    result = verify_answer(response_text, ground_truth)
    # result: {"correct": bool, "method": str, "extracted_answer": str|None}
"""

import re
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
# Combined verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_answer(response: str, ground_truth: str) -> dict:
    """
    Verify whether a model response contains the correct answer.

    Uses dual-layer verification:
      1. math_verify semantic checking (primary)
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
    extracted = extract_boxed(response)
    if extracted is None:
        extracted = extract_answer_tag(response)

    # Layer 1: math_verify
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
        ("The answer is \\boxed{42}", "42", True),
        ("The answer is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
        ("The answer is \\boxed{3}", "5", False),
        ("<think>work</think><answer>120</answer>", "120", True),
        ("The answer is \\boxed{x = 1}", "x = 1", True),
        ("No answer here", "42", False),
    ]
    for response, gt, expected in tests:
        result = verify_answer(response, gt)
        status = "PASS" if result["correct"] == expected else "FAIL"
        print(f"[{status}] correct={result['correct']}, method={result['method']}, "
              f"extracted={result['extracted_answer']}, expected={expected}")

#!/usr/bin/env python3
"""
Batch rollout generation using vLLM.

Reads extracted prompts, generates multiple rollout responses per prompt
using a specified base model, and saves the outputs.

Usage:
    python batch_rollout.py \
        --input /data-1/dataset/extracted_prompts.jsonl \
        --output /data-1/dataset/rollouts.jsonl \
        --model Qwen/Qwen3-4B-Base \
        --num-rollouts 2 \
        --limit 10 \
        --max-tokens 4096
"""

import argparse
import json
import os
import sys
import time


def load_prompts(path: str, limit: int | None = None) -> list[dict]:
    """Load extracted prompts from JSONL file."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def format_prompt_for_model(prompt_messages: list[dict]) -> str:
    """
    Format chat messages into a text prompt for a base (non-chat) model.

    Since we're using base models (not instruct/chat-tuned), we format the
    messages as a simple concatenation that the model can continue from.
    """
    parts = []
    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
    parts.append("Assistant: <think>\n")
    return "".join(parts)


def format_prompt_with_chat_template(prompt_messages: list[dict], tokenizer) -> str:
    """
    Format chat messages using the model's tokenizer chat template.

    For SFT/chat-tuned models that were trained with a specific chat template
    (e.g., Qwen3's <|im_start|>/<|im_end|> format), we use the tokenizer's
    apply_chat_template to produce the correct format, then append <think>
    to seed chain-of-thought reasoning.
    """
    formatted = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Seed the thinking tag so the model starts with chain-of-thought
    formatted += "<think>\n"
    return formatted


def run_rollouts(
    prompts: list[dict],
    model_name: str,
    num_rollouts: int,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    use_chat_template: bool = False,
) -> list[dict]:
    """Generate rollouts for each prompt using vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # Choose prompt formatter
    if use_chat_template:
        tokenizer = llm.get_tokenizer()
        print("Using model's chat template for prompt formatting")
        formatter = lambda msgs: format_prompt_with_chat_template(msgs, tokenizer)
    else:
        formatter = format_prompt_for_model

    # Build text prompts
    text_prompts = []
    prompt_indices = []
    for idx, p in enumerate(prompts):
        text = formatter(p["prompt"])
        for _ in range(num_rollouts):
            text_prompts.append(text)
            prompt_indices.append(idx)

    print(f"Generating {len(text_prompts)} rollouts for {len(prompts)} prompts...")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    start = time.time()
    outputs = llm.generate(text_prompts, sampling_params)
    elapsed = time.time() - start
    print(f"Generation completed in {elapsed:.1f}s")

    # Group outputs by prompt
    results = []
    for prompt_idx, p in enumerate(prompts):
        rollout_responses = []
        for out_idx, out in enumerate(outputs):
            if prompt_indices[out_idx] == prompt_idx:
                generated_text = out.outputs[0].text
                # Prepend the <think> we added as seed
                full_response = "<think>\n" + generated_text
                rollout_responses.append(full_response)

        results.append({
            "prompt": p["prompt"],
            "reference_answer": p["reference_answer"],
            "chosen": p["chosen"],
            "rollouts": rollout_responses,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch rollout generation with vLLM")
    parser.add_argument("--input", required=True, help="Path to extracted prompts JSONL")
    parser.add_argument("--output", required=True, help="Path to output rollouts JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base", help="Model name/path")
    parser.add_argument("--num-rollouts", type=int, default=2, help="Rollouts per prompt")
    parser.add_argument("--limit", type=int, default=None, help="Max prompts to process")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per rollout")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--chat-template", action="store_true",
                        help="Use model's tokenizer chat template for prompt formatting (for SFT/chat models)")
    args = parser.parse_args()

    prompts = load_prompts(args.input, limit=args.limit)
    print(f"Loaded {len(prompts)} prompts")

    results = run_rollouts(
        prompts=prompts,
        model_name=args.model,
        num_rollouts=args.num_rollouts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_chat_template=args.chat_template,
    )

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} rollout results to {args.output}")


if __name__ == "__main__":
    main()

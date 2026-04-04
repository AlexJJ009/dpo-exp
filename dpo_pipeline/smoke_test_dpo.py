"""
Smoke test: minimal DPO training loop using TRL's DPOTrainer.

Creates a tiny synthetic preference dataset (~10 pairs) and runs
a short DPO training (5 steps) with a small model to verify the
environment is working correctly.
"""

import json
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def create_synthetic_preference_data(num_pairs: int = 10) -> Dataset:
    """Create a minimal synthetic preference dataset for DPO smoke testing."""
    prompts = []
    chosen = []
    rejected = []

    math_problems = [
        ("What is 2 + 3?", "The answer is 5.", "The answer is 7."),
        ("What is 10 - 4?", "The answer is 6.", "The answer is 8."),
        ("What is 3 * 7?", "The answer is 21.", "The answer is 24."),
        ("What is 15 / 3?", "The answer is 5.", "The answer is 4."),
        ("What is 8 + 9?", "The answer is 17.", "The answer is 16."),
        ("What is 100 - 37?", "The answer is 63.", "The answer is 67."),
        ("What is 6 * 8?", "The answer is 48.", "The answer is 46."),
        ("What is 20 / 5?", "The answer is 4.", "The answer is 5."),
        ("What is 11 + 12?", "The answer is 23.", "The answer is 22."),
        ("What is 50 - 18?", "The answer is 32.", "The answer is 38."),
        ("What is 9 * 9?", "The answer is 81.", "The answer is 72."),
        ("What is 36 / 6?", "The answer is 6.", "The answer is 7."),
    ]

    for i in range(num_pairs):
        problem = math_problems[i % len(math_problems)]
        prompts.append(problem[0])
        chosen.append(problem[1])
        rejected.append(problem[2])

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    })


def main():
    print("=" * 60)
    print("DPO Smoke Test")
    print("=" * 60)

    # 1. Check environment
    print(f"\nPython version: {sys.version}")

    import trl
    print(f"TRL version: {trl.__version__}")
    assert trl.__version__ == "0.29.0", f"Expected TRL 0.29.0, got {trl.__version__}"

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU (smoke test only)")

    # 2. Create synthetic dataset
    print("\nCreating synthetic preference dataset...")
    dataset = create_synthetic_preference_data(num_pairs=10)
    print(f"Dataset size: {len(dataset)} pairs")
    print(f"Sample: prompt='{dataset[0]['prompt']}', chosen='{dataset[0]['chosen']}', rejected='{dataset[0]['rejected']}'")

    # 3. Load a small model for smoke testing
    # Use Qwen3-4B-Base from local cache
    model_name = "Qwen/Qwen3-4B-Base"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 4. Configure DPO training
    print("\nConfiguring DPO training...")
    training_args = DPOConfig(
        output_dir="/tmp/dpo_smoke_test",
        num_train_epochs=1,
        max_steps=5,
        per_device_train_batch_size=2,
        learning_rate=5e-7,
        beta=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        max_length=128,
        gradient_checkpointing=True,
    )

    # 5. Initialize trainer
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. Run training
    print("\nStarting DPO training (5 steps)...")
    train_result = trainer.train()

    # 7. Verify results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)

    metrics = train_result.metrics
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.2f}s")

    # Check training logs for per-step loss
    log_history = trainer.state.log_history
    print(f"\nPer-step logs ({len(log_history)} entries):")

    losses = []
    for entry in log_history:
        if "loss" in entry:
            loss_val = entry["loss"]
            step = entry.get("step", "?")
            print(f"  Step {step}: loss={loss_val:.4f}")
            losses.append(loss_val)

    # Validate
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    all_ok = True

    # Check loss is finite
    if losses:
        for i, loss in enumerate(losses):
            if not (isinstance(loss, (int, float)) and abs(loss) < float("inf") and loss == loss):
                print(f"FAIL: Step {i} loss is not finite: {loss}")
                all_ok = False
        if all(isinstance(l, (int, float)) and abs(l) < float("inf") and l == l for l in losses):
            print(f"PASS: All {len(losses)} loss values are finite numbers")
    else:
        print("FAIL: No loss values recorded")
        all_ok = False

    # Check final training loss
    final_loss = metrics.get("train_loss")
    if final_loss is not None and isinstance(final_loss, (int, float)) and abs(final_loss) < float("inf") and final_loss == final_loss:
        print(f"PASS: Final training loss is finite: {final_loss:.4f}")
    else:
        print(f"FAIL: Final training loss is not finite: {final_loss}")
        all_ok = False

    if all_ok:
        print("\n*** SMOKE TEST PASSED ***")
    else:
        print("\n*** SMOKE TEST FAILED ***")
        sys.exit(1)

    # Save summary
    summary = {
        "status": "PASSED" if all_ok else "FAILED",
        "model": model_name,
        "trl_version": trl.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_pairs": len(dataset),
        "max_steps": 5,
        "final_loss": final_loss,
        "per_step_losses": losses,
    }
    summary_path = "/data-1/dpo-experiment/dpo_pipeline/smoke_test_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()

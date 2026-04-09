"""
DPO Training script for Gemma3-4B-SFT (SFT-finetuned model) with DeepSpeed ZeRO 2.

Trains on strict-filtered preference pairs and saves checkpoint.

Usage (inside dpo-harness Docker container):
    accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
        dpo_pipeline/train_dpo_gemma3_4b_sft.py
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


# ======================== Configuration ========================
MODEL_NAME = "/data-1/.cache/gemma3-4b-base-sft-stage-1"
DATASET_PATH = "/data-1/dataset/dpo/dpo-gemma3-4b-sft/dpo-gemma3-4b-sft-pairs.jsonl"
OUTPUT_DIR = "/data-1/checkpoints/gemma3-4b-sft-dpo"
LOG_DIR = "/data-1/checkpoints/gemma3-4b-sft-dpo/training_logs"

# Training hyperparameters
BETA = 0.1
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2  # 8 GPUs x 1 x 2 = effective batch 16
MAX_LENGTH = 2048
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 5
SAVE_STEPS = 200


class MetricLoggerCallback(TrainerCallback):
    """Callback to track all training metrics and check for NaN/Inf."""

    def __init__(self):
        self.step_logs = []
        self.has_nan_inf = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        self.step_logs.append(entry)

        for key in ["loss", "rewards/chosen", "rewards/rejected", "rewards/margins"]:
            if key in logs:
                val = logs[key]
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    print(f"WARNING: {key} = {val} at step {state.global_step}")
                    self.has_nan_inf = True


def _copy_preprocessor_config(model_name: str, output_dir: str):
    """Copy preprocessor_config.json from source model to checkpoint if missing.

    Gemma3 is architecturally multimodal (Gemma3ForConditionalGeneration).
    vLLM tries to init multimodal processing when loading checkpoints and
    fails with OSError if preprocessor_config.json is absent.
    """
    import shutil
    dst = Path(output_dir) / "preprocessor_config.json"
    if dst.exists():
        return

    # Try resolving the HF cache snapshot path
    from huggingface_hub import try_to_load_from_cache
    try:
        cached = try_to_load_from_cache(model_name, "preprocessor_config.json")
        if cached and isinstance(cached, str):
            shutil.copy2(cached, dst)
            print(f"Copied preprocessor_config.json from HF cache to {dst}")
            return
    except Exception:
        pass

    # Fallback: check if model_name is a local directory
    src = Path(model_name) / "preprocessor_config.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied preprocessor_config.json from {src} to {dst}")
    else:
        print(f"WARNING: preprocessor_config.json not found for {model_name}. "
              "vLLM evaluation may fail for multimodal architectures.")


def load_preference_dataset(path: str) -> Dataset:
    """Load preference pairs from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} preference pairs from {path}")
    print(f"Columns: {dataset.column_names}")
    return dataset


def main():
    print("=" * 70)
    print("Gemma3-4B-SFT DPO Training (DeepSpeed ZeRO 2)")
    print("=" * 70)

    # ======================== Environment Check ========================
    print(f"\nPython: {sys.version}")
    import trl
    print(f"TRL: {trl.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ======================== Load Dataset ========================
    print("\nLoading dataset...")
    dataset = load_preference_dataset(DATASET_PATH)

    # ======================== Load Model & Tokenizer ========================
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Gemma3 sliding window — use eager for compatibility
        local_files_only=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e9:.2f}B parameters")

    # ======================== Training Config ========================
    print("\nConfiguring DPO training...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        dataloader_num_workers=4,
        seed=42,
        ddp_find_unused_parameters=False,
    )

    # ======================== Initialize Trainer ========================
    print("Initializing DPOTrainer...")
    metric_logger = MetricLoggerCallback()

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[metric_logger],
    )

    total_steps = len(trainer.get_train_dataloader()) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    print(f"\nTraining plan:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size (per device): {PER_DEVICE_BATCH_SIZE}")
    print(f"  Num devices: {training_args.world_size}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * training_args.world_size}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Estimated total steps: {total_steps}")
    print(f"  Beta: {BETA}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")

    # ======================== Train ========================
    print("\n" + "=" * 70)
    print("Starting DPO training...")
    print("=" * 70)
    train_result = trainer.train()

    # ======================== Save Final Checkpoint ========================
    print("\nSaving final model checkpoint...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Gemma3 is architecturally multimodal — vLLM requires preprocessor_config.json
    _copy_preprocessor_config(MODEL_NAME, OUTPUT_DIR)

    print(f"Checkpoint saved to {OUTPUT_DIR}")

    # ======================== Extract & Save Metrics ========================
    if not training_args.local_rank or training_args.local_rank == 0:
        print("\n" + "=" * 70)
        print("Training Results")
        print("=" * 70)

        metrics = train_result.metrics
        print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
        print(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.1f}s")
        print(f"Samples/second: {metrics.get('train_samples_per_second', 'N/A')}")

        log_history = trainer.state.log_history
        step_metrics = []
        for entry in log_history:
            if "loss" in entry:
                step_metrics.append({
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "rewards/chosen": entry.get("rewards/chosen"),
                    "rewards/rejected": entry.get("rewards/rejected"),
                    "rewards/margins": entry.get("rewards/margins"),
                    "learning_rate": entry.get("learning_rate"),
                })

        if step_metrics:
            first = step_metrics[0]
            last = step_metrics[-1]
            print(f"\nStep {first['step']}:")
            print(f"  loss={first['loss']}")
            print(f"  rewards/chosen={first.get('rewards/chosen')}")
            print(f"  rewards/rejected={first.get('rewards/rejected')}")
            print(f"  rewards/margins={first.get('rewards/margins')}")
            print(f"\nStep {last['step']}:")
            print(f"  loss={last['loss']}")
            print(f"  rewards/chosen={last.get('rewards/chosen')}")
            print(f"  rewards/rejected={last.get('rewards/rejected')}")
            print(f"  rewards/margins={last.get('rewards/margins')}")

        # ======================== Validation ========================
        print("\n" + "=" * 70)
        print("Validation Checks")
        print("=" * 70)

        all_ok = True

        if len(step_metrics) >= 2:
            first_loss = step_metrics[0]["loss"]
            final_loss = step_metrics[-1]["loss"]
            if final_loss < first_loss:
                print(f"PASS: Loss decreased: {first_loss:.4f} -> {final_loss:.4f}")
            else:
                print(f"FAIL: Loss did not decrease: {first_loss:.4f} -> {final_loss:.4f}")
                all_ok = False
        else:
            print("FAIL: Not enough steps to compare loss")
            all_ok = False

        margins = [(m["step"], m["rewards/margins"]) for m in step_metrics if m.get("rewards/margins") is not None]
        if len(margins) >= 2:
            first_margin = margins[0][1]
            final_margin = margins[-1][1]
            if final_margin > first_margin:
                print(f"PASS: Reward margins increased: {first_margin:.4f} -> {final_margin:.4f}")
            else:
                print(f"FAIL: Reward margins did not increase: {first_margin:.4f} -> {final_margin:.4f}")
                all_ok = False

        if step_metrics and step_metrics[-1].get("rewards/chosen") is not None:
            final_chosen = step_metrics[-1]["rewards/chosen"]
            final_rejected = step_metrics[-1]["rewards/rejected"]
            if final_chosen > final_rejected:
                print(f"PASS: Final chosen reward ({final_chosen:.4f}) > rejected ({final_rejected:.4f})")
            else:
                print(f"FAIL: Final chosen reward ({final_chosen:.4f}) <= rejected ({final_rejected:.4f})")
                all_ok = False

        if not metric_logger.has_nan_inf:
            print("PASS: No NaN or Inf values detected in training metrics")
        else:
            print("FAIL: NaN or Inf values detected during training")
            all_ok = False

        # ======================== Save Training Log ========================
        training_summary = {
            "model": MODEL_NAME,
            "dataset": DATASET_PATH,
            "dataset_size": len(dataset),
            "checkpoint_path": OUTPUT_DIR,
            "deepspeed": "ZeRO Stage 2",
            "num_gpus": training_args.world_size,
            "hyperparameters": {
                "beta": BETA,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * training_args.world_size,
                "max_length": MAX_LENGTH,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "lr_scheduler": LR_SCHEDULER,
            },
            "results": {
                "final_train_loss": metrics.get("train_loss"),
                "training_runtime_seconds": metrics.get("train_runtime"),
                "total_steps": len(step_metrics),
                "first_step_loss": step_metrics[0]["loss"] if step_metrics else None,
                "final_step_loss": step_metrics[-1]["loss"] if step_metrics else None,
                "first_step_margins": step_metrics[0].get("rewards/margins") if step_metrics else None,
                "final_step_margins": step_metrics[-1].get("rewards/margins") if step_metrics else None,
                "final_rewards_chosen": step_metrics[-1].get("rewards/chosen") if step_metrics else None,
                "final_rewards_rejected": step_metrics[-1].get("rewards/rejected") if step_metrics else None,
            },
            "validation": {
                "loss_decreased": all_ok,
                "no_nan_inf": not metric_logger.has_nan_inf,
            },
            "step_metrics": step_metrics,
        }

        log_path = Path(LOG_DIR) / "training_summary.json"
        with open(log_path, "w") as f:
            json.dump(training_summary, f, indent=2)
        print(f"\nTraining summary saved to {log_path}")

        if all_ok:
            print("\n*** ALL VALIDATION CHECKS PASSED ***")
        else:
            print("\n*** SOME VALIDATION CHECKS FAILED ***")
            sys.exit(1)


if __name__ == "__main__":
    main()

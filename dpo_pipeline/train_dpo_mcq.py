"""
DPO Training script for MCQ cross-domain experiments with DeepSpeed ZeRO 2.

Configurable via environment variables:
    DPO_MODEL_NAME      - Path to SFT model checkpoint
    DPO_DATASET_PATH    - Path to preference pairs JSONL
    DPO_OUTPUT_DIR      - Checkpoint output directory
    DPO_LOG_DIR         - Training log directory (default: OUTPUT_DIR/training_logs)

Usage (inside dpo-harness Docker container):
    DPO_MODEL_NAME=/data-1/.cache/Qwen3-4B-Base-Med-SFT/checkpoint-134 \
    DPO_DATASET_PATH=/data-1/dataset/dpo/dpo-med-sft-sci/dpo-med-sft-sci-pairs.jsonl \
    DPO_OUTPUT_DIR=/data-1/checkpoints/qwen3-4b-med-sft-dpo-sci \
    accelerate launch --config_file trl/accelerate_configs/zero2.yaml \
        dpo_pipeline/train_dpo_mcq.py
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


# ======================== Configuration (from env, resolved at runtime) ========================

# Training hyperparameters (overridable via env)
BETA = 0.1
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = int(os.environ.get("DPO_PER_DEVICE_BATCH", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("DPO_GRAD_ACCUM", "2"))
MAX_LENGTH = 2048
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 5
SAVE_STEPS = 50
# 8B memory optimizations
PRECOMPUTE_REF = os.environ.get("DPO_PRECOMPUTE_REF", "").lower() in ("1", "true", "yes")
OPTIMIZER = os.environ.get("DPO_OPTIM", "adamw_torch")
# Early stopping: stop when loss <= threshold for `patience` consecutive log checks
EARLY_STOP_THRESHOLD = float(os.environ.get("DPO_EARLY_STOP_LOSS", "0.01"))
EARLY_STOP_PATIENCE = int(os.environ.get("DPO_EARLY_STOP_PATIENCE", "3"))


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


class LossEarlyStoppingCallback(TrainerCallback):
    """Stop training when loss drops below a threshold for N consecutive checks.

    Saves checkpoint before stopping to preserve the best model state.
    """

    def __init__(self, threshold: float = 0.01, patience: int = 3, min_steps: int = 20):
        self.threshold = threshold
        self.patience = patience
        self.min_steps = min_steps
        self._below_count = 0
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        if state.global_step < self.min_steps:
            return

        loss = logs["loss"]
        if loss <= self.threshold:
            self._below_count += 1
            if self._below_count >= self.patience and not self.triggered:
                self.triggered = True
                print(f"\n>>> EARLY STOP: loss={loss:.4f} <= {self.threshold} "
                      f"for {self.patience} consecutive checks at step {state.global_step}. "
                      f"Saving checkpoint and stopping.")
                control.should_save = True
                control.should_training_stop = True
        else:
            self._below_count = 0


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
    MODEL_NAME = os.environ.get("DPO_MODEL_NAME")
    DATASET_PATH = os.environ.get("DPO_DATASET_PATH")
    OUTPUT_DIR = os.environ.get("DPO_OUTPUT_DIR")

    if not all([MODEL_NAME, DATASET_PATH, OUTPUT_DIR]):
        print("ERROR: Must set DPO_MODEL_NAME, DPO_DATASET_PATH, DPO_OUTPUT_DIR")
        sys.exit(1)

    LOG_DIR = os.environ.get("DPO_LOG_DIR", os.path.join(OUTPUT_DIR, "training_logs"))

    print("=" * 70)
    print("MCQ Cross-Domain DPO Training (DeepSpeed ZeRO 2)")
    print("=" * 70)
    print(f"\n  Model:   {MODEL_NAME}")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Output:  {OUTPUT_DIR}")

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
        local_files_only=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e9:.2f}B parameters")

    # ======================== Training Config ========================
    print("\nConfiguring DPO training...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    dpo_kwargs = {}
    if PRECOMPUTE_REF:
        dpo_kwargs["precompute_ref_log_probs"] = True
        print("  precompute_ref_log_probs=True (8B memory optimization)")
    if OPTIMIZER != "adamw_torch":
        dpo_kwargs["optim"] = OPTIMIZER
        print(f"  optimizer={OPTIMIZER}")

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
        save_total_limit=10,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        **dpo_kwargs,
        lr_scheduler_type=LR_SCHEDULER,
        dataloader_num_workers=4,
        seed=42,
        ddp_find_unused_parameters=False,
    )

    # ======================== Initialize Trainer ========================
    print("Initializing DPOTrainer...")
    metric_logger = MetricLoggerCallback()
    early_stopper = LossEarlyStoppingCallback(
        threshold=EARLY_STOP_THRESHOLD,
        patience=EARLY_STOP_PATIENCE,
        min_steps=20,
    )
    print(f"  Early stopping: loss <= {EARLY_STOP_THRESHOLD} for {EARLY_STOP_PATIENCE} consecutive checks")

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[metric_logger, early_stopper],
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
            print(f"  loss={first['loss']}, margins={first.get('rewards/margins')}")
            print(f"\nStep {last['step']}:")
            print(f"  loss={last['loss']}, margins={last.get('rewards/margins')}")

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
                "early_stopped": early_stopper.triggered,
                "early_stop_threshold": EARLY_STOP_THRESHOLD,
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

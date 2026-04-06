# DPO Pipeline Guide

A comprehensive guide for running the full DPO (Direct Preference Optimization) workflow — preference data generation, DPO training, and evaluation — for any model and any dataset, without needing the experiment harness or an agent.

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Preference Data Generation](#2-preference-data-generation)
3. [DPO Training](#3-dpo-training)
4. [Evaluation](#4-evaluation)
5. [Quick-Start Recipes](#5-quick-start-recipes)
6. [File Layout Reference](#6-file-layout-reference)

---

## 1. Overview & Architecture

### Pipeline Flow

```
Source Dataset
      │
      ▼
┌─────────────────┐
│ Extract Prompts  │  extract_prompts.py / extract_prompts_offset.py
│ (parse JSONL)    │
└────────┬────────┘
         │  {output-prefix}-extracted.jsonl
         ▼
┌─────────────────┐
│ Batch Rollout    │  batch_rollout.py (vLLM inference)
│ (N responses/    │
│  prompt)         │
└────────┬────────┘
         │  {output-prefix}-rollouts.jsonl
         ▼
┌─────────────────┐
│ Build Pairs      │  build_pairs.py (verify answers, pair correct vs incorrect)
└────────┬────────┘
         │  {output-prefix}-pairs.jsonl
         ▼
┌─────────────────┐
│ Clean Pairs      │  clean_pairs.py (re-verify rejected responses)
└────────┬────────┘
         │  {output-prefix}-pairs.jsonl (cleaned, in-place)
         ▼
┌─────────────────┐
│ DPO Train        │  train_dpo_{size}.py (TRL DPOTrainer)
└────────┬────────┘
         │  /data-1/checkpoints/{model-short-name}-dpo/
         ▼
┌─────────────────┐
│ Evaluate         │  offline_eval.py (vLLM + math verification)
└─────────────────┘
         │  eval_metrics.json + eval_details.parquet
```

### Docker Containers

| Container | Purpose | Key Software |
|---|---|---|
| **`dpo-harness`** | Preference data generation + DPO training | TRL v0.29.0, vLLM 0.12.0, PyTorch 2.9.1+cu126, CUDA 12.6 |
| **`verl-harness`** | Evaluation (offline inference + scoring) | vLLM 0.12.0, PyTorch 2.9.1+cu126, CUDA 12.6 |

The `dpo-harness` image is built on top of `verl-harness`, upgrading TRL from 0.9.6 to 0.29.0. Use `dpo-harness` for data generation and training. Use `verl-harness` for evaluation with `offline_eval.py`.

### Building the Docker Image

```bash
cd /data-1/dpo-experiment
docker build --network=host -f dpo_pipeline/Dockerfile -t dpo-harness .
```

### Running a Container

**Data generation / training (`dpo-harness`):**

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  dpo-harness bash
```

**Evaluation (`verl-harness`):**

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  verl-harness:latest bash
```

Both containers mount `/data-1` so all datasets, checkpoints, and outputs are accessible at the same paths inside and outside the container.

---

## 2. Preference Data Generation

### End-to-End Pipeline Command

The orchestrator script `run_pipeline.py` chains three steps: prompt extraction, batch rollout, and preference pair building.

```bash
python dpo_pipeline/run_pipeline.py \
  --model Qwen/Qwen3-4B-Base \
  --num-rollouts 16 \
  --limit 1200 \
  --output-prefix dpo-4b \
  --max-tokens 4096 \
  --temperature 0.7 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --strict
```

### Parameter Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `Qwen/Qwen3-4B-Base` | Model for rollout generation. HuggingFace ID or local path (e.g., `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1`) |
| `--num-rollouts` | int | 16 | Number of rollout responses generated per prompt |
| `--limit` | int | None (all) | Maximum number of prompts to process |
| `--output-prefix` | str | `dpo-4b` | Prefix for all output filenames |
| `--max-tokens` | int | 4096 | Maximum tokens per rollout response |
| `--temperature` | float | 0.7 | Sampling temperature for generation |
| `--tensor-parallel-size` | int | 1 | Tensor parallelism size for vLLM (set to number of GPUs, e.g., 8) |
| `--gpu-memory-utilization` | float | 0.9 | Fraction of GPU memory vLLM is allowed to use |
| `--strict` | flag | off | Enable strict three-layer filtering for rejected responses |
| `--offset` | int | None | Skip N valid prompts before extraction (see below) |
| `--chat-template` | flag | off | Use model's tokenizer chat template (required for SFT/chat models) |

### Output Files

All output files are written to `/data-1/dataset/` with the naming convention:

| File | Contents |
|---|---|
| `{output-prefix}-extracted.jsonl` | Extracted prompts with reference answers |
| `{output-prefix}-rollouts.jsonl` | Multiple model responses per prompt |
| `{output-prefix}-pairs.jsonl` | DPO preference pairs (chosen vs rejected) |

For example, `--output-prefix dpo-4b` produces:
- `/data-1/dataset/dpo-4b-extracted.jsonl`
- `/data-1/dataset/dpo-4b-rollouts.jsonl`
- `/data-1/dataset/dpo-4b-pairs.jsonl`

### Using `--offset` to Avoid Prompt Overlap

When generating preference data for multiple models from the same source dataset, use `--offset` so each model gets a different set of prompts:

```bash
# Model A: first 1200 prompts
python dpo_pipeline/run_pipeline.py \
  --model Qwen/Qwen3-4B-Base \
  --limit 1200 --output-prefix dpo-4b \
  --tensor-parallel-size 8 --strict

# Model B: next 2000 prompts (skip first 1200)
python dpo_pipeline/run_pipeline.py \
  --model Qwen/Qwen3-8B-Base \
  --limit 2000 --offset 1200 --output-prefix dpo-8b \
  --tensor-parallel-size 8 --strict
```

When `--offset` is specified, the pipeline uses `extract_prompts_offset.py` instead of `extract_prompts.py`. Both scripts validate each sample (require 2 messages, user/assistant roles, non-empty content, extractable reference answer) and the offset counts only valid samples, not raw lines.

### How `--strict` Filtering Works

With `--strict` enabled, rejected responses must pass a three-layer filter in `build_pairs.py`:

1. **Think tags**: Response must have complete `<think>...</think>` tags with non-empty reasoning content
2. **Answer completeness**: Response must contain an extractable answer in `\boxed{}` or `<answer>...</answer>` format
3. **Verified incorrect**: The extracted answer must be verified as incorrect by the dual-layer verification system (`answer_verify.py` — tries `math_verify` semantic LaTeX first, falls back to regex-based comparison)

Without `--strict`, only step 3 (answer verification) is applied.

### Post-Processing with `clean_pairs.py`

After generating pairs, run `clean_pairs.py` to catch false negatives — cases where the rejected response's answer is actually correct but was misclassified:

```bash
python dpo_pipeline/clean_pairs.py \
  --pairs /data-1/dataset/dpo-4b-pairs.jsonl \
  --extracted /data-1/dataset/dpo-4b-extracted.jsonl \
  --output /data-1/dataset/dpo-4b-pairs.jsonl
```

| Flag | Type | Description |
|---|---|---|
| `--pairs` | str (required) | Path to the pairs JSONL file to clean |
| `--extracted` | str (required) | Path to the extracted prompts JSONL (for reference answers) |
| `--output` | str (required) | Path to write cleaned pairs (can be same as `--pairs` for in-place) |

The script re-verifies every rejected response using both `math_verify` (semantic LaTeX) and regex independently. If **either** method determines the rejected response is actually correct, the pair is removed. This typically removes 3–5% of pairs.

### Source Dataset

The default source dataset is:
```
/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl
```

This path is set as `SOURCE_DATA` in `run_pipeline.py`. To use a different source, either:
- Edit `SOURCE_DATA` in `run_pipeline.py`, or
- Call the sub-scripts directly with custom `--input` paths (see `extract_prompts.py --input`)

### Example: Generating Data for a New Model

```bash
# Inside dpo-harness container
docker run --rm --gpus all --ipc=host -v /data-1:/data-1 dpo-harness bash

# Generate preference data for a custom model
python dpo_pipeline/run_pipeline.py \
  --model /data-1/.cache/my-custom-model \
  --num-rollouts 16 \
  --limit 2000 \
  --output-prefix dpo-custom \
  --max-tokens 4096 \
  --temperature 0.7 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --strict \
  --chat-template   # add this if the model is an SFT/chat model

# Clean the pairs
python dpo_pipeline/clean_pairs.py \
  --pairs /data-1/dataset/dpo-custom-pairs.jsonl \
  --extracted /data-1/dataset/dpo-custom-extracted.jsonl \
  --output /data-1/dataset/dpo-custom-pairs.jsonl
```

---

## 3. DPO Training

### Creating a Training Script

Training scripts are not parameterized via argparse — all configuration is set as Python constants at the top of the file. To train a new model, copy an existing training script and modify the constants.

**Template (based on `train_dpo_4b.py`):**

```bash
cp dpo_pipeline/train_dpo_4b.py dpo_pipeline/train_dpo_{your_model}.py
```

### Key Constants to Change

| Constant | Description | Example |
|---|---|---|
| `MODEL_NAME` | HuggingFace ID or local path to the base model | `Qwen/Qwen3-4B-Base` or `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1` |
| `DATASET_PATH` | Path to the preference pairs JSONL | `/data-1/dataset/dpo-4b-pairs.jsonl` |
| `OUTPUT_DIR` | Directory to save the trained checkpoint | `/data-1/checkpoints/qwen3-4b-dpo` |
| `LOG_DIR` | Directory for training logs and summary | `/data-1/checkpoints/qwen3-4b-dpo/training_logs` |

### Hyperparameters

The default hyperparameters work well for both 4B and 8B models. Adjust these based on model size:

| Parameter | 4B Value (`train_dpo_4b.py`) | 8B Value (`train_dpo_8b.py`) | Notes |
|---|---|---|---|
| `BETA` | 0.1 | 0.1 | DPO preference strength |
| `LEARNING_RATE` | 5e-7 | 5e-7 | Conservative LR for DPO |
| `NUM_EPOCHS` | 1 | 1 | Single epoch is standard for DPO |
| `PER_DEVICE_BATCH_SIZE` | 2 | 1 | Reduced for 8B to fit in memory |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | 16 | Adjusted so effective batch = 16 for both |
| `MAX_LENGTH` | 2048 | 2048 | Maximum sequence length |
| `WARMUP_RATIO` | 0.1 | 0.1 | Fraction of steps for LR warmup |
| `WEIGHT_DECAY` | 0.01 | 0.01 | |
| `LR_SCHEDULER` | cosine | cosine | |
| `LOGGING_STEPS` | 5 | 5 | |
| `SAVE_STEPS` | 200 | 200 | |

### Optimizer Choice

**Critical for avoiding OOM:**

- **Models ≤ 4B parameters**: Use the default Adam optimizer (no explicit `optim` setting needed in DPOConfig). Adam requires storing two full copies of model parameters as optimizer states.
- **Models ≥ 8B parameters**: Use `optim="adafactor"` — a memory-efficient optimizer that uses factored second moments instead of full per-parameter states. Also set `precompute_ref_log_probs=True` to free the reference model from GPU memory during training.

These two flags are the key differences in `train_dpo_8b.py`:

```python
training_args = DPOConfig(
    ...
    precompute_ref_log_probs=True,  # precompute ref logprobs, then free ref model
    optim="adafactor",              # memory-efficient optimizer
)
```

### Launching Training

Training must run inside the `dpo-harness` Docker container:

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  dpo-harness \
  python dpo_pipeline/train_dpo_4b.py
```

For the 8B model:

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  dpo-harness \
  python dpo_pipeline/train_dpo_8b.py
```

For an SFT-based model:

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  dpo-harness \
  python dpo_pipeline/train_dpo_4b_sft.py
```

### Monitoring Training Progress

**Live logs:**

```bash
docker logs -f <container_id>
```

The training script logs every `LOGGING_STEPS` (default 5) steps with:
- `loss` — DPO loss (should decrease)
- `rewards/chosen` — reward for chosen responses
- `rewards/rejected` — reward for rejected responses
- `rewards/margins` — chosen minus rejected (should increase)
- `learning_rate` — current LR

**After training completes**, inspect the summary:

```bash
cat /data-1/checkpoints/{model-short-name}-dpo/training_logs/training_summary.json | python -m json.tool
```

The `training_summary.json` contains:
- Model and dataset info
- All hyperparameters
- Per-step metrics (loss, rewards, margins, LR)
- Validation results (loss decreased, margins increased, chosen > rejected, no NaN/Inf)

### Output: Checkpoint Directory Structure

```
/data-1/checkpoints/{model-short-name}-dpo/
├── config.json                 # Model architecture config
├── generation_config.json      # Generation defaults
├── model-00001-of-000XX.safetensors  # Model weights (sharded)
├── model-00002-of-000XX.safetensors
├── model.safetensors.index.json      # Shard index
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── merges.txt / vocab.json     # Tokenizer files
├── training_logs/
│   └── training_summary.json   # Metrics and validation results
└── checkpoint-NNN/             # Intermediate checkpoint (if save_steps hit)
```

---

## 4. Evaluation

### Running Evaluation

Evaluation uses `offline_eval.py` from the verl repository, and must run inside the **`verl-harness`** container:

```bash
docker run --rm --gpus all --ipc=host \
  -v /data-1:/data-1 \
  verl-harness:latest bash
```

Inside the container:

```bash
cd /data-1/verl07/verl

python recipe/joint_training/offline_eval.py \
  --model_path /data-1/checkpoints/{model-short-name}-dpo \
  --test_files \
    /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
    /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
    /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
    /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
    /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
    /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
    /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet \
  --n 3 \
  --tensor_parallel 8 \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_tokens 4096 \
  --gpu_memory_utilization 0.85 \
  --output_dir /data-1/checkpoints/{model-short-name}-dpo/inference_n3
```

### Parameter Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model_path` | str | (required) | Path to the model checkpoint directory |
| `--test_files` | str list | MATH-500 + AIME-2025 | Paths to benchmark dataset parquet files |
| `--n` | int | 8 | Number of responses per prompt (power-of-2 enables multi-k metrics) |
| `--n_per_dataset` | str list | [] | Per-dataset n overrides as `path:n` pairs |
| `--tensor_parallel` | int | 4 | Number of GPUs for tensor parallelism |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--top_p` | float | 0.95 | Top-p nucleus sampling |
| `--max_tokens` | int | 4096 | Maximum tokens per response |
| `--gpu_memory_utilization` | float | 0.85 | Fraction of GPU memory for vLLM |
| `--output_dir` | str | (required) | Directory for eval_metrics.json and eval_details.parquet |
| `--seed` | int | 42 | Random seed for reproducibility |

### Benchmark Datasets

All benchmark datasets are available in two variants: **plain** (no system prompt) and **with_system_prompt** (includes a chain-of-thought system prompt). Use the `_with_system_prompt` variants for models trained with thinking-style prompts.

| Benchmark | Samples | With System Prompt | Plain |
|---|---|---|---|
| **MATH-500** | 500 | `/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet` | `/data-1/dataset/MATH-500/math500-test.parquet` |
| **AIME-2025** | 30 | `/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet` | `/data-1/dataset/AIME-2025/aime-2025.parquet` |
| **AMC-2023** | 40 | `/data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet` | `/data-1/dataset/AMC23/amc23-test.parquet` |
| **AQUA** | 254 | `/data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet` | `/data-1/dataset/AQUA/aqua-test.parquet` |
| **GSM8K** | 1319 | `/data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet` | `/data-1/dataset/gsm8k/test.parquet` |
| **MAWPS** | 355 | `/data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet` | `/data-1/dataset/MAWPS/mawps-test.parquet` |
| **SVAMP** | 300 | `/data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet` | `/data-1/dataset/SVAMP/svamp-test.parquet` |

### Recording Results

After evaluation, record results in two files:

1. **`INFERENCE_RESULTS.md`** — Add a new EVAL-XX entry following the template:

```markdown
## EVAL-XX: EXP-YY {model description}

| Field | Value |
|---|---|
| **Source Experiment** | EXP-YY (description) |
| **Model Weights** | `/data-1/checkpoints/{model-short-name}-dpo/` |
| **Checkpoint Step** | N (final) |
| **Sub-Model** | N/A |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP |
| **Generation Params** | temperature=1.0, top_p=0.95, n=3, max_tokens=4096 |
| **Date** | YYYY-MM-DD |

### Results

| Benchmark | Samples | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |
```

2. **`EXPERIMENT_INDEX.md`** — Add a new EXP-XX entry documenting the training setup, dataset, and cross-referencing the EVAL entry.

Both files live in `/data-1/verl07/verl/recipe/joint_training/`.

### Output Files

Evaluation produces two output files in `--output_dir`:

| File | Contents |
|---|---|
| `eval_metrics.json` | Summary metrics per benchmark: mean@n, pass@k, maj@k, extraction_fail, plus generation parameters and timing |
| `eval_details.parquet` | Per-response raw data: data_source, ground_truth, acc, score, pred, verification_method, response_text, n |

---

## 5. Quick-Start Recipes

### Recipe A: New Model Checkpoint → Generate DPO Data → Train → Evaluate

End-to-end workflow for a new model checkpoint.

```bash
# === Step 1: Generate preference data (inside dpo-harness) ===
docker run --rm --gpus all --ipc=host -v /data-1:/data-1 dpo-harness bash -c '
  python dpo_pipeline/run_pipeline.py \
    --model /data-1/.cache/{your-model} \
    --num-rollouts 16 \
    --limit 2000 \
    --output-prefix dpo-{your-model-short} \
    --max-tokens 4096 \
    --temperature 0.7 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --strict
'

# === Step 2: Clean the pairs (inside dpo-harness) ===
docker run --rm --gpus all --ipc=host -v /data-1:/data-1 dpo-harness bash -c '
  python dpo_pipeline/clean_pairs.py \
    --pairs /data-1/dataset/dpo-{your-model-short}-pairs.jsonl \
    --extracted /data-1/dataset/dpo-{your-model-short}-extracted.jsonl \
    --output /data-1/dataset/dpo-{your-model-short}-pairs.jsonl
'

# === Step 3: Create and run training script (inside dpo-harness) ===
# First, copy and edit the training script:
#   cp dpo_pipeline/train_dpo_4b.py dpo_pipeline/train_dpo_{your_model}.py
#   Edit MODEL_NAME, DATASET_PATH, OUTPUT_DIR, LOG_DIR constants
docker run --rm --gpus all --ipc=host -v /data-1:/data-1 dpo-harness bash -c '
  python dpo_pipeline/train_dpo_{your_model}.py
'

# === Step 4: Evaluate (inside verl-harness) ===
docker run --rm --gpus all --ipc=host -v /data-1:/data-1 verl-harness:latest bash -c '
  cd /data-1/verl07/verl && \
  python recipe/joint_training/offline_eval.py \
    --model_path /data-1/checkpoints/{your-model-short}-dpo \
    --test_files \
      /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
      /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
      /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
      /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
      /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
      /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
      /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet \
    --n 3 \
    --tensor_parallel 8 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir /data-1/checkpoints/{your-model-short}-dpo/inference_n3
'

# === Step 5: Review results ===
cat /data-1/checkpoints/{your-model-short}-dpo/inference_n3/eval_metrics.json | python -m json.tool
```

### Recipe B: Switch to a Different Source Dataset

To generate preference data from a different dataset:

**Option 1 — Change the `SOURCE_DATA` constant:**

Edit `run_pipeline.py` and change the `SOURCE_DATA` variable:

```python
SOURCE_DATA = "/data-1/dataset/{your-dataset-dir}/{your-file}.jsonl"
```

The source JSONL must have the format: each line is a JSON object with a `messages` field containing a list of `[{role: "user", content: "..."}, {role: "assistant", content: "...", info: {reference_answer: "..."}}]`.

Then run `run_pipeline.py` as usual.

**Option 2 — Call the sub-scripts directly:**

```bash
# Extract prompts from custom dataset
python dpo_pipeline/extract_prompts.py \
  --input /data-1/dataset/{your-dataset}.jsonl \
  --output /data-1/dataset/custom-extracted.jsonl \
  --limit 2000

# Generate rollouts
python dpo_pipeline/batch_rollout.py \
  --input /data-1/dataset/custom-extracted.jsonl \
  --output /data-1/dataset/custom-rollouts.jsonl \
  --model Qwen/Qwen3-4B-Base \
  --num-rollouts 16 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9

# Build preference pairs
python dpo_pipeline/build_pairs.py \
  --input /data-1/dataset/custom-rollouts.jsonl \
  --output /data-1/dataset/custom-pairs.jsonl \
  --strict

# Clean pairs
python dpo_pipeline/clean_pairs.py \
  --pairs /data-1/dataset/custom-pairs.jsonl \
  --extracted /data-1/dataset/custom-extracted.jsonl \
  --output /data-1/dataset/custom-pairs.jsonl
```

### Recipe C: Run the Pipeline on a Different Model Family

To adapt the pipeline for a different model family (e.g., Gemma, Llama, Mistral):

**What to check:**

1. **Tokenizer compatibility**: The model must have a tokenizer loadable by `AutoTokenizer.from_pretrained()`. Verify it works:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("your-model", trust_remote_code=True)
   print(tok.pad_token, tok.eos_token)
   ```

2. **Chat template**: If the model is an SFT/chat model, its tokenizer must support `apply_chat_template()`. Use `--chat-template` in `run_pipeline.py` and the evaluation script will use it automatically.

3. **Max tokens**: Adjust `--max-tokens` based on the model's context window. The default of 4096 works for most models, but some may support more or less.

4. **vLLM support**: The model architecture must be supported by vLLM. Check the [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

5. **Training script adjustments**:
   - For models > 8B: Use `optim="adafactor"` and `precompute_ref_log_probs=True`
   - For models with different pad token conventions: Update the `if tokenizer.pad_token is None` block
   - Flash Attention 2: Requires `attn_implementation="flash_attention_2"` in `AutoModelForCausalLM.from_pretrained()` — verify your model supports it

6. **Think/answer format**: The pipeline assumes responses use `<think>...</think>` for reasoning and `\boxed{}` or `<answer>...</answer>` for answers. If your model uses a different format, you may need to adapt `build_pairs.py` and `answer_verify.py`.

---

## 6. File Layout Reference

### Pipeline Stage → Input → Output

| Stage | Script | Input | Output |
|---|---|---|---|
| Extract Prompts | `extract_prompts.py` | `/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl` | `/data-1/dataset/{prefix}-extracted.jsonl` |
| Extract (with offset) | `extract_prompts_offset.py` | `/data-1/dataset/EnsembleLLM-data/am_deepseek_r1_filtered_ad.jsonl` | `/data-1/dataset/{prefix}-extracted.jsonl` |
| Batch Rollout | `batch_rollout.py` | `/data-1/dataset/{prefix}-extracted.jsonl` | `/data-1/dataset/{prefix}-rollouts.jsonl` |
| Build Pairs | `build_pairs.py` | `/data-1/dataset/{prefix}-rollouts.jsonl` | `/data-1/dataset/{prefix}-pairs.jsonl` |
| Clean Pairs | `clean_pairs.py` | `/data-1/dataset/{prefix}-pairs.jsonl` + `{prefix}-extracted.jsonl` | `/data-1/dataset/{prefix}-pairs.jsonl` (cleaned) |
| Answer Verify | `answer_verify.py` | (utility module, imported by `build_pairs.py` and `clean_pairs.py`) | — |
| DPO Train (4B) | `train_dpo_4b.py` | `/data-1/dataset/dpo-4b-pairs.jsonl` | `/data-1/checkpoints/qwen3-4b-dpo/` |
| DPO Train (8B) | `train_dpo_8b.py` | `/data-1/dataset/dpo-8b-pairs.jsonl` | `/data-1/checkpoints/qwen3-8b-dpo/` |
| DPO Train (4B SFT) | `train_dpo_4b_sft.py` | `/data-1/dataset/dpo-4b-sft-pairs.jsonl` | `/data-1/checkpoints/qwen3-4b-sft-dpo/` |
| Evaluate | `offline_eval.py` | Model checkpoint + benchmark parquets | `eval_metrics.json` + `eval_details.parquet` |

### Naming Conventions

**Checkpoints:**
```
/data-1/checkpoints/{model-short-name}-dpo/
```
Examples:
- `/data-1/checkpoints/qwen3-4b-dpo/` — Qwen3-4B-Base DPO
- `/data-1/checkpoints/qwen3-8b-dpo/` — Qwen3-8B-Base DPO
- `/data-1/checkpoints/qwen3-4b-sft-dpo/` — Qwen3-4B SFT → DPO

**Datasets:**
```
/data-1/dataset/dpo-{model-short-name}-pairs.jsonl
```
Examples:
- `/data-1/dataset/dpo-4b-pairs.jsonl` — 6,013 preference pairs (Qwen3-4B-Base)
- `/data-1/dataset/dpo-8b-pairs.jsonl` — 7,934 preference pairs (Qwen3-8B-Base)
- `/data-1/dataset/dpo-4b-sft-pairs.jsonl` — 5,860 preference pairs (Qwen3-4B-SFT)

**Evaluation outputs:**
```
/data-1/checkpoints/{model-short-name}-dpo/inference_n{N}/
├── eval_metrics.json       # Summary metrics per benchmark
└── eval_details.parquet    # Per-response raw results
```

### All Scripts in `dpo_pipeline/`

| Script | Description |
|---|---|
| `run_pipeline.py` | End-to-end orchestrator (extract → rollout → pairs) |
| `extract_prompts.py` | Extract prompts and reference answers from source dataset |
| `extract_prompts_offset.py` | Extract prompts with offset for avoiding overlap |
| `batch_rollout.py` | Generate N model rollout responses per prompt using vLLM |
| `build_pairs.py` | Build DPO preference pairs from rollouts (verify + pair) |
| `clean_pairs.py` | Post-process pairs by re-verifying rejected responses |
| `answer_verify.py` | Dual-layer answer verification module (math_verify + regex) |
| `train_dpo_4b.py` | DPO training for Qwen3-4B-Base |
| `train_dpo_8b.py` | DPO training for Qwen3-8B-Base (with Adafactor optimizer) |
| `train_dpo_4b_sft.py` | DPO training for Qwen3-4B-SFT checkpoint |
| `smoke_test_dpo.py` | Minimal smoke test for DPO environment validation |
| `Dockerfile` | Docker image definition for `dpo-harness` |

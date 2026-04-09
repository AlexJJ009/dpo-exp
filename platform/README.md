# Platform Deployment Guide

## 1. Build Custom Image

On the platform's image builder, select a base image with Python 3.12 + CUDA 12.6.

Paste the content of `dockerfile_instructions.txt` into the Dockerfile instructions field.

Do NOT add the "Conda" software package (it overwrites Python 3.12 with 3.11).

## 2. Directory Layout on DolphinFS

```
/mnt/dolphinfs/.../lgx/
├── dpo-exp/                    <- git clone the repo here
├── dataset/                    <- training data + test benchmarks
│   ├── code/                   <- code-train.jsonl, example.json
│   ├── EnsembleLLM-data-processed/
│   │   ├── HumanEval/test.jsonl
│   │   ├── MBPP/test.jsonl
│   │   └── BigCodeBench/test.jsonl
│   └── dpo/                    <- DPO intermediates (auto-created)
├── .cache/                     <- model weights (e.g. Qwen3-4B-Base-Code-SFT/)
└── checkpoints/                <- training outputs (auto-created)
```

## 3. Setup Steps

```bash
# 1. Clone repo
cd /mnt/dolphinfs/.../lgx
git clone <repo-url> dpo-exp

# 2. Copy data and model weights to the directory layout above

# 3. Copy platform files to hope_dir
cp dpo-exp/platform/jupyter.sh  hope_dir/jupyter.sh
cp dpo-exp/platform/run.hope    hope_dir/run.hope

# 4. Edit jupyter.sh: set LGX_DIR to your actual path
# 5. Edit run.hope:   set docker image name to your custom image
```

## 4. Run

Submit the job via `run.hope`. The platform will:
1. Allocate GPU nodes
2. Start the container with your custom image
3. Run `jupyter.sh`, which:
   - Starts Jupyter Lab in background (for monitoring)
   - Runs the DPO pipeline via `experiments/run_4b_code_sft_code.sh`

## 5. Key Differences from Local Machine

| | Local (this machine) | Platform |
|--|--|--|
| Execution | `docker run` wraps each step | Direct execution inside container |
| `USE_DOCKER` | `1` (auto-detected) | `0` (set in jupyter.sh) |
| Base path | `/data-1` | `/mnt/dolphinfs/.../lgx` |
| GPU allocation | All GPUs always available | Managed by YARN/run.hope |

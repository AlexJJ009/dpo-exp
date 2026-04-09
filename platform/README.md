# Platform Deployment Guide

## 1. Upload Wheels to DolphinFS

On this machine (has internet), the wheels are pre-downloaded in `platform/wheels/` (23MB).
Upload them to your DolphinFS path:

```bash
# From this machine, copy to DolphinFS (adjust the target path)
scp -r platform/wheels/ <dolphinfs-host>:/mnt/dolphinfs/.../lgx/dpo-wheels/
```

Or from a Jupyter session on the platform:
```bash
# If you can access this machine from the platform
rsync -av <this-machine>:/data-1/dpo-experiment/platform/wheels/ /mnt/dolphinfs/.../lgx/dpo-wheels/
```

## 2. Build Custom Image

1. Select a base image that **already has vllm 0.12.0 + torch** (check other people's images)
2. Do NOT add "Conda" software package
3. Paste the content of `dockerfile_instructions.txt` into Dockerfile instructions
4. **Edit the path** in the first RUN line to match your DolphinFS wheels location

Note: `deepspeed` is not in the wheels (needs compilation). Install it at runtime
in `jupyter.sh` if your base image doesn't have it, or skip it if the base image
already includes it.

## 3. Directory Layout

```
lgx/
├── dpo-exp/          <- git clone the repo
├── dpo-wheels/       <- uploaded wheel files (23MB)
├── dataset/
│   ├── code/code-train.jsonl
│   └── EnsembleLLM-data-processed/HumanEval/test.jsonl ...
├── .cache/Qwen3-4B-Base-Code-SFT/checkpoint-38/
└── checkpoints/      <- auto-created
```

## 4. Run

Edit `jupyter.sh`: set `LGX_DIR` to your actual DolphinFS path.
Edit `run.hope`: set the docker image to your custom-built image.

Submit the job. The pipeline runs automatically.

## 5. What's in the Wheels

Only packages NOT in typical sglang/vllm base images (23MB total):
- trl 0.29.0 + accelerate + datasets
- math-verify + latex2sympy2-extended + pylatexenc
- Small dependencies (httpx, rich, etc.)

Packages expected from base image (NOT included):
- torch, transformers, tokenizers, numpy, vllm

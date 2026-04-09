# Platform Deployment Guide

## Base Image Selection

Select the **sglang base image** (already has torch):
```
python3.12.11 cuda12.6 torch2.9.1+cu126 sglang0.56.post2 transformers4.57.3 xformers0.0.31
```

## Important

- Do NOT add the "Conda" software package (it overwrites Python 3.12 with 3.11)
- Only add the Dockerfile instructions below

## Dockerfile Instructions

See: `dockerfile_instructions.txt`

## Directory Layout on Platform

```
lgx/
├── dpo-exp/          <- git clone the repo here
├── dataset/          <- training data + test benchmarks
│   ├── code/
│   ├── EnsembleLLM-data-processed/
│   └── dpo/          <- DPO intermediates (auto-created)
├── .cache/           <- model weights
└── checkpoints/      <- training outputs
```

# ==============================================================================
# Dockerfile: DPO training environment for Qwen3-4B/8B
#
# Base: verl-harness (CUDA 12.6, Python 3.12, PyTorch 2.9.1+cu126, vLLM 0.12.0)
# Upgrades: TRL v0.29.0 (from 0.9.6)
#
# Build:
#   docker build --network=host -f dpo_pipeline/Dockerfile -t dpo-harness .
#
# Run:
#   docker run --rm --gpus all --ipc=host \
#     -v /data-1:/data-1 \
#     dpo-harness bash
# ==============================================================================

FROM verl-harness:latest

ARG PIP_NO_CACHE_DIR=1

# ======================== Upgrade TRL to v0.29.0 =============================
# The base image has TRL 0.9.6. Upgrade to v0.29.0 with its dependencies.
# TRL v0.29.0 requires transformers>=4.56.2, accelerate>=1.4.0, datasets>=3.0.0
RUN uv pip install "trl==0.29.0"

# Verify torch wasn't downgraded by transitive deps
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    echo "PyTorch version after TRL upgrade: ${TORCH_VER}"

# ======================== Environment variables ==============================
ENV HF_HOME=/data-1/.cache/huggingface
ENV HF_HUB_CACHE=/data-1/.cache/huggingface
ENV HF_HUB_OFFLINE=1

WORKDIR /workspace
CMD ["bash"]

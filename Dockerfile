# ==============================================================================
# Dockerfile: Self-contained DPO training environment
#
# Two build modes:
#   1. From vLLM official image (default, for any machine):
#      docker build --network=host -t dpo-harness .
#
#   2. From platform base image (for company platform):
#      docker build --network=host \
#        --build-arg BASE_IMAGE=<platform-image-name> \
#        --build-arg INSTALL_TORCH=1 \
#        -t dpo-harness .
#
# Target versions:
#   PyTorch 2.9.1+cu126 | vLLM 0.12.0 | TRL 0.29.0 | DeepSpeed 0.18.9
#   Python 3.12 | CUDA 12.6
# ==============================================================================

ARG BASE_IMAGE=vllm/vllm-openai:v0.12.0
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1

# Set to 1 when base image does NOT have torch/vLLM pre-installed
ARG INSTALL_TORCH=0

# ======================== System deps ========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# ======================== Core ML stack (only if base image is bare) =========
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
      pip install --upgrade pip && \
      pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126 && \
      pip install vllm==0.12.0 && \
      pip install flash-attn --no-build-isolation; \
    fi

# ======================== DPO training stack =================================
RUN pip install --upgrade pip && \
    pip install "trl==0.29.0" "deepspeed==0.18.9"

# ======================== Math verification ==================================
RUN pip install math-verify latex2sympy2-extended pylatexenc

# ======================== Verify =============================================
RUN python -c "\
import torch; print(f'PyTorch:      {torch.__version__}'); \
import vllm;  print(f'vLLM:         {vllm.__version__}'); \
import trl;   print(f'TRL:          {trl.__version__}'); \
import deepspeed; print(f'DeepSpeed:    {deepspeed.__version__}'); \
import transformers; print(f'Transformers: {transformers.__version__}')"

# ======================== Environment ========================================
ENV HF_HUB_OFFLINE=1
ENV VLLM_USE_V1=1
ENV VLLM_NO_USAGE_STATS=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["bash"]

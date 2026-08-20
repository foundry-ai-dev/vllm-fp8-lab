# Slim vLLM serving image. The torch wheels pulled in by vLLM bundle the CUDA
# runtime, so a CUDA base image is unnecessary — the host (vast.ai) only has to
# provide the NVIDIA driver via the container runtime.
FROM python:3.12-slim

# openssh is required by vast.ai's ssh launch mode — its launch script runs
# ssh/sshd inside the container and hangs forever if they're missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates openssh-server openssh-client \
    && rm -rf /var/lib/apt/lists/*

# vLLM >= 0.17 is required for Qwen3.8 support; hf_transfer speeds up the
# weight download at container start.
RUN pip install --no-cache-dir "vllm==0.17.*" hf_transfer

# Runtime deps a bare slim base is missing, kept as a layer AFTER the big pip
# layer so cached hosts don't re-pull the ~6 GB vLLM layer:
# - tmux: vast.ai's default /root/.bashrc force-attaches it on ssh login
# - gcc + libc6-dev: triton JIT-compiles kernel launchers at engine startup
#   and dies with "Failed to find C compiler" without them
# - jq: extracting fields from API responses when curl-testing on the instance
# - nano: on-instance edits and reading the vllm source (slim base has no editor)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux gcc libc6-dev jq nano \
    && rm -rf /var/lib/apt/lists/*

# Weights are downloaded at container start onto the instance volume
# (HF_HOME), never baked into the image.
# FLASH_ATTN ships precompiled in the vllm wheel. The default backend for this
# model (FlashInfer) JIT-compiles CUDA at startup and requires the full nvcc
# toolkit, which this image deliberately does not carry.
ENV VLLM_MODEL=Qwen/Qwen3.8-27B-FP8 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_PORT=8000 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

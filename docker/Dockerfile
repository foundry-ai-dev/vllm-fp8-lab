# Slim vLLM serving image. The torch wheels pulled in by vLLM bundle the CUDA
# runtime, so a CUDA base image is unnecessary — the host (vast.ai) only has to
# provide the NVIDIA driver via the container runtime.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# vLLM >= 0.17 is required for Qwen3.8 support; hf_transfer speeds up the
# weight download at container start.
RUN pip install --no-cache-dir "vllm==0.17.*" hf_transfer

# Weights are downloaded at container start onto the instance volume
# (HF_HOME), never baked into the image.
ENV VLLM_MODEL=Qwen/Qwen3.8-27B-FP8 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_PORT=8000

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

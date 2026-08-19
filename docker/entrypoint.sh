#!/usr/bin/env bash
# Download the model weights, then serve them with vLLM (OpenAI-compatible API
# plus Prometheus metrics on the same port at /metrics).
#
# Runs either as the image ENTRYPOINT (vast.ai "docker run" mode) or backgrounded
# from an onstart script in ssh mode:
#   nohup /usr/local/bin/entrypoint.sh > /workspace/vllm.log 2>&1 &
#
# Tunables (env):
#   VLLM_MODEL     HF repo to serve        (default Qwen/Qwen3.8-27B-FP8)
#   VLLM_PORT      listen port             (default 8000)
#   MAX_MODEL_LEN  context window          (default 65536; raise if KV headroom allows)
#   GPU_UTIL       gpu-memory-utilization  (default 0.92)
#   KV_DTYPE       kv-cache-dtype          (default auto = bf16. fp8 KV forces the
#                                           FlashInfer backend, which JIT-compiles
#                                           CUDA and needs the full nvcc toolkit —
#                                           not present in this slim image)
#   SERVED_NAME    model name in the API   (default qwen3.8-27b)
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.92}"
KV_DTYPE="${KV_DTYPE:-auto}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"

command -v nvidia-smi >/dev/null || {
  echo "ERROR: no GPU visible (nvidia-smi not found)." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Download first so a network failure is distinguishable from a serve failure.
# ~28 GB for the FP8 checkpoint; hf_transfer saturates most vast.ai links.
echo "Downloading ${MODEL} to ${HF_HOME:-~/.cache/huggingface} ..."
hf download "$MODEL" >/dev/null
echo "Download complete."

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype "$KV_DTYPE" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder

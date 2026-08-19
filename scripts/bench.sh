#!/usr/bin/env bash
# Quick single-stream throughput check against the deployed endpoint.
#
#   ./bench.sh http://IP:PORT [max_tokens]
#
# Sends one non-streaming completion, computes tokens/sec from the usage block,
# then prints a few engine gauges from the vLLM /metrics endpoint.
set -euo pipefail

BASE="${1:?usage: $0 http://IP:PORT [max_tokens]}"
MAX_TOKENS="${2:-256}"
MODEL="${MODEL:-qwen3.8-27b}"

body=$(cat <<JSON
{"model": "$MODEL",
 "messages": [{"role": "user", "content": "Write a detailed explanation of how paged attention works in vLLM."}],
 "max_tokens": $MAX_TOKENS, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
JSON
)

start=$(date +%s.%N)
resp=$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
elapsed=$(echo "$(date +%s.%N) - $start" | bc)

completion=$(echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
echo "generated ${completion} tokens in ${elapsed}s"
echo "throughput: $(echo "scale=1; $completion / $elapsed" | bc) tok/s (single stream)"

echo
echo "engine metrics:"
curl -sf "$BASE/metrics" | grep -E '^vllm:(num_requests_(running|waiting)|gpu_cache_usage_perc)' || true

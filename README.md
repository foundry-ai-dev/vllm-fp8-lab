# vllm-fp8-lab

Serve a 27B-class open model with **FP8 weights on a single 48 GB GPU**, using a
slim vLLM docker image built in CI and rented GPU capacity on [vast.ai](https://vast.ai).

Default model: [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
(official FP8 release, 262K native context). The model is an env var — any FP8
checkpoint that fits the card works.

```
GitHub Actions ──build──▶ ghcr.io/foundry-ai-dev/vllm-fp8-lab
                                     │  pull image
                                     ▼
                          vast.ai 1x RTX 6000 Ada (48 GB)
                          ├─ entrypoint: hf download weights → NVMe (~28 GB)
                          └─ vllm serve ──▶ OpenAI API + /metrics on :8000
```

## Why FP8 on 48 GB

| | BF16 | FP8 |
|---|---|---|
| 27B weights | ~54 GB — **doesn't fit** | ~28 GB |
| KV cache headroom (at 0.92 util) | — | ~15 GB → 131K context (fp8 KV) |

The RTX 6000 Ada (sm89) has hardware FP8, and vLLM serves FP8 checkpoints on it
out of the box. FP8 KV cache (`--kv-cache-dtype fp8`) doubles context capacity
versus bf16 KV; bump `MAX_MODEL_LEN` toward the native 262144 if you need it.

## The image

`docker/Dockerfile` is deliberately small: `python:3.12-slim` + `pip install vllm`.
The torch wheels bundle the CUDA runtime, so no CUDA base image is needed — the
host only provides the driver. Weights are **not** baked in; `docker/entrypoint.sh`
downloads them to the instance NVMe at first start (hf_transfer, a few minutes on
a fast link), then execs `vllm serve` with Qwen3.8's recommended parsers
(`--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`).

CI (`.github/workflows/docker.yml`) pushes `:latest` + a git-sha tag to ghcr.io on
any change under `docker/`. The ghcr package is public so vast.ai hosts can pull
anonymously.

## Run it

Requires the `vastai` CLI with an API key set, and vLLM ≥ 0.17 in the image
(pinned in the Dockerfile). Prefer doing it by hand? [RUNBOOK.md](RUNBOOK.md)
walks through every step manually — rent, ssh, download weights, start the API.

```bash
# 1. find a machine (cheapest 1x RTX 6000 Ada with fast download)
./scripts/create-instance.sh search

# 2. rent it — pulls the image, downloads weights, starts serving
./scripts/create-instance.sh create <OFFER_ID>

# 3. wait for "Application startup complete" (5–10 min: image pull + 28 GB weights)
./scripts/create-instance.sh status <INSTANCE_ID>   # shows IP + mapped port
#    (or ssh in and: tail -f /workspace/vllm.log)

# 4. smoke test: streamed chat + tool-call round trip
python scripts/test_chat.py --base-url http://IP:PORT/v1

# 5. throughput
./scripts/bench.sh http://IP:PORT
```

### Sampling settings (per model card)

| Mode | temperature | top_p | extras |
|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | top_k 20 |
| Instruct / tool use | 0.7 | 0.80 | top_k 20, presence_penalty 1.5 |

The chat template opens assistant turns with `<think>`; the `qwen3` reasoning
parser splits that into `reasoning_content` so API clients get clean answers.

### Tunables (env on the instance)

| Var | Default | Notes |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3.8-27B-FP8` | any FP8 checkpoint that fits |
| `MAX_MODEL_LEN` | `131072` | native max 262144 |
| `GPU_UTIL` | `0.92` | gpu-memory-utilization |
| `KV_DTYPE` | `fp8` | `auto` for bf16 KV |

## Observability

vLLM exposes Prometheus metrics at `/metrics` on the same port. Point a
Prometheus scrape job at the mapped port to get engine dashboards — see
[prometheus-grafana-lab](https://github.com/foundry-ai-dev/prometheus-grafana-lab)
for a monitor node with prebuilt vLLM dashboards.

## Cost

A 1x RTX 6000 Ada on vast.ai typically runs **$0.50–0.80/hr**. Instances bill
until destroyed, not just stopped:

```bash
vastai destroy instance <INSTANCE_ID>
```

## License

MIT

# Manual runbook

Step-by-step CLI walkthrough of what the automation does, for running the whole
flow by hand: rent the GPU, ssh in, get the weights, start the API, test it,
tear it down. Useful for debugging, or for swapping any piece (model, image,
flags) interactively.

One vast.ai concept up front: standard instances **are** docker containers — you
choose the image when you rent, and the **host machine** pulls it before the
container starts. You never `docker pull` inside the instance. "Manual mode"
here means renting with the lab image but *without* the autostart command, so
the container just sits there with vLLM installed and you drive every step
yourself over ssh.

## 0. Prerequisites (once)

```bash
pip install vastai

# API key: cloud.vast.ai → avatar (top right) → Keys → API Keys (64-char hex)
vastai set api-key <YOUR_API_KEY>

# SSH key: same Keys page → SSH Keys tab → add your ~/.ssh/id_rsa.pub
# (or: vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)")
```

## 1. Find a machine

```bash
vastai search offers \
  'gpu_name=RTX_6000Ada num_gpus=1 disk_space>=120 inet_down>=500 rentable=true verified=true' \
  -o 'dph'
```

Reading the columns: `$/hr` is the rental price, `Net_down` matters because the
28 GB weight download happens on the instance (800 Mbps ≈ 5 min), `R` is host
reliability (stay ≥ 99), `Max_Days` is how long the machine is guaranteed to
stick around. Note the offer `ID` you want.

## 2. Rent it (manual mode — no autostart)

```bash
vastai create instance <OFFER_ID> \
  --image ghcr.io/foundry-ai-dev/vllm-fp8-lab:latest \
  --disk 120 \
  --env '-p 8000:8000' \
  --ssh --direct
```

- `--env '-p 8000:8000'` asks vast to map container port 8000 to a public port.
- `--ssh --direct` gives you a direct ssh daemon in the container.
- Omitting `--onstart-cmd` is what makes this manual: in ssh mode vast does not
  run the image ENTRYPOINT, so nothing starts until you start it.

The command prints `'new_contract': <INSTANCE_ID>` — that's your instance ID.

## 3. Wait for boot, then connect

```bash
vastai show instance <INSTANCE_ID>       # status: loading → running
vastai ssh-url <INSTANCE_ID>             # → ssh://root@IP:PORT

ssh -p <PORT> root@<IP>
```

While it says `loading`, the host is pulling the ~6 GB image; first rental of
this image on a given host takes a few minutes, later rentals hit the host cache.
If the console then shows "Successfully loaded \<image\>" but the instance sits
**inactive/stopped**, that's a known vast quirk — `vastai start instance
<INSTANCE_ID>` kicks it (see Gotchas).

End-to-end expectation for a fresh host: **15–25 min** from `create` to first
token (image pull + 28 GB weights + model load), dominated by real download
speed, which runs well below the listing (see Gotchas).

## 4. On the instance: weights + API

```bash
nvidia-smi                     # sanity: 1x RTX 6000 Ada, 49 GB

# You're already inside tmux — vast's /root/.bashrc auto-attaches a session on
# login, so the server below survives ssh drops without any extra step.

# Grab the weights (~28 GB → /workspace/hf, the instance NVMe volume).
# HF_HOME and HF_HUB_ENABLE_HF_TRANSFER=1 are already set in the image env.
hf download Qwen/Qwen3.8-27B-FP8

# Start the OpenAI-compatible API (same flags the entrypoint uses).
# FLASH_ATTN + bf16 KV avoids FlashInfer's JIT, which would need the full
# CUDA toolkit — see Gotchas below.
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3.8-27b \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Model load takes a couple of minutes after the download; you're up when the log
prints `Application startup complete`. Detach from tmux with `Ctrl-b d`
(reattach later with `tmux attach`).

Quick check from inside the instance:

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/metrics | grep '^vllm:' | head
```

## 5. Test from your laptop

Find the public port vast mapped to 8000:

```bash
vastai show instance <INSTANCE_ID>       # ports column: 8000/tcp -> <PUBLIC_PORT>
```

```bash
curl -s http://<IP>:<PUBLIC_PORT>/v1/models

# one completion by hand (instruct sampling; thinking disabled so a small
# max_tokens can't get eaten by the reasoning phase — see notes below):
curl -s http://<IP>:<PUBLIC_PORT>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"Say hello in five words."}],
       "temperature":0.7, "top_p":0.8, "presence_penalty":1.5, "max_tokens":128,
       "chat_template_kwargs": {"enable_thinking": false}}'

# or the repo's scripts:
python scripts/test_chat.py --base-url http://<IP>:<PUBLIC_PORT>/v1
./scripts/bench.sh http://<IP>:<PUBLIC_PORT>
```

**Primary quick test (PowerShell).** Set the base URL once, then re-edit only
`$q` between tests (up-arrow recalls the big line unchanged):

```powershell
$base = 'http://<IP>:<PUBLIC_PORT>'
$q = 'What does BF16 mean when referencing a LLM?'
$r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -ContentType 'application/json' -Body (@{ model = 'qwen3.8-27b'; messages = @(@{ role = 'user'; content = $q }); max_tokens = 4096; temperature = 0.7; top_p = 0.8; presence_penalty = 1.5; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5); $r.choices[0].message.content
```

Notes that save debugging time:
- `chat_template_kwargs.enable_thinking = $false` gives fast direct answers.
  Remove it for thinking mode, but then use `max_tokens` ≥ 2048 — the reasoning
  happens first, and a small budget dies inside it, returning EMPTY content with
  `finish_reason: "length"`. The reasoning text is in
  `$r.choices[0].message.reasoning`.
- Debug any odd response with:
  `"finish: $($r.choices[0].finish_reason) | tokens: $($r.usage.completion_tokens)"`

## 6. Knobs

All of these are just vllm serve flags when running manually (the automated
path exposes them as container env vars — see README):

| Change | Flag |
|---|---|
| Different model | positional arg — any FP8 checkpoint ≤ ~40 GB |
| Longer context | `--max-model-len 131072` fits in the bf16-KV budget; the native 262144 realistically needs fp8 KV |
| fp8 KV cache (doubles context capacity) | `--kv-cache-dtype fp8` — but this forces FlashInfer, which needs the full nvcc toolkit the slim image doesn't have (see Gotchas) |
| More/less VRAM headroom | `--gpu-memory-utilization 0.90–0.95` |

## Gotchas (all hit in the field, all fixed in the image)

- **"Successfully loaded \<image\>" but instance shows inactive/stopped** — vast
  sometimes leaves a fresh instance stopped after the image load. `vastai start
  instance <ID>` fixes it. Onstart runs on every start, not just the first.
- **Custom images need `openssh-server`** — vast's ssh launch mode runs ssh
  inside the container; without it the launch script loops forever
  (`ssh: command not found` in `vastai logs`) and onstart never runs.
- **vast's `/root/.bashrc` force-attaches tmux on login** — without tmux in the
  image, every ssh session dies with exit 127. Even with it, non-interactive
  commands (`ssh host 'cmd'`) get eaten by the auto-attach; for scripted access
  force a TTY and pipe keystrokes: `printf 'cmd\rexit\r' | ssh -tt -p PORT root@IP`.
- **Triton needs a C compiler at engine startup** — pip-installed vLLM on a
  slim base dies with `Failed to find C compiler`; `gcc` + `libc6-dev` fix it.
- **fp8 KV cache forces the FlashInfer backend**, which JIT-compiles CUDA and
  needs the full nvcc toolkit (the pip `nvidia-cuda-nvcc-cu12` wheel ships only
  `ptxas`, not `nvcc`). Without the toolkit, run `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
  with `--kv-cache-dtype auto` — FlashAttention is precompiled in the vllm wheel.
- **Advertised bandwidth is optimistic** — a host listing 800 Mbps delivered
  ~200 Mbps; budget download time accordingly.
- **Expected performance**: ~20 tok/s single-stream for this dense 27B FP8 on a
  6000 Ada — memory-bandwidth-bound (~34 tok/s theoretical ceiling). First
  request after startup pays ~60 s of CUDA graph warmup.

## 7. Teardown — don't skip

Instances bill until **destroyed** (stopped instances still bill for storage):

```bash
vastai destroy instance <INSTANCE_ID>
vastai show instances                    # verify: empty
```

## Appendix: fully manual, no custom image

To reproduce from a stock template instead (proves nothing magic is in the
image): rent the same offer with `--image vastai/pytorch` (or any CUDA python
image), then on the instance:

```bash
pip install "vllm==0.17.*" hf_transfer
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
```

and continue from step 4. A stock CUDA template already has gcc (and usually
nvcc, so FlashInfer's JIT may even work there); on a bare slim base you'd also
need `apt-get install gcc libc6-dev tmux openssh-server` — which is exactly
what the custom image pre-bakes, along with the two lines above, so cold
starts are pull-and-go.

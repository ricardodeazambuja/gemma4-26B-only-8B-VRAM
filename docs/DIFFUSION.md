# DiffusionGemma experiment (`feat/diffusiongemma`)

Testing [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)
on this rig (RTX 2070 8 GB / 31 GB RAM) and bridging it to pi.

## Problem

DiffusionGemma is a block-diffusion LM on the Gemma 4 base (25.2B total /
3.8B active MoE): instead of one token per forward pass it denoises a
256-token canvas in parallel, claiming 15–20 tok per pass. Potentially a big
energy lever for this repo — *if* it runs here, and *if* pi can talk to it.

Two obstacles:

1. **Upstream support is a day-old draft** — llama.cpp PR
   [#24423](https://github.com/ggml-org/llama.cpp/pull/24423)
   (`danielhanchen:diffusion-visual-updates`), CLI-only.
2. **No `llama-server` support** — the PR ships `llama-diffusion-cli` only, so
   there is no OpenAI endpoint for pi. That's the gap this branch fills.

## Solution

Three pieces, smallest possible diff against a moving draft PR:

1. **PR checkout + CUDA build** in `vendor/llama.cpp-diffusion/` (gitignored),
   built with the same conda toolchain as the main build
   (`llamacpp-cuda` env, CUDA 12.2, arch 75), target `llama-diffusion-cli`.
2. **JSONL stdio mode** patched into `examples/diffusion/diffusion-cli.cpp`
   (tracked as `patches/diffusion-cli-jsonl-mode.patch`): with
   `LLAMA_DIFFUSION_JSONL=1` and `-cnv`, the CLI reads
   `{"messages":[...]}` lines on stdin and emits `{"content":...,"ms":...}`
   lines on stdout. Model loads once, stays resident; stateless per request
   (full history each line — exactly how pi resends context). Why this and
   not patching `llama-server`: the server's slot/sampling machinery has no
   notion of canvas denoising, and the PR is a draft that will churn — a
   30-line additive patch survives rebases; a server integration won't.
3. **`scripts/diffusion-shim.mjs`** — zero-dependency Node HTTP server that
   keeps ONE CLI process alive and translates OpenAI
   `POST /v1/chat/completions` (incl. emulated SSE streaming) to the JSONL
   protocol. `scripts/run-diffusion-shim.sh` wires paths and the CPU/GPU
   split (`--cpu-moe` default, `NCMOE=n` to mirror run-server.sh tuning).

```
pi  ──openai-completions──►  shim :8082  ──jsonl stdio──►  llama-diffusion-cli (model resident)
```

Register in pi (`~/.pi/agent/models.json`): provider `diffusion`, api
`openai-completions`, baseUrl `http://127.0.0.1:8082/v1`, model id
`diffusiongemma-26b-a4b`.

## How to run

```bash
# 1. build (one-off, ~15 min)
cd vendor/llama.cpp-diffusion && git apply ../../patches/diffusion-cli-jsonl-mode.patch  # if fresh checkout
mamba run -n llamacpp-cuda cmake --build build -j --target llama-diffusion-cli

# 2. model (one-off, 16.8 GB → ../Gemma4/models/diffusiongemma-26b-a4b/)
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF --include "*Q4_K_M*" \
   --local-dir ../Gemma4/models/diffusiongemma-26b-a4b

# 3. run (stop the regular server first — both models don't fit in 31 GB)
../Gemma4/scripts/stop-server.sh
./scripts/run-diffusion-shim.sh
```

## Caveats / status

- **Prefill is paid every turn.** The PR keeps no KV state across turns; each
  request re-prefills the full history. Fine for smoke tests, brutal for long
  pi sessions — the opposite of the R1 cache discipline the extensions rely
  on. Real fix belongs upstream.
- **No tool calls yet.** The shim folds tool-role messages into user turns
  and ignores `tools`; pi extensions that register tools will not fire for
  this provider. Chat + the passive extensions (plan/memory injection) work.
- **No QAT quant exists** — Q4_K_M will lose more quality than the main
  model's QAT Q4_K_XL.
- **Draft PR** — expect rebases; keep the patch additive.

## Results

(to be filled after the smoke test: load success, RAM/VRAM split, effective
tok/s vs the ~30 tok/s autoregressive baseline, J/token if stats allow)

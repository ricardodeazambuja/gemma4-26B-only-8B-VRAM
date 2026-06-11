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

## Results (RTX 2070 8 GB / 31 GB RAM, Q4_K_M, `-ngl 99 --cpu-moe -c 4096`)

**It runs, and the bridge works.** JSONL mode + shim produced coherent chat
through the OpenAI endpoint on the first try.

Measured vs predicted memory:

| | predicted | measured |
|---|---|---|
| Host RSS (experts on CPU + buffers) | ~16–17 GiB | **16.5 GiB** |
| VRAM (`-n 2048`) | 3–4 GiB | **7.88 GiB** — MHA worst-case KV + ubatch buffers won |

Architecture notes (from the GGUF): 30 layers, **MHA not GQA** (16 heads ×
512-dim K/V on full-attn layers, 256-dim on SWA layers, window 1024,
dual-RoPE 1M/10k, `attention.causal=false`, vocab 262144, ctx_train 256k).
MHA makes KV ~0.92 MiB/token-layer on full layers — context is the VRAM
budget's enemy here, not weights.

Timing (first turn, all experts on CPU, greedy): 38.8 s/turn for a short
prompt; entropy-bound sampler ran 17/48 steps before converging, ~2.0 s per
denoising step → effective ~6 tok/s. Below the ~30 tok/s AR baseline, but
untuned: `NCMOE=n` (experts partially on GPU) is unexplored, and per-step
cost is dominated by the CPU expert forward, exactly like AR prefill.

**Crash found:** with `-n 2048` (→ `n_ubatch` 4096) VRAM sat at 7.9/8.0 GiB;
a long prompt (pi's full system prompt) needed one more 363 MiB compute
buffer and the PR **aborts** on the failed alloc
(`GGML_ASSERT(m.pkv_buf != nullptr)`, diffusion-gemma.cpp:726) instead of
returning an error — taking the whole process down. Mitigated by `-n 1024`;
worth reporting upstream (graceful failure), and the shim should auto-restart
its child.

**pi end-to-end: ✅** With `-n 1024` (VRAM 6.4 GiB idle / 7.0 GiB after a
turn) and a slim system prompt:

```
$ pi -p --provider diffusion --model diffusiongemma-26b-a4b \
     --system-prompt "You are a concise assistant." \
     "what kind of language model are you?"
I am Gemma 4, a large language model.
```

pi's *default* coding system prompt + tool schemas do not fit ctx 4096 —
usable pi sessions need either a trimmed prompt or more context, which on
8 GB VRAM means trading `-ngl` (CPU attention layers) for KV room.

## Speed verdict: loses to autoregressive on this hardware — decisively

NCMOE sweep, fixed prompt, greedy (identical 162-token output everywhere, so
times are directly comparable). AR baseline: ~30 tok/s decode (§8, NCMOE=22).

| Config | s/step | effective tok/s | VRAM |
|---|---|---|---|
| all experts CPU, `-fa on` forced | ~4.4 | 1.5 | 6.4 GiB |
| NCMOE 28 (2 expert layers GPU), `-fa on` | — | 3.2 | 7.4 GiB |
| NCMOE 27 (3 layers GPU), `-fa on` | — | 3.6 | 7.9 GiB |
| NCMOE 26 (4 layers GPU) | **OOM** (wanted +2.6 GiB) | — | — |
| all experts CPU, `-fa auto` | 2.23 | 4.1 | — |
| **NCMOE 27, `-fa auto` (best)** | **1.76** | **~5.2 (est)** | 7.9 GiB |

- `-fa on` must NOT be forced: the FA tensor lands on CPU with CPU-resident
  experts, and forced-on FA ran 2.7× slower than auto (which disables it).
- Best tunable config is **~5.8× slower than the AR baseline**, and the
  VRAM wall (MHA KV + canvas buffers) caps GPU experts at 3 of 30 layers.

**Why it can't win here:** block diffusion spends ~10,750 token-forwards to
emit 162 tokens (256-token canvas × ~21 steps × 2 blocks) — **~66× the
per-token compute of AR decode**. Its bet is that compute is nearly free
(whole model GPU-resident, massive parallelism). On 8 GB, 27/30 expert
layers live on the CPU, so each canvas pass costs ~2 s and the 66×
multiplier lands on the slowest path. The same split that gives AR 30 tok/s
gives diffusion 5. The headline "1100 tok/s on H100 FP8" needs the weights
in VRAM; on a 24 GB consumer card (Q4_K_M fully resident) this model would
plausibly beat AR — that is the hardware where this experiment becomes
interesting again.

**What survives the negative result:** the JSONL stdio mode
(`patches/diffusion-cli-jsonl-mode.patch`, upstreamable), the shim pattern
(any future diffusion arch with server-less support can reuse it), the
OOM-abort finding worth reporting on PR #24423, and these measurements.

## Lessons Learned & System Integration Updates (June 2026)

During active deployment and testing on the RTX 2070 8 GB VRAM GPU, we discovered critical bottlenecks and resolved them as follows:

### 1. The Double-OOM Buffering Wall
* **Graph Compute Buffer OOM:** When generating at the default canvas size `-n 1024`, `llama-diffusion-cli` allocates a massive **3.14 GB** compute buffer for parallel denoising. When combined with offloaded experts (e.g. `NCMOE=22`), the static model plus this buffer instantly overflows 8 GB cards. 
  * **Fix:** We reduced the default canvas size parameter from `-n 1024` to **`-n 512`** in [run-diffusion-shim.sh](file:///home/ricardodeazambuja/backup/GitStuff/Gemma4-diffusion/scripts/run-diffusion-shim.sh). This halves the compute buffer to **~1.5 GB**, reclaiming over **1.6 GB of VRAM** and allowing several expert layers (`NCMOE=27` or `28`) to fit on the GPU.
* **Inference-Time Lazy Allocation OOM:** The CLI allocates a **256 MiB** self-conditioning buffer (`sc_dev_buf`) on the *first turn*. This means a model can load successfully but crash with a hard assertion (`GGML_ASSERT(m.sc_dev_buf != nullptr)`) as soon as the first prompt is sent.
  * **Fix:** Keep NCMOE at **`28`** or higher to leave at least 400+ MiB of VRAM headroom, or run with `--diffusion-gpu-sampling off` (disabling device-resident self-conditioning).

### 2. Process Management & VRAM Leaks
* **Orphaned CLI Processes:** When the Node HTTP shim server exits or crashes, the child `llama-diffusion-cli` can sometimes be orphaned. Since the child doesn't listen on a network port, port-based checks did not detect it, leaving a stale model process consuming **7.7 GB of VRAM**.
  * **Fix:** Adapted [stop-server.sh](file:///home/ricardodeazambuja/backup/GitStuff/Gemma4-diffusion/scripts/stop-server.sh) to automatically run a process-reaping command (`pkill -f "llama-diffusion-cli"`) whenever you stop the diffusion server.

### 3. Worktree Resource Sharing
* **Binary & Model Paths:** As a separate Git worktree, the `Gemma4-diffusion` directory lacked the compiled `vendor/llama.cpp` CUDA backend and the `models/` weight directories, causing the script to default to Vulkan and fail to find the GGUF models.
  * **Fix:** Symlinked the main checkout's `models/` and `vendor/llama.cpp/` folders to the worktree to share models and CUDA build outputs seamlessly without duplicating disk storage.

### 4. Client Context Compaction Loop
* **Models Sync:** If `pi`’s local `models.json` context window is smaller than the prompt, it goes into an infinite compaction loop.
  * **Fix:** Adapted [configure-pi.sh](file:///home/ricardodeazambuja/backup/GitStuff/Gemma4-diffusion/scripts/configure-pi.sh) to support updating the context window for any provider via `PROVIDER=diffusion`, and wired `start.sh` to sync the window automatically during startup.

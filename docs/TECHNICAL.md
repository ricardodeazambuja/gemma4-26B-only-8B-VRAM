# Technical write-up: running Gemma 4 26B-A4B (QAT) on an 8 GB GPU with `pi`

This document explains **how** the setup in this repo works, **why** each decision was made,
and the **caveats** discovered along the way. It is the engineering companion to the
[README](../README.md) (quickstart).

- [1. Goal & constraints](#1-goal--constraints)
- [2. The model](#2-the-model)
- [3. Architecture: how the pieces fit](#3-architecture-how-the-pieces-fit)
- [4. The `--cpu-moe` memory split](#4-the---cpu-moe-memory-split)
- [5. Why not Ollama](#5-why-not-ollama)
- [6. The CUDA-version wall (and the Vulkan workaround)](#6-the-cuda-version-wall-and-the-vulkan-workaround)
- [7. Building llama.cpp against the local CUDA](#7-building-llamacpp-against-the-local-cuda)
- [8. Performance analysis](#8-performance-analysis)
- [9. Tuning](#9-tuning)
- [10. Wiring up `pi`](#10-wiring-up-pi)
- [11. Caveats & gotchas](#11-caveats--gotchas)
- [12. Reproducibility](#12-reproducibility)
- [13. Running bigger models](#13-running-bigger-models)
- [14. Multimodal: images via the mmproj](#14-multimodal-images-via-the-mmproj)

---

## 1. Goal & constraints

Run **Gemma 4 26B-A4B** (a 26B-parameter mixture-of-experts model with 4B active per token),
in the unsloth **QAT 4-bit GGUF** (~14 GB on disk), on a laptop, and drive it with
[`pi`](https://github.com/badlogic/pi-mono) (a local coding agent), reproducibly.

**Test machine:**

| | |
|---|---|
| GPU | NVIDIA RTX 2070 Max-Q — **8 GB VRAM**, compute capability **7.5** (Turing), driver **535.309** |
| Driver max CUDA | **12.2** (this matters — see §6) |
| CPU | Intel Core i7-8750H (6 cores / 12 threads) |
| RAM | 31 GB |
| OS | Ubuntu 22.04, glibc 2.35 |

The headline tension: a **14 GB** model cannot fit in **8 GB** of VRAM. The solution is to split
the model across VRAM and system RAM in a way that exploits the MoE structure (§4).

---

## 2. The model

`unsloth/gemma-4-26B-A4B-it-qat-GGUF` → `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`.

Architecture, read straight from the GGUF metadata (we parsed the header by hand — there was no
`gguf` Python module in the env):

| Key | Value |
|---|---|
| `general.architecture` | `gemma4` |
| `gemma4.block_count` | **30** layers |
| `gemma4.expert_count` | **128** experts per layer |
| `gemma4.expert_used_count` | **8** (top-8 routing) |
| `gemma4.embedding_length` | 2816 |
| `gemma4.feed_forward_length` | 2112 |
| `gemma4.expert_feed_forward_length` | 704 |
| `gemma4.attention.head_count` | 16 |
| `gemma4.attention.head_count_kv` | mostly 8, every 6th layer 2 (sliding-window pattern) |
| `gemma4.context_length` | 262144 (256K trained) |

llama.cpp reports it as `gemma4 26B.A4B`, **25.23 B** params, **13.26 GiB** in this quant.

Two terms worth defining (both detailed in unsloth's
[Gemma 4 QAT guide](https://unsloth.ai/docs/models/gemma-4/qat)):

- **QAT (Quantization-Aware Training):** the model was fine-tuned with quantization simulated in the
  forward pass, so the 4-bit weights keep much more quality than naive post-training quantization —
  unsloth cites **~72% lower memory** at near-original quality. This is what makes a 4-bit 26B model
  genuinely usable.
- **UD-Q4_K_XL:** unsloth's "Unsloth Dynamic" quant — different tensors get different bit-widths
  (important ones kept higher) rather than a uniform Q4. It also fixes a scale-format mismatch when
  converting the QAT weights to GGUF: per unsloth, naive `Q4_0` conversion of this model lands at
  **70.2%** top-1 accuracy, while the dynamic method reaches **85.6%** (+15.4) *and* is smaller, by
  matching the BF16 QAT scales much more exactly (99.96% vs 24.77% byte-exactness).

Per the same guide, the recommended **sampling** settings are temperature **1.0**, top-p **0.95**,
top-k **64**.

The repo is **public**: no Hugging Face token is needed, and `scripts/setup.sh` downloads the file
automatically via `huggingface_hub`.

---

## 3. Architecture: how the pieces fit

```
┌──────┐   OpenAI /v1/chat/completions    ┌───────────────── llama-server ─────────────────┐
│  pi  │ ───────────────────────────────▶ │  GPU (8 GB):  attention + KV cache + N expert   │
│      │   http://127.0.0.1:8080/v1       │               layers (CUDA or Vulkan)           │
└──────┘                                  │  RAM:         remaining MoE expert FFN weights  │
                                          │               (via --cpu-moe / --n-cpu-moe)     │
                                          └─────────────────────────────────────────────────┘
```

`pi` talks to the model over the **OpenAI chat-completions API**. Its provider config
(`~/.pi/agent/models.json`) points at any OpenAI-compatible endpoint — ollama, llama-server,
vLLM, a hosted API, etc. We register a `llamacpp` provider with `baseUrl
http://127.0.0.1:8080/v1` and `api: openai-completions`.

This keeps the **inference engine decoupled from the agent**: we can run llama-server on Vulkan
or on a custom CUDA build behind the same endpoint and `pi` never changes. The engine choice is
therefore driven purely by what features we need from it — namely `--cpu-moe` (§5).

---

## 4. The `--cpu-moe` memory split

A dense 26B model would need ~14 GB resident wherever it computes. An **MoE** model is different:
each token is routed to only **8 of 128** experts per layer. The experts are most of the
*parameters* but only a fraction of them are *touched per token*.

`llama.cpp`'s `--cpu-moe` exploits this by placing tensors on different devices:

| Tensor class | Device | Why |
|---|---|---|
| Attention (Q/K/V/O), embeddings, router, norms | GPU VRAM | small, hit every token, latency-sensitive |
| KV cache | GPU VRAM | grows with context, needs fast access |
| **MoE expert FFN weights** | **system RAM** | the bulk of the 26B params, but sparse per-token |

- `--cpu-moe` (`-cmoe`): **all** layers' experts → RAM.
- `--n-cpu-moe N` (`-ncmoe N`): experts of the **first N** layers → RAM; the rest stay on GPU.
  Lower N ⇒ more experts on GPU ⇒ faster, but more VRAM.

`--cpu-moe` moves **only the expert FFN weights**. Everything else — the attention weights,
embeddings, router, norms, and the **KV cache (the context)** — stays on the GPU. The context is
therefore *not* in system RAM; it occupies VRAM alongside the model weights, which is why context
length and on-GPU experts draw from the same 8 GB budget (§9).

The KV cache stays small: at `CTX=16384` it is only ~0.6 GB here, kept down by flash attention
(`-fa auto`) and Gemma 4's **sliding-window attention** (every 6th layer uses just `KV=2` heads
instead of 8). Its size depends on context length and model dimensions — **not** on the weight
quant — so a higher-precision quant costs the *same* VRAM for context.

`--cpu-moe` is also **backend-agnostic** — it is a tensor-placement directive, so the RAM/VRAM
split is identical whether the GPU backend is CUDA or Vulkan (§6).

VRAM at `CTX=16384`: `--cpu-moe` (0 expert layers on GPU) ≈ 3.5 GB; `NCMOE=22` (8 expert layers on
GPU) ≈ 6.8 GB. On an 8 GB card, `NCMOE=22` is about the ceiling.

---

## 5. Why not Ollama

Ollama is a popular local server and `pi`'s default provider, so it's the obvious first choice —
but it **cannot do `--cpu-moe`.** It performs only automatic *whole-layer* GPU offload; it cannot
keep a layer's attention on the GPU while putting that same layer's experts on the CPU, and there
is no environment variable, Modelfile parameter, or CLI flag for per-expert placement. (Tracked
upstream: [ollama#11772](https://github.com/ollama/ollama/issues/11772),
[ollama#14579](https://github.com/ollama/ollama/issues/14579).)

`llama.cpp`'s `llama-server` does have `--cpu-moe`, and since `pi` only needs an OpenAI endpoint
(§3), pointing it at llama-server instead is a drop-in swap. That is the whole reason for the
engine choice — nothing about `pi` requires either one.

---

## 6. The CUDA-version wall (and the Vulkan workaround)

The fast path is conda-forge's prebuilt `llama.cpp`. conda-forge ships **CUDA 12.9** (`cuda129`)
and **CUDA 13.0** (`cuda130`) variants. Both **crash at compute** on this machine:

```
CUDA error: device kernel image is invalid
```

### Why

This is a SASS-too-new failure. Driver 535 supports up to **CUDA 12.2**. The conda binary's
compiled GPU kernels (cubins/SASS) were produced by **CUDA 12.9**'s `nvcc`.

A subtle point that cost us time: **CUDA "minor-version compatibility" covers the runtime *API*,
not the compiled SASS.** An app linked against the 12.9 *runtime libraries* can call into a 12.2
driver — but a *kernel binary* emitted by the 12.9 toolchain is not guaranteed loadable by the
12.2 driver. SASS forward-compatibility across the whole 12.x line is not promised (and the
cross-major "forward compatibility" package is datacenter-GPU-only). The conda build ships SASS
without usable PTX fallback for this case, so the driver rejects the image at first kernel launch.

Tellingly, `llama-server --list-devices` *succeeds* on the broken build — enumerating devices
does not load a kernel. The crash only appears at the first real compute (model warmup). This
later informed how we smoke-test our own build (§11).

### The fix that needs no build: Vulkan

The conda `cuda129` package **also compiles in the Vulkan backend.** Vulkan talks to the GPU
through the driver's own Vulkan ICD (shipped with driver 535) — there is no CUDA-toolkit-version
wall. Select it with `--device Vulkan1` (the NVIDIA device; `Vulkan0` is the Intel iGPU). Because
`--cpu-moe` is backend-agnostic (§4), the experts-in-RAM / attention-on-GPU split is identical.

`scripts/run-server.sh` defaults to Vulkan and auto-detects the NVIDIA Vulkan device:

```bash
llama-server --list-devices | grep -iE 'Vulkan[0-9]+: NVIDIA'   # -> "Vulkan1: NVIDIA ..."
```

Vulkan works and is the zero-build default. But it is **slow** on this Turing GPU (§8) — which is
why we also build a native CUDA backend.

---

## 7. Building llama.cpp against the local CUDA

`scripts/build-llama-cuda.sh` builds llama.cpp from source against the CUDA version the driver
actually supports, so the kernels load. It is fully auto-detecting.

### Detection

```bash
DRIVER_CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' ...)   # -> 12.2
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')   # -> 75
```

### Isolated toolchain (nothing system-wide)

```bash
mamba create -n llamacpp-cuda -c conda-forge \
  cuda-toolkit=12.2 cuda-version=12.2 \
  gxx_linux-64=12 gcc_linux-64=12 \
  cmake ninja git make
```

- **`cuda-toolkit=12.2`** — cubins emitted by the 12.2 `nvcc` load natively on the 12.2 driver.
- **`gcc 12`** — `nvcc` 12.2 only accepts a host compiler up to GCC 12. The conda-forge default
  GCC (14) would be rejected, so we pin it. Configure confirmed:
  `The CUDA compiler identification is NVIDIA 12.2.140 with host compiler GNU 12.4.0`.

### Configure & build

```bash
cmake -S vendor/llama.cpp -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_HOST_COMPILER=<env>/bin/x86_64-conda-linux-gnu-g++ \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_RPATH=<env>/lib
cmake --build build -j --target llama-server llama-cli llama-bench
```

Notes:

- `CMAKE_CUDA_ARCHITECTURES=75` builds SASS for sm_75 only (the RTX 2070) — fast build, smallest
  binary. By default the conda build targets a dozen arches.
- `CMAKE_CUDA_HOST_COMPILER` is set **explicitly** to the conda GCC 12. `nvcc`'s default host
  compiler is plain `gcc`/`cc` on `PATH`, which can silently resolve to the *system* compiler.
- `CMAKE_BUILD_RPATH=<env>/lib` bakes the path to `libcudart.so.12` etc. into the binary so it
  finds the CUDA runtime at launch without `LD_LIBRARY_PATH` juggling.
- `LLAMA_CURL=OFF` drops a build dependency we don't need (we download models separately).
- `LLAMA_REF` defaults to the **latest `b*` release tag** (resolved with `git ls-remote ... | sort
  -V | tail -1`, which handles `bNNNN` numerically) rather than `master`, for reproducibility.

### Smoke test

The script ends by running an **actual decode** (`llama-bench ... -n 8`), not `--list-devices` —
because the broken conda build passed `--list-devices` too (§6). A printed result row
(`... | 28.00 ± 0.68`) proves the kernels load and run.

`run-server.sh` auto-uses the built binary and its env when invoked with `BACKEND=cuda`.

---

## 8. Performance analysis

All numbers: RTX 2070 8 GB, `CTX=16384`, `NCMOE=22` (8 expert layers on GPU, 22 in RAM), same
model.

| Backend | Tool | Prompt (pp) | **Generation (tg)** |
|---|---|---|---|
| CPU only (`--device none -ngl 0`) | llama-bench | 3.1 t/s | **1.9 t/s** |
| Vulkan (stock conda build) | llama-bench | 47 t/s | **4.35 t/s** |
| Vulkan (stock conda build) | server | — | **4.9 t/s** |
| **CUDA (built from source)** | llama-bench | 63 t/s | **25.25 t/s** |
| **CUDA (built from source)** | server | — | **23.5 t/s** |

(Token-gen is run-to-run noisy on a laptop ±10–20% from background load — e.g. CPU-only measured
1.9 and 2.4 tok/s on two runs. Treat these as representative, not exact.)

### How these were measured

Two tools, both from the relevant env. **`llama-bench`** is the clean micro-benchmark (loads the
model, runs `pp`/`tg`, prints tokens/s); **server** numbers are `predicted_per_second` from a real
`/v1/chat/completions` request — what `pi` actually experiences.

```bash
M=models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

# CPU only — no GPU at all. --device none excludes every offload device; -ngl 0 keeps all layers
# on the CPU. (llama-bench accepts --device none; --cpu-moe is a server-only flag.)
mamba run -n llamacpp      llama-bench -m $M --device none -ngl 0           -n 64 -p 32

# Vulkan, 8 expert layers on GPU (the NCMOE=22 config)
mamba run -n llamacpp      llama-bench -m $M --device Vulkan1 -ngl 99 --n-cpu-moe 22 -n 64 -p 64

# CUDA (source build), same split
mamba run -n llamacpp-cuda vendor/llama.cpp/build/bin/llama-bench \
                                       -m $M -ngl 99 --n-cpu-moe 22         -n 64 -p 64

# server-side (start the server, then time a request):
NCMOE=22 BACKEND=cuda bash scripts/run-server.sh    # or drop BACKEND for Vulkan
curl -s localhost:8080/v1/chat/completions -d '{"model":"gemma-4-26b-a4b-qat",
  "messages":[{"role":"user","content":"Write a haiku about RAM."}],"max_tokens":300}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["predicted_per_second"])'
```

**Verifying the CPU-only number really used no GPU.** Because this machine's conda CUDA kernels
crash on *any* launch (§6), a CPU-only run that completes is already proof no CUDA kernel ran. We
also confirmed it directly: with the server stopped, while the CPU `llama-bench` ran,
`nvidia-smi --query-compute-apps` showed **no compute process** and GPU memory stayed at the idle
desktop baseline (~302 MiB, zero delta). The 6–36% `utilization.gpu` seen meanwhile is *graphics*
(desktop compositing), not compute — `utilization.gpu` counts both. The `dev` column in
llama-bench's own output also reads `none`.

### Two conclusions

**(a) The GPU is not slower than the CPU — Vulkan was just slow.** Pure CPU is ~1.9 tok/s; Vulkan
roughly doubles it; CUDA is **~5× the Vulkan number** (4.8× server-to-server, 5.8× bench-to-bench)
at the *same* CPU/GPU split. Since the
split is identical, the only thing that changed is GPU-side efficiency — so the Vulkan MoE path
(Turing has no tensor cores, and the hybrid CPU↔GPU dispatch is poorly pipelined there) was the
real bottleneck.

**(b) RAM bandwidth is *not* the limiter here.** It is tempting to assume that with 22 of 30
layers' experts in RAM, the model is memory-bound. The arithmetic says otherwise: only **8 of 128
experts fire per token**, so each token streams roughly *tens of MB* of expert weights per layer,
not the full ~0.45 GB. At 25 tok/s across 22 CPU-resident layers that's on the order of ~15 GB/s
of RAM reads — comfortably under dual-channel DDR4 bandwidth. The earlier "it's RAM-bound"
hypothesis was wrong, and measuring (rather than trusting the rule-of-thumb "CUDA is only ~15–40%
faster than Vulkan") is what caught it.

**Lesson:** measure backends on your actual hardware. For MoE token-generation on an older NVIDIA
GPU, Vulkan can be ~5× slower than CUDA, which is well outside the usual gap.

**Recommendation:** if you care about tokens/sec, **use the CUDA build** (`build-llama-cuda.sh`,
then `BACKEND=cuda`). Vulkan is the zero-build fallback for when you can't build or aren't on
NVIDIA — not the throughput choice.

### Why pi feels slower than the benchmark tok/s

The benchmark figures above are a best case: a tiny prompt, a fresh context, and a short
generation. Interactive use through `pi` is the opposite on all three counts, so the *felt* latency
is much higher than the decode rate suggests. The token-generation rate is real — but it is only
one of three contributors to wall-clock time, and usually the smallest.

A representative single reply (5,639-token prompt — system prompt + tool definitions + history +
file contents — asking a trivial question):

| Phase | Time | Visible? |
|---|---|---|
| Prefill (process the prompt) | ~14 s | no output yet (time-to-first-token) |
| Generation (331 tokens @ ~20 tok/s) | ~17 s | mostly hidden reasoning |
| **Total** | **~31 s** | for a one-token answer |

The three contributors:

1. **Time-to-first-token (prefill).** Before the first output token, the server must process the
   entire prompt. `pi`'s prompts are large, and prefill is partly **CPU-bound** here — the
   `--cpu-moe` experts run on the CPU for *every* prompt token — so TTFT grows with context and is
   slower than a fully-GPU model. The server's **prompt cache** mitigates this on later turns: the
   constant prefix (system prompt + tool defs) is prefilled once, so subsequent turns only reprocess
   new tokens (your message + tool results). The first turn is the worst.

2. **Hidden reasoning tokens.** Gemma 4 thinks before answering (`thinking = 1`); the chain-of-
   thought goes to `reasoning_content`, not `content`. At a high thinking level most generated
   tokens are *reasoning you don't see*, so you pay decode time while the visible answer appears to
   stall. The `pi` thinking level (`--thinking off|low|medium|high`, or `defaultThinkingLevel` in
   `~/.pi/agent/settings.json`) trades reasoning quality for responsiveness; the decode rate itself
   is unchanged, only how many tokens get generated.

3. **The agentic loop.** A single `pi` task is many model calls — generate a tool call, run it, feed
   the result back, reason again. Each step is a fresh prefill + reasoning + generation cycle, so
   wall-clock stacks across steps.

None of these is the model running slower than measured; the decode rate stays ~low-20s tok/s. The
gap is prefill latency + invisible reasoning + multi-step orchestration, which the raw tok/s number
does not capture. Levers if responsiveness matters: lower the thinking level, keep the context lean,
and use the CUDA backend (faster prefill *and* decode).

---

## 9. Tuning

`--n-cpu-moe N` (env `NCMOE`) is the main lever. Each of the 30 layers' experts is ~0.45 GB in
this quant. With ~4.4 GB VRAM free after attention + KV at `CTX=16384`, about **8 layers'**
experts fit on the GPU ⇒ `NCMOE=22`.

| Setting | Experts on GPU | VRAM used | Gen (CUDA) |
|---|---|---|---|
| `--cpu-moe` | 0 | ~3.5 GB | slower |
| `NCMOE=22` | 8 | ~6.8 GB | ~23.5 t/s |

- `NCMOE=22` leaves ~1.2 GB headroom — about the limit on 8 GB once the desktop uses some VRAM.
  Going lower (more experts on GPU) risks OOM. If it OOMs, raise `NCMOE` or lower `CTX`.
- **`--no-mmap`:** with CPU tensor overrides, llama.cpp warns that mmap is slower; we load fully
  into RAM (the box has 31 GB).
- **`-fa auto`:** flash attention reduces KV-cache footprint.

### Context size

The **context window** is the `-c` argument to `llama-server`, exposed as the **`CTX`** env var on
all run scripts. It is the second VRAM consumer after the experts: the KV cache lives in VRAM
(~0.6 GB at 16K) and grows roughly linearly with `CTX`, so context and on-GPU experts (`NCMOE`)
compete for the same 8 GB.

| | |
|---|---|
| Default | **`CTX=32768`** (32K — the most that fits at the fast `NCMOE=22`) |
| Maximum | **262144** (256K — the value Gemma 4 was trained for; `gemma4.context_length`) |
| Practical max on 8 GB | ~160K, and only with all experts in RAM (`--cpu-moe`). 256K does not fit. |

#### Measured ceilings (RTX 2070, 8 GB, CUDA)

Context trades directly against on-GPU experts (= speed). Both share the 8 GB:

| Config | Experts on GPU | Context | VRAM used / free | Speed |
|---|---|---|---|---|
| `NCMOE=22` (default) | 8 layers | 16K | 6700 / 1274 MiB | ~23 tok/s |
| `NCMOE=22` (default) | 8 layers | **32K** | 6960 / 1013 MiB | ~23 tok/s |
| `--cpu-moe` | 0 layers | **128K** | 5716 / 2257 MiB | slower |
| `--cpu-moe` | 0 layers | 192K | — (won't fit) | — |

So: **~32K at full speed**, or up to **~128K** (≈160K hard ceiling) if you push every expert to RAM
with `--cpu-moe` — at the cost of speed (all 30 expert layers then run on the CPU, and prefill of a
large prompt is CPU-bound). Dial `NCMOE` between 22 and 30 to trade speed for context.

```bash
CTX=8192  BACKEND=cuda bash scripts/start.sh                 # smaller, frees VRAM
CTX=65536 NCMOE=27 BACKEND=cuda bash scripts/start.sh        # ~64K, some experts still on GPU
CTX=131072 BACKEND=cuda bash scripts/start.sh                # 128K, all experts in RAM (slower)
```

**Two knobs, one budget.** Raising `CTX` enlarges the KV cache (more VRAM); if it won't fit, raise
`NCMOE` (push experts back to RAM) or lower `CTX`. (The `-fit` auto-checker aborts cleanly at load
rather than crashing if `-ngl 99` + the requested context can't fit.) Flash attention (`-fa auto`)
and Gemma 4's sliding-window layers (every 6th uses `KV=2` heads) keep the cache small to begin with.

**Keep `pi` in sync.** `pi`'s own `contextWindow` (in `~/.pi/agent/models.json`) is independent of
the server's `-c`; if they disagree, `pi` plans around its own value. Pass the same `CTX` to
`configure-pi.sh` so they match:

```bash
CTX=32768 bash scripts/configure-pi.sh
```

### Auto-tuning the split: `scripts/benchmark-config.sh`

The fit boundary and the speed of each `(CTX, NCMOE)` pair are GPU-specific — free VRAM, driver, and
whatever else is on the card all move them. Rather than guess from the *Measured ceilings* table
above, `scripts/benchmark-config.sh` measures them on **your** hardware. For each context it launches
the *real* `llama-server` (through `run-server.sh`, on an isolated port — 8099 — so it never touches a
server you already have on 8080), waits for `/health`, runs one **discarded warm-up** generation (the
first CUDA decode pays a one-time graph-capture cost that would unfairly penalise GPU-heavy configs),
then times a short `/completion` and reads `predicted_per_second` straight from the server's own
timings. A config whose server dies while loading is reported as OOM. Pin one context with `CTX=`, or
sweep several with `CTX_LIST=`:

```bash
CTX=131072 bash scripts/benchmark-config.sh                    # optimise NCMOE for a 128K window
CTX_LIST=16384,32768,65536,131072 NCMOE_LIST=22,27,30 bash scripts/benchmark-config.sh
```

A representative sweep on this RTX 2070 (CUDA), reporting the **fastest NCMOE that fit** per context:

| CTX | Fastest fitting NCMOE | Gen (low fill) | Verdict |
|---|---|---|---|
| 16K  | `NCMOE=27` | ~23.6 t/s | snappy |
| 32K  | `NCMOE=27` | ~24.0 t/s | snappy |
| 65K  | `NCMOE=22` | ~28&nbsp;t/s | snappy |
| **128K** | `NCMOE=27` | **~23.7 t/s** | still snappy |

**What it shows: context is nearly free on this model.** Gemma 4's KV cache is tiny — flash attention
plus the sliding-window layers (every 6th uses `KV=2` heads) hold it to ~0.6 GB at 16K — so growing
the window 8× (16K→128K) only moved `NCMOE=27`'s VRAM from ~4.6 to ~7.0 GB and left generation flat at
~23–24 tok/s. What sets speed is the **backend** (CUDA vs Vulkan) and **`NCMOE`**, *not* the context
length. So the practical *"max context while still snappy"* here is the **full ~128K window at
`NCMOE=27` (~23 tok/s)**, with headroom to spare — better than the *Measured ceilings* table above
implies. That table assumed a large context forces every expert to RAM (`--cpu-moe`); in fact
`NCMOE=27` keeps the last 3 layers' experts on the GPU even at 128K and stays fast. (Even all-CPU
`NCMOE=30` held ~22 tok/s — on this box RAM bandwidth isn't the bottleneck, just as §8 notes.)

**Two caveats the numbers carry:**

- **Measured at low context fill.** The probe times generation against a near-empty KV cache, so
  "~23 tok/s at 128K" means *while the window is mostly empty*. Actually filling 128K is slower —
  attention runs over more tokens — and the tool deliberately doesn't pay the ~35-minute prefill that
  measuring it would cost (llama-bench has no `-c` flag; reaching a 128K depth via `-d` means
  prefilling 128K tokens at ~50 t/s). Use the numbers to **rank** configs; treat the absolute value as
  an optimistic ceiling.
- **Single-shot, so it picks up background load.** Each cell is one timed generation; if the desktop
  is using the GPU mid-run you'll see dips (a contended run here briefly read ~16 t/s where clean runs
  read ~23). And because `NCMOE=22` is a *marginal* fit (~7.6 GB of ~7.7 GB free), a transient VRAM
  blip can flip it from fit to OOM. Run it on an idle machine for clean numbers, and prefer
  `NCMOE=27` as the robust "snappy everywhere" pick.

**Driven from `start.sh` (measure once, reuse forever).** You don't have to run the benchmark by
hand. The first time `start.sh` brings up a fresh server with no saved result, it offers to run it,
then writes the outcome to a small gitignored cache (`.gemma4-tuning`, keyed `backend:ctx`, shared via
`scripts/_tuning.sh`). It has two modes:

- **No `CTX` set (the default launch):** it sweeps `CTX_LIST × NCMOE_LIST`, prints the fastest split
  that fits at each context, and **prompts you to pick which context to launch** (default = the
  largest that fit). Your choice is remembered as `backend:chosen` so later launches reuse it — no
  re-sweeping. This is the answer to "show me what's possible across context sizes."
- **`CTX=` pinned:** it tunes only `NCMOE` for that single context.

Every later launch reads the cache and applies the split (and chosen context) instantly — the slow
part runs *once*, not on every start. A declined offer is remembered (so it never nags), `AUTOTUNE=1`
forces a fresh sweep, `AUTOTUNE=0` disables it, and an explicit `NCMOE=` bypasses the whole thing.

---

## 10. Wiring up `pi`

`scripts/configure-pi.sh` idempotently adds a provider to `~/.pi/agent/models.json`:

```json
"llamacpp": {
  "api": "openai-completions",
  "apiKey": "dummy",
  "baseUrl": "http://127.0.0.1:8080/v1",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "gemma-4-26b-a4b-qat", "reasoning": true, "input": ["text","image"], ... }]
}
```

- `apiKey` is required by the schema but ignored by llama-server.
- `compat.supportsDeveloperRole: false` — some OpenAI-compatible servers don't understand the
  `developer` role used for reasoning models; this sends the system prompt as a `system` message.
- Run with `pi --provider llamacpp --model gemma-4-26b-a4b-qat`.

**Reasoning:** Gemma 4 thinks before answering (`thinking = 1` in the chat template). llama.cpp
returns the chain-of-thought in a separate `reasoning_content` field and the answer in `content`.
A too-small `max_tokens` lets the thinking consume the whole budget, leaving `content` empty —
give it room. `pi` handles this; it matters mainly for raw `curl` tests.

**Tool calls work.** Verified end-to-end: with tools enabled, `pi` had the model read a file
(extracting a value it could not otherwise know) and write a new file. The GGUF carries
`<|tool_response>` tokens and the chat template emits OpenAI-format tool calls.

### Sampling / model parameters

Sampling is set on the **server**, not in `pi` (which exposes no sampling flags). `run-server.sh`
passes `--temp`/`--top-p`/`--top-k` to `llama-server`, so they become the default for every request.
The defaults follow unsloth's [Gemma 4 recommendation](https://unsloth.ai/docs/models/gemma-4/qat):

| Env var | Flag | Default | llama.cpp's own default |
|---|---|---|---|
| `TEMP` | `--temp` | **1.0** | 0.8 |
| `TOP_P` | `--top-p` | **0.95** | 0.95 |
| `TOP_K` | `--top-k` | **64** | 40 |
| `EXTRA_ARGS` | (verbatim) | — | for `--min-p`, `--repeat-penalty`, `--seed`, … |

```bash
TEMP=0.7 bash scripts/start.sh                                    # more deterministic
EXTRA_ARGS="--min-p 0.01 --repeat-penalty 1.1 --seed 42" bash scripts/start.sh
```

These are server-wide defaults; a request that includes `temperature`/`top_p` overrides them for
that call. Confirm the live values at `GET /props` → `default_generation_settings.params` (e.g.
`temperature`, `top_k`, `top_p`).

---

## 11. Caveats & gotchas

A consolidated list of the non-obvious things that bit us — useful if you adapt this.

**Backend / CUDA**
- The conda `cuda129`/`cuda130` builds crash on drivers older than their CUDA (`device kernel
  image is invalid`). Use Vulkan or build from source against the driver's CUDA. (§6)
- `--list-devices` is **not** a validity test — the broken build passes it; only a real decode
  loads a kernel. Smoke-test with `llama-bench -n 8`, not device enumeration.
- A CPU-only *server* **does** work: `BACKEND=cpu bash scripts/run-server.sh` runs the conda
  binary with `--device none -ngl 0` and loads/serves cleanly (verified — model loads, no crash).
  The **source-built** `llamacpp-cuda` binary is the exception: its CUDA backend can still grab
  warmup compute at `-ngl 0` and re-trigger the kernel-image crash, so for CPU-only *benchmarking*
  use `llama-bench --device none -ngl 0` (its scheduler honors it).
- `llama-bench` accepts `--n-cpu-moe` but **not** `--cpu-moe` (it errors). The server accepts both.

**Process management**
- `pkill -f "llama-server"` matches *your own shell command* (which contains that string) and
  kills the subshell before it relaunches — surfaced as a confusing exit 144. Kill by **port**
  (`scripts/stop-server.sh` uses `ss`/`lsof`) or by PID instead.
- The server runs as bare `llama-server` (launched via `mamba run`), not `.../bin/llama-server`,
  so `pgrep -f "bin/llama-server"` silently matches nothing.
- Hiding the GPU via `CUDA_VISIBLE_DEVICES=""` to force a CPU-only *server* is unreliable (it does
  not affect the Vulkan device, and an empty value is interpreted inconsistently). The dependable
  CPU-only path is `llama-bench --device none -ngl 0` (above). (For the record, `mamba run` does
  **not** scrub the environment — env vars set before it do reach the child; verified with a
  sentinel var.)
- `mamba activate` needs `mamba init` in the shell first; in scripts, call binaries by full path
  or via `mamba run -n <env>`.

**Build script**
- A `... | tee log | grep -q PATTERN` under `set -o pipefail` **falsely fails**: `grep -q` exits on
  first match and closes the pipe, sending SIGPIPE to the producer, which `pipefail` then reports
  as failure even though the match succeeded. Write to a file first, then `grep` the file.
- `tee "$BUILD_DIR/../configure.log"` fails if `$BUILD_DIR` doesn't exist yet (the `..` can't
  resolve). Write logs to a directory that already exists.

**Model / inference**
- Small `max_tokens` ⇒ empty `content` (reasoning ate the budget). Not a bug.
- Gemma 4 prints harmless load warnings overriding control-token types
  (`<|tool_response>`, `</s>`); ignore them.
- `-c 248000` does not fit in 8 GB VRAM. Start at 16K.
- To inspect a GGUF (architecture, modality, tensor inventory) use `utils/inspect-gguf.sh <file>`.
  It uses the `gguf` Python module, auto-resolved from the vendored `llama.cpp/gguf-py` if it isn't
  pip-installed — so no extra setup when the llama.cpp source is checked out.

---

## 12. Reproducibility

Everything is scripted and parameterized; nothing touches the system outside conda envs.

```bash
# Vulkan (zero build):
bash scripts/setup.sh
bash scripts/configure-pi.sh
bash scripts/start.sh

# CUDA (~5× faster, +~20 min build):
BACKEND=cuda bash scripts/setup.sh        # also runs build-llama-cuda.sh
bash scripts/configure-pi.sh
BACKEND=cuda bash scripts/start.sh
```

| Script | Role |
|---|---|
| `setup.sh` | env + model download; `BACKEND=cuda` also builds CUDA |
| `build-llama-cuda.sh` | build llama.cpp vs the local CUDA (auto-detect, smoke-test) |
| `configure-pi.sh` | register the `llamacpp` provider in pi |
| `start.sh` / `run-server.sh` / `run-pi.sh` / `stop-server.sh` | run/stop server + pi |
| `make-speed-chart.py` | regenerate `docs/speed.svg` from measured numbers |

Overrides worth knowing: `BACKEND` (vulkan\|cuda), `NCMOE`, `CTX`, `PORT`, `MODEL`/`MODEL_FILE`/
`MODEL_REPO`, `CUDA_VER`/`CUDA_ARCH`/`LLAMA_REF`. See each script's header.

### Environments created

| Env | Purpose | Backend |
|---|---|---|
| `llamacpp` | conda-forge llama.cpp (`cuda129`) + `huggingface_hub` | Vulkan (CUDA kernels unusable on driver 535) |
| `llamacpp-cuda` | source-built llama.cpp + matching CUDA 12.2 toolchain | CUDA (native, fast) |

Both envs are captured as manifests: [`environment.yml`](../environment.yml) (runtime) and
[`environment-build.yml`](../environment-build.yml) (CUDA toolchain). The build manifest is pinned
to CUDA 12.2 with a note to adjust it — `cuda-toolkit` must match the driver's max CUDA. The
recommended build path remains `scripts/build-llama-cuda.sh`, which auto-detects the CUDA version
and GPU arch rather than relying on a static pin.

### References

- Model: <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF>
- **unsloth Gemma 4 QAT guide** (QAT explained, UD-quant accuracy numbers, recommended sampling):
  <https://unsloth.ai/docs/models/gemma-4/qat>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- pi: <https://github.com/badlogic/pi-mono>
- Ollama MoE-offload requests: [#11772](https://github.com/ollama/ollama/issues/11772),
  [#14579](https://github.com/ollama/ollama/issues/14579)

---

## 13. Running bigger models

With `--cpu-moe`, two things set what you can run: the **quantized model must fit in RAM**
(`--no-mmap` loads the CPU-resident weights there; it's a safe upper bound to use the *file size*),
and **generation speed tracks _active_ params, not total**. That distinction decides whether
"bigger" helps.

### The Gemma 4 GGUF lineup (unsloth)

| Model | Type | Active/token | Notes |
|---|---|---|---|
| E2B, E4B | small "efficient" | tiny | the ones bundled in ollama |
| 12B | **dense** | 12B | all params active |
| **26B-A4B** | **MoE** | **4B** | what this repo runs — the only big MoE |
| 31B | **dense** | 31B | all params active |

Each comes in QAT and non-QAT GGUF repos.

### Path A — higher-precision quant of the *same* 26B-A4B MoE  ✅ recommended

The QAT repo has only the 4-bit `UD-Q4_K_XL` (~14 GB). The
[non-QAT repo](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) has the full range. It's the
same MoE (4B active), so the **attention + KV "important stuff" still fits in 8 GB VRAM and runs on
the GPU** at every quant — but it is **not free**, and not the same speed (see the caveat).

| Quant | Size | Fits in 32 GB RAM? |
|---|---|---|
| Q4_K_XL (QAT, current) | 14 GB | yes |
| Q5_K_XL | 21 GB | yes — comfortable |
| Q6_K_XL | 23 GB | yes — good headroom |
| Q8_0 / Q8_K_XL | 27–28 GB | borderline; may fit because some experts live in VRAM, but test it |

```bash
MODEL_REPO=unsloth/gemma-4-26B-A4B-it-GGUF \
MODEL_FILE=gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf bash scripts/setup.sh
BACKEND=cuda NCMOE=24 bash scripts/start.sh    # raise NCMOE: bigger experts -> fewer fit on the GPU
```

> **A higher quant runs SLOWER, not the same speed.** A bigger quant scales up *everything*,
> including the GPU-resident tensors, so two things change:
> 1. **Fewer expert layers fit in VRAM.** At Q4 each layer's experts are ~0.41 GB and 8 fit
>    (`NCMOE=22`). At Q6 they're ~0.6 GB, so only ~4–6 fit (`NCMOE≈24–26`) — more experts fall back
>    to the slower CPU path. (The attention backbone + KV — a few GB — still fits regardless.)
> 2. **The RAM-resident experts stream ~1.5× more bytes/token** (Q6 vs Q4), and at ~23 tok/s the
>    RAM reads were already ~15 GB/s — pushing toward the DDR4 ceiling.
>
> Net: expect generation to fall from ~23 tok/s into roughly the **mid-teens for Q6**, lower for Q8.
> The clean ~23 tok/s figure is specifically a Q4 result. (Estimate — not yet measured here.)

> **Caveat — diminishing returns.** You're already on the **QAT** 4-bit, which is *trained* to be
> high-quality at 4-bit and is roughly competitive with non-QAT Q5/Q6. So you may be trading a
> noticeable speed drop and ~9 GB more RAM for a *small* quality gain. Worth trying, not a clear win.

### Path B — a bigger *model* (31B)  ⚠️ not worth it here

The only model bigger than 26B-A4B is the **31B, which is dense** — every token uses all 31B
params. `--cpu-moe` does nothing (there are no experts to place), so on an 8 GB GPU most of it runs
on the CPU. Expect **~1–3 tok/s** (recall CPU-only on the MoE — which computes just 4B active — was
~2 tok/s; a dense 31B computes ~8× more per token). It runs, but it's not interactive.

### Aside — MTP (self-speculative decoding) is *not* a win here

Gemma 4 ships a **Multi-Token Prediction** head, and llama.cpp added support for it
([ggml-org/llama.cpp#23398](https://github.com/ggml-org/llama.cpp/pull/23398), merged 2026-06-07). A
small "draft" head proposes the next few tokens and the full model verifies them in one batched pass:

```bash
llama-server -m <model>.gguf --model-draft <mtp-head>.gguf --spec-type draft-mtp --spec-draft-n-max 4
```

**Quality is safe** — speculative decoding is lossless by construction (the full model checks every
drafted token, so the output distribution is identical; the PR replicates Gemma's AIME-26 ~87%). But
**it doesn't help *this* setup**, for structural reasons:

- The headline **>2× speedup is the *dense* 31B**. On the **MoE 26B-A4B** the author saw *no* speedup;
  others report only ~10–30% — and only on big GPUs with the whole model resident in VRAM. MoE's
  bottleneck is streaming experts from RAM (`--cpu-moe`), and verifying *K* draft tokens activates the
  *union* of experts those tokens route to ⇒ **more** RAM traffic per step, working against the exact
  thing that limits us.
- On **8 GB it may not even load**: there's a reported model-load crash for "26B-A4B target + draft on
  a 16 GB card" when the target nearly fills VRAM before the draft loads. We're already at ~7 GB at
  `NCMOE=22`; making room for the draft head + its KV means pushing experts back to RAM (lower `NCMOE`)
  — trading away the speed that makes this rig fast, to chase a gain that nets ~zero on MoE.
- Practical blockers anyway: the stock build predates the merge (rebuild via `build-llama-cuda.sh`),
  and the QAT GGUF carries **no MTP tensors** — you'd need a separate draft head (QAT-matched heads
  exist at `huggingface.co/boxwrench/gemma-4-qat-mtp-assistant-heads`).

So MTP is a **dense-model / big-VRAM** optimization. If you ever run the dense 31B on a larger GPU it's
a real >2× win; for 26B-A4B on 8 GB, the CUDA backend + `NCMOE` tuning is where the tok/s lives.

### Bottom line

**26B-A4B is the sweet spot for this hardware** — it's the largest model that stays fast, precisely
because only 4B params are active per token. With 32 GB RAM the realistic upgrade is a **Q5/Q6 quant
of the same MoE** (marginally better quality, but slower — into the mid-teens tok/s), *not* a bigger
model — and *not* MTP (see the aside above). The clean ~23 tok/s belongs to the Q4 QAT file you're
already running.

---

## 14. Multimodal: images via the mmproj

Gemma 4 is a **natively multimodal** model (text + image + audio). That describes the *upstream*
weights, though — it does **not** mean the GGUF you run is multimodal, and there is **no "native
multimodal mode"** to switch on in llama.cpp.

**Why the text GGUF can't see.** Inspecting `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (arch `gemma4`,
658 tensors) shows only language tensors — `token_embd`, `blk.N.attn_*`, expert FFNs — and **zero**
vision/audio tensors or metadata. llama.cpp's converter **splits the vision/audio tower into a
separate `mmproj` GGUF** loaded by the `libmtmd` subsystem. "Encoder baked into the upstream
weights" ≠ "encoder in the GGUF": the split is a llama.cpp packaging convention, and it applies even
to models (like Gemma 3) whose encoder is conceptually part of the model. So multimodal here is
always **main GGUF + `--mmproj` projector**, never a flag on the text model alone.

**The projector.** `unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf` (~1.19 GB). It pairs fine with
the **QAT** weights even though it lives in the non-QAT repo — the projector is quant-agnostic. What
it contains:

| Key | Value |
|---|---|
| `general.architecture` | `clip` |
| `clip.vision.projector_type` | `gemma4v` |
| vision tensors (`v.blk.*`) | 352 |
| audio / conformer tensors | **0** |

So this file is **vision-only**. Gemma 4 *can* do audio — this llama.cpp's `mtmd` even ships the code
(`models/gemma4a.cpp`, `mtmd_audio_preprocessor_gemma4a`, the "gemma4 audio conformer") — but the
BF16 mmproj here has no audio conformer, so `--image` enables **images, not audio**. Audio would need
a different/unified projector (the `gemma4ua`/`gemma4uv` path).

**How `run-server.sh --image` wires it.** The flag adds `--mmproj "$MMPROJ" --no-mmproj-offload`:

- `--no-mmproj-offload` keeps the projector on the **CPU**. A 1.2 GB BF16 tower won't fit in the
  ~1 GB of VRAM left after `NCMOE=22` experts + KV cache; offloading it would OOM. Image *encoding*
  is therefore CPU-bound (a one-off cost per image), but token **decode stays ~full speed** because
  the language model's GPU split is unchanged.
- A benign `-fit` warning appears at load — `failed to fit params to free device memory:
  n_gpu_layers already set by user to 99, abort`. That's just the auto-fitter declining to override
  the pinned `-ngl 99`; the server loads and serves normally.

**Verified.** Sent a synthetic scene (red circle, blue square, green triangle, the text "42") via the
OpenAI `image_url` format; the model returned all three shapes with correct colors and read "42".
Decode held at ~30 tok/s with the image in context. (As always, give it enough `max_tokens` to finish
its hidden reasoning before the visible answer — see §8.)

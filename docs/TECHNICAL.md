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

### Exact build currently in use (pin this to reproduce)

The `vendor/llama.cpp/build/bin/llama-server` we run **right now** (built 2026-06-12) is:

| | |
|---|---|
| **Repo** | `https://github.com/ggml-org/llama.cpp` (upstream `ggml-org`, no fork) |
| **Commit** | `88a39274ecf88ba11686acd357b59685b1cbf03d` (`git describe` → `b9549-57-g88a3927`) |
| **Upstream PR** | #18039 — *"spec: add EAGLE3 speculative decoding support"* |
| **Source state** | **pristine** — clean working tree, **no local patches** (the build script applies none) |
| **Toolchain** | mamba env `llamacpp-cuda`: `cuda-toolkit 12.2` (nvcc V12.2.140), `gcc/gxx_linux-64 12.4.0` |
| **CMake** | `Release`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_NATIVE=ON`, `LLAMA_CURL=OFF`, `CMAKE_CUDA_ARCHITECTURES=75` (sm_75 / Turing), Ninja |
| **Targets** | `llama-server llama-cli llama-bench` |

**Reproduce the exact binary** (the commit is *not* a release tag, so you must pin it — the script
otherwise builds the latest tag, see below):

```bash
LLAMA_REF=88a39274ecf88ba11686acd357b59685b1cbf03d \
CUDA_VER=12.2 CUDA_ARCH=75 \
./scripts/build-llama-cuda.sh
```

Caveats for an *exact* match:
- **`GGML_NATIVE=ON`** compiles the CPU paths with `-march=native`, i.e. tuned to the **build
  host's** microarchitecture. A different CPU yields a functionally-equivalent but not
  byte-identical binary.
- This commit **is** the EAGLE3 PR. EAGLE3 is **dormant** unless `llama-server` is launched with an
  eagle3 draft model, so this binary serves normal Gemma 4 (MTP/none) correctly — we sit on it only
  because that's where the EAGLE3 evaluation left the checkout (see §13's "EAGLE3 — tried, doesn't
  work on this build" note). It was *not* chosen for any improvement over the latest tag.

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
  -V | tail -1`, which handles `bNNNN` numerically) rather than `master`. **Note:** this default is
  *not* what the current build uses — our live binary is pinned to commit `88a3927` (above), which is
  57 commits past tag `b9549` and would be *replaced* by a newer tag if you ran the script without
  setting `LLAMA_REF`. Pin `LLAMA_REF` to reproduce the exact build.

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
then times `RUNS` short `/completion`s (default 5) and reports the **median** of `predicted_per_second`
read straight from the server's own timings. A config whose server dies while loading is reported as
OOM. Each config prints live progress (`loading… runs: 1/5 2/5 …`) to stderr as it goes. Pin one
context with `CTX=`, or sweep several with `CTX_LIST=`:

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
- **Median of `RUNS` runs, to smooth the GPU clock.** The 2070's boost clock bounces between probes —
  each config reloads the 14 GB model first, so the GPU idles and then ramps back unevenly, and a lone
  timed run could read anywhere from ~20 to ~30 tok/s for the *same* `NCMOE` purely on clock state.
  Left unchecked that made the per-context "fastest NCMOE" pick depend on which probe happened to clock
  high. So each config is timed `RUNS` times (default 5; `RUNS=` to change) and reported as the
  **median**, with the min–max spread shown alongside (`median of 5: 19.6–30.1`) so you can see how
  noisy that config was. A telltale that you're looking at clock noise rather than a real difference:
  prompt-processing and generation rise and fall *together* across probes (a hot probe reads high on
  both), and the same `NCMOE` varies wildly across contexts even though context size barely affects
  decode speed at low fill. Still: run it on an idle machine, and because `NCMOE=22` is a *marginal*
  fit (~7.6 GB of ~7.7 GB free) a transient VRAM blip can flip it from fit to OOM, so prefer `NCMOE=27`
  as the robust "snappy everywhere" pick.

**Driven from `start.sh` (measure once, reuse forever).** You don't have to run the benchmark by
hand. The first time `start.sh` brings up a fresh server with no saved result, it offers to run it,
then writes the outcome to a small gitignored cache (`.gemma4-tuning`, keyed `backend:ctx`, shared via
`scripts/_tuning.sh`). It has two modes:

- **No `CTX` set (the default launch):** it sweeps `CTX_LIST × NCMOE_LIST`, prints the fastest split
  that fits at each context, and **prompts you to pick which context to launch** (default = the
  largest that fit). Your choice is remembered as `backend:chosen:kvquant` so later launches reuse it — no
  re-sweeping. This is the answer to "show me what's possible across context sizes."
- **`CTX=` pinned:** it tunes only `NCMOE` for that single context.

Every later launch reads the cache and applies the split (and chosen context) instantly — the slow
part runs *once*, not on every start. A declined offer is remembered (so it never nags), `AUTOTUNE=1`
forces a fresh sweep, `AUTOTUNE=0` disables it, and an explicit `NCMOE=` bypasses the whole thing.

### Guided setup: `start.sh --menu`

Everything above — plus the KV-cache and sampling knobs below — is reachable from one interactive
front-end: `bash scripts/start.sh --menu`. It is deliberately **not** a new configuration system.
Each prompt just sets the same environment variable the equivalent command line would, then the
normal launch path runs unchanged. Each step shows its valid values and a one-line explanation,
**every** prompt accepts `q` (or Ctrl-D) to cancel cleanly, and picking auto-tune prints an
up-front heads-up that the first measurement takes a few minutes (it's cached afterwards):

| Menu step | Sets | Notes |
|---|---|---|
| Backend | `BACKEND` (unset = auto-detect) | cuda / vulkan / cpu |
| Strategy | — | **auto-tune** (reuse-or-measure, the sweep above) vs **manual** |
| Context | `CTX` (+ the explicit flag) | manual, or "auto-tune one specific context" |
| Expert split | `NCMOE` | manual mode only; blank = all experts on CPU |
| KV cache | `KVQUANT` | f16 / q8_0 / q4_0 / other |
| Sampling | `TEMP` / `TOP_P` / `TOP_K` | defaults follow unsloth |
| Image | `--image` | loads the vision projector |

It runs only for a **fresh** server (a reused one can't be reconfigured) and requires a real TTY (it
refuses on a pipe). The `--menu` token is consumed by `start.sh` and never forwarded to `pi`. Because
every step maps to a plain env var, the menu is purely ergonomics — each choice has a non-interactive
equivalent, e.g. `BACKEND=cuda CTX=65536 NCMOE=20 KVQUANT=q8_0 bash scripts/start.sh`.

One thing the menu does that the bare env-var path does **not**: once the final context is resolved
(after the sweep-and-pick, if any), it runs `configure-pi.sh` automatically so pi's `contextWindow`
matches the server's `-c`. Without that, the server can serve 128K while pi silently caps the
context at its own configured window — a 128K server with a 32K client. The sync is gated on
`--menu` (an explicit interactive opt-in, since it edits `~/.pi/agent/models.json`); env-var
launches only get a printed reminder to run `configure-pi.sh` themselves.

### KV-cache quantization: `KVQUANT`

`CTX` and `NCMOE` both spend the same 8 GB; **`KVQUANT`** adds a third lever by shrinking the KV
cache *itself* in VRAM. Set it to a llama.cpp cache type — `q8_0` (near-lossless, recommended),
`q5_1`, `q5_0`, `q4_1`, `q4_0`/`iq4_nl` (aggressive), or `f16` (default/off). `run-server.sh` maps it
to `-ctk`/`-ctv`.

Implementation notes (and the reasons behind them):

- **It forces flash attention on.** llama.cpp refuses to quantize the V cache without it
  (`llama-context.cpp`: *"V cache quantization requires flash_attn"*). So any quant type overrides
  `-fa` to `on` — the script makes `-fa` a variable for exactly this. The valid type list is read
  from `common/arg.cpp` and validated up-front (a typo errors before the 14 GB load, not after).
- **It's a long-context lever, not a default.** At 32K the KV cache is ~0.6 GB — quantizing it is
  noise. The payoff grows with context. Measured here (CUDA, RTX 2070, 8 GB), at the same `NCMOE`
  `q8_0` freed ~976 MiB at 65536 and ~2 GB at 131072 — enough to fit a *faster* expert split, so
  the gain shows up as both higher tok/s and a larger usable context:

  | Context | f16 (off) | `q8_0` | Gain |
  |---|---|---|---|
  | 65536  | `NCMOE=22` → 26.1 tok/s | `NCMOE=20` → 30.5 tok/s | +17% |
  | 131072 | `NCMOE=27` (only fit) → 19.5 tok/s | `NCMOE=22` → 27.0 tok/s | +38% |

  At 128K, f16's KV cache is large enough that *only* the slowest all-but-three-layers-on-CPU split
  fits; `q8_0` shrinks it so the fast split fits. Same 4-bit model weights in both columns — only the
  KV-cache precision changes. (tok/s measured at low context fill; use to rank, not as a deep-context
  promise.) Below ~64K it rarely earns the small quality cost.
- **It is a tuning dimension.** Because quantizing the KV cache frees VRAM, the *fastest fitting
  `NCMOE` changes* — so the auto-tune cache is keyed `backend:ctx:kvquant`, and the benchmark probes
  with the same `KVQUANT`. A `q8_0` tune and an `f16` tune never collide; existing pre-`KVQUANT`
  caches are migrated to the `:f16` slot rather than discarded.
- **Backend caveat:** verified working on CUDA *and* Vulkan, but flash-attn + KV-quant on the older
  Vulkan path can be slow — `run-server.sh` prints a warning there. CUDA is the recommended path.

```bash
CTX=131072 KVQUANT=q8_0 bash scripts/start.sh        # 128K context, quantized KV
CTX=131072 KVQUANT=q8_0 bash scripts/configure-pi.sh # keep pi's window in sync
```

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

**The llama.cpp source itself is *not* auto-pinned.** To reproduce the exact `llama-server` binary
in use today, pin the commit (the script otherwise builds whatever the latest release tag is):

```bash
LLAMA_REF=88a39274ecf88ba11686acd357b59685b1cbf03d \
CUDA_VER=12.2 CUDA_ARCH=75 bash scripts/build-llama-cuda.sh
```

Full build provenance (commit, toolchain, CMake flags, caveats) is in **§7, "Exact build currently
in use".**

### References

- Model: <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF>
- **unsloth Gemma 4 QAT guide** (QAT explained, UD-quant accuracy numbers, recommended sampling):
  <https://unsloth.ai/docs/models/gemma-4/qat>
- llama.cpp: <https://github.com/ggml-org/llama.cpp> — **build in use: commit `88a3927`**
  (`88a39274ecf88ba11686acd357b59685b1cbf03d`, `git describe` `b9549-57-g88a3927`, PR #18039)
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

### Aside — MTP (self-speculative decoding): measured — big at greedy, marginal at temp 1.0

Gemma 4 ships a **Multi-Token Prediction** head, and llama.cpp supports it — core MTP
([#22673](https://github.com/ggml-org/llama.cpp/pull/22673), merged 2026-05-16) plus the Gemma 4 wiring
([#23398](https://github.com/ggml-org/llama.cpp/pull/23398), merged 2026-06-07). A small "draft" head
proposes the next few tokens and the full model verifies them in one batched pass:

```bash
llama-server -m <model>.gguf --model-draft <mtp-head>.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

**Quality is safe** — speculative decoding is lossless by construction (the full model checks every
drafted token, so the output distribution is identical). Three objections that earlier ruled it out
are now retired — two by inspection, the third **by measurement** (2026-06-12; full numbers in
[`mtp-benchmark.md`](mtp-benchmark.md)):

- **The QAT head exists — inside our own repo.** Unsloth ships a QAT-matched, smart-4bit MTP head in
  `unsloth/gemma-4-26B-A4B-it-qat-GGUF` itself: `mtp-gemma-4-26B-A4B-it.gguf` (**0.25 GB**), plus an
  `MTP/` folder with Q8_0 (0.46 GB) and F16/BF16 (0.86 GB) variants. No separate or third-party repo.
  With `-hf` the bundled head auto-loads; with our local-file launch, add
  `--model-draft .../models/gemma4-26b-a4b-qat/mtp-gemma-4-26B-A4B-it.gguf`.
- **VRAM cost is small.** The head is **0.25 GB**, not multiple GB — roughly one `NCMOE` step, plus a
  little KV. Unsloth budgets "~2 GB extra RAM/VRAM headroom" and lists 26B-A4B 4-bit at 17–18 GB
  *total* (RAM+VRAM) *with* MTP; we have 8 + 32 = 40 GB. (The reported load-crash on a 16 GB card was a
  specific bug, not a size law.)
- **Speedup is real but acceptance-limited — measured here** (full study + noise analysis in
  [`mtp-benchmark.md`](mtp-benchmark.md)). The CUDA build already supports it (`vendor/llama.cpp` @
  `04eb4c4`, #23398 — no rebuild) and MTP loads at the **same `NCMOE=27`** as baseline. At **greedy**
  the draft head hits 79–88 % acceptance and decode runs **+19–31 %** faster (≈21→27 tok/s) — a real win
  that *refutes* the old worry below. At the rig's **default sampling** (temp 1.0) acceptance falls to
  66–74 % and the gain is **within measurement noise** — this Max-Q rig's single-config baseline scatters
  ±13 %, so "+2–7 %" is *no measurable gain*, not a small one. Use `--spec-draft-n-max 2` with
  **`--spec-draft-p-min 0`**: a positive `p_min` (a draft-confidence floor, not a relaxed accept rule)
  **degenerates the output** under sampling — n-max 6/p_min 0.7 and n-max 4/p_min 0.75 both collapsed to
  `fmt-fmt…` loops at temp 1.0 (fake 85–100 % acceptance, garbage text). So: a clear win at greedy/low
  temp, nothing measurable at temp 1.0, and `p_min>0` is unsafe here.

  > The earlier reasoning here was *wrong*: it claimed verifying *K* tokens activates the *union* of
  > their experts ⇒ **more** RAM traffic ⇒ no win on `--cpu-moe`. The greedy +25 % shows the opposite —
  > batched verification **amortizes** expert-weight streaming across accepted tokens (consecutive
  > tokens route to overlapping experts). The limiter is draft **acceptance**, not RAM bandwidth.

**Net:** not the dense-only / big-VRAM dismissal it was, and not a free lunch either. It's lossless and
costs only a 0.25 GB head that fits at the current NCMOE, so enabling it never hurts output. At the
default temp 1.0 it buys nothing measurable; its value is at **greedy / low-temperature** (coding) work,
where +20–30 % is real. If enabled, `--spec-type draft-mtp --spec-draft-n-max 2`.

**Context is cheap — run 64k.** Gemma 4's sliding-window attention (`n_swa=1024`) caps most layers' KV,
so **64k decodes at essentially the same tok/s as 32k** (~22–25 either way; both fit at `NCMOE≈25–27`).
Decode speed is set by `NCMOE` (expert placement), not context length. So the 64k target costs nothing.

**EAGLE3 — tried, doesn't work on this build (2026-06-12).** llama.cpp `88a3927` added EAGLE3 spec
decoding with a Gemma 4 draft (`RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`). We converted it,
rebuilt at `88a3927`, and benchmarked it — it's **not usable**: draft acceptance is only ~42 % (vs MTP's
79–88 %; not a quant artifact), so its *heavy* draft (a full transformer layer) makes it **slower than
baseline even at greedy** (−19 % at n-max 3); temp 1.0 **degenerates**; and on the chat-template path
(i.e. real use through pi) it **segfaults**. The fresh eagle3 support in this build is too immature —
revisit only on a much newer llama.cpp. (Our `vendor/llama.cpp` checkout **still sits at `88a3927`**
— that's just where this evaluation left it; EAGLE3 stays dormant unless `llama-server` is launched
with the draft, so the live binary serves normal Gemma 4 fine. See §7, "Exact build currently in
use", for the pinned-commit reproduction.) **Bottom line for this rig today: there is no temp-1.0
spec-decoding win — MTP helps only at low temperature, so for faster coding, lower the temperature.**

### Bottom line

**26B-A4B is the sweet spot for this hardware** — it's the largest model that stays fast, precisely
because only 4B params are active per token. With 32 GB RAM the realistic upgrade is a **Q5/Q6 quant
of the same MoE** (marginally better quality, but slower — into the mid-teens tok/s), *not* a bigger
model. **MTP** is a lossless free-tok/s lever (0.25 GB head already in our repo, fits at the current
NCMOE) — measured at +18–31 % greedy but only +4–7 % at temp 1.0 (acceptance-limited; see the aside
and [`mtp-benchmark.md`](mtp-benchmark.md)). The clean ~23 tok/s belongs to the Q4 QAT file you're
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

## 15. The harness layer: `pi-extensions/`

Everything above gets the model *running*; this section is about making it *useful*. The
problem statement, what we built, why those shapes and not others, and what's still open.
(Per-extension specs and usage live in each `pi-extensions/<name>/README.md`, and the
cross-cutting rules in `pi-extensions/README.md`. This section is the engineering rationale.)

### Problem statement

A 4-bit 26B-A4B MoE on a 120k window is a competent but fallible coding agent, and its
failure modes are *systematic*, not random:

| Weakness (observed) | Consequence in real sessions |
|---|---|
| Weak self-verification | Ships syntactically broken edits, believes they're fine |
| Perseveration | Repeats an identical failing tool call until the context fills with errors |
| No working memory discipline | Re-reads 800-line files to find one signature; re-decides the task mid-way |
| No autonomous termination | Unattended runs stop before the objective is met, or never stop at all |
| No cross-session memory | Every session restarts from zero |
| Thin world knowledge | A 26B can't carry the long tail; needs search + read |
| Hand-waving from memory | Trusts recollection — states what it never derived, ran, or read, as if proven |
| Prompt-rule blindness | Instructions in the system prompt decay; the model "knows" but doesn't *do* |
| Capability ceiling | Some plans are just wrong, and no amount of self-review by the same model fixes that |

Cloud harnesses paper over all of these with a bigger model. The constraint here is the
opposite: **the local model is the only intelligence at runtime**, so every weakness must be
covered by *deterministic code* around it. The second constraint is energy: prefill dominates
laptop inference cost (§8), so every token the harness injects is a standing tax paid on
every request.

### Solution shape, and why

One pi extension per weakness, all obeying six cross-cutting rules (R1–R6, documented in
`pi-extensions/README.md`). The two that drive most design decisions:

- **Enforce > persuade (R4).** A prompt rule ("verify your edits") relies on exactly the
  attention that a small model lacks — so the harness *does the thing* instead: verified-edits
  runs the checker itself, symbols intercepts the oversized read, loop-breaker counts the
  failures. Deterministic code does not get distracted.
- **KV-cache discipline (R1).** llama.cpp reuses KV cache only for an unchanged prompt
  *prefix*. So everything static (system prompt, tool schemas, MEMORY.md) is byte-stable for
  the whole session, and everything dynamic (plan state, recalled memories, nudges) is
  injected at the *tail*. This is the difference between paying prefill once and paying it
  every turn.

The full mapping:

| Weakness | Extension | Mechanism (one line) |
|---|---|---|
| Weak self-verification | `verified-edits` | Auto-runs the cheapest checker after every edit; errors appended in-band |
| Perseveration | `loop-breaker` | 3 identical failing calls → one tail nudge to change approach |
| File re-reading | `symbols` | Outline tools + big-read interception |
| Task drift | `plan` | External checklist (the steps) re-injected at tail; survives compaction; defers the finish to `goal_done` |
| No cross-session memory | `semantic-memory` | Passive recall: embed the user turn, inject top matches at tail |
| Rule blindness | `operating-manual` | If-then triggers in the stable prefix + JIT nudges |
| Thin world knowledge | `web-search` + `fetch-page` | Stealth Playwright search → readable-text reads |
| Unmeasured cost | `stats` | llama.cpp timings → per-session token/energy accounting |
| Over-thinking | `thinking-router` | Per-turn thinking budget routed by input difficulty |
| Hand-waving from memory | `grounding` | Engineering mindset in the prefix + a prove-it check at the tail: derive / simulate / reference, never trust recollection |
| No autonomous termination | `goal` | Machine-checkable north-star drives the loop until `done_when` passes; the nudge **anneals** across cycles (explore → consolidate → commit → decide) so the budget ends on a forced decision, not a hard cut; bounded cycles; verifies plan's steps, no checklist of its own |
| Capability ceiling | `advisor` | Escalate to a stronger external agent (below) |

`plan` and `goal` split cleanly so they don't duplicate: `plan` owns the *steps* (the
checklist), `goal` owns the *objective* + `done_when` (the finish), and `goal_done` reads
`plan`'s persisted state to confirm the steps are complete before accepting — one checklist,
one done-decision.

**Annealed termination — land the answer, don't yank it.** A flat loop coaches cycle 1 and cycle
19 identically and then cuts the work off at the budget. `goal` instead *anneals* the nudge over
its own `cycle / max_cycles` counter — generous and exploratory early, increasingly directive late —
through four phases (**explore → consolidate → commit → decide**) chosen by *reserved cycle counts*
(not raw temperature), so the arc stays sane at any budget. The honesty floor never melts: every
phase keeps "verified, or explicitly marked *unverified*"; only the emphasis and effort-triage cool.
The terminal *ramps* rather than guillotines — the final cycle is an explicit "you cannot iterate
further, decide now," and the model's own exit is `goal_conclude`, which lands the work as a new
**`concluded`** status (`partial` or `abandoned`, with a one-line reason) — distinct from `blocked`
(ran out of road) and `done` (verified). So an unattended run ends on a *stated decision*, not a
silent cut. (An optional second channel cools the *sampling temperature* on the same cosine schedule,
`PI_GOAL_TEMP_ANNEAL=1`; off by default. Full rationale:
[`goal-annealing-prd.md`](goal-annealing-prd.md) and the
[extension README](../pi-extensions/goal/README.md).)

Two extensions round out the harness without covering a model *weakness*, so they sit outside
the table above: **`pipe`** chains slash-commands into one ordered agent directive
(`/pipe /goal … /plan …`), and **`toolsets`** is context economy — it gates situational tool
groups so the per-request tool tax shrinks. The tool definitions are themselves a standing
prefill cost, so two levers attack it: the R5 wording pass (terse, model-optimal descriptions)
and `toolsets` (announce fewer tools). `toolsets` sets the active set *once per session* on
purpose — tool schemas live in the KV-cached prefix, so toggling them mid-session re-prefills
(the same R1 logic that governs every injection here).

### The escalation path: `advisor`

The last row is qualitatively different and deserves its own rationale. Eleven of the twelve
extensions assume the model's plan is *recoverable* — verify it, nudge it, remind it. But a
wrong plan executed carefully is still wrong, and a model cannot reliably review its own
reasoning. Cloud harnesses solve this with a stronger reviewer model. The local equivalent:
an `advisor` tool that serializes the whole session branch into a transcript and asks an
**external agent of the user's choosing** for a verdict.

**Solution chosen: drive an interactive TUI through tmux** (the existing `tui-driver`
project: start a session, paste a prompt, wait for the screen to stabilize, scrape the
reply). The transcript goes to a 0600 file in a per-process 0700 mkdtemp dir; the prompt
hands the advisor the file path; the reply comes back as the tool result (capped, full text
saved).

**Why this over the alternatives considered:**

- *Direct API call to a cloud model* — needs per-provider key management, billing wiring,
  and request-format code inside the extension. The TUI route reuses agents the user has
  **already installed, authenticated, and paid for** (agy, claude, …), at zero integration
  cost per new agent. Configurability falls out for free: the agent is one string in a
  config file.
- *A second pi/Gemma instance as reviewer* — no capability lift; self-review by the same
  weights is exactly the failure mode this exists to escape. (Still possible via config if
  someone wants a fresh-context second opinion.)
- *MCP or RPC integration per agent* — strictly more machinery for fewer supported agents;
  tui-driver already handles approval prompts, throttling, orphan reaping, and works with
  *any* TUI unmodified.
- *No default agent, on purpose* — consulting an external agent can cost money, so an
  unconfigured tool returns a teaching error (R2) with the exact config to write, instead
  of silently picking a vendor.

**Caveats** (also in the extension README): screen-scraping is inherently fragile — TUI
chrome can leak into replies, and reply extraction anchors on the echoed prompt; the call
is synchronous, so Gemma blocks for up to `timeoutSec` while the advisor thinks; the
advised TUI must be able to read the transcript file without an interactive approval
prompt (or use `inlineTranscript` to paste the text); a kept-alive session holds whatever
resources the advised agent holds until tui-driver's idle watchdog reaps it.

### Future work

- **Async advisor.** tui-driver already has `send-async`/`poll`; the extension could return
  immediately and inject the advisor's verdict at tail (`deliverAs: "steer"`) when it lands,
  letting Gemma keep working instead of blocking.
- **Auto-escalation.** loop-breaker and advisor are natural partners: after the nudge has
  fired twice with no change in behavior, suggest (not force — cost) an `advisor` call.
- **Structured verdicts (R6).** Today the advisor replies free-form; a fill-in template
  (Sound?/Missed:/Next:) would make verdicts parseable and injectable as plan steps.
- **Reply cleaning.** A per-TUI post-filter (strip spinners, box-drawing, status lines)
  would harden the scraped replies.
- **Engine levers** (tracked in `pi-extensions/README.md`): speculative decoding via `--model-draft` is now
  **done and measured** — MTP is wired into `start.sh` (`MTP=1`) and benchmarked (lossless;
  +15–30 % at greedy/coding temp, within measurement noise at temp 1.0; see §13 and
  [`mtp-benchmark.md`](mtp-benchmark.md)). Still open: GBNF/JSON-schema-constrained tool calls
  if the custom build exposes them.

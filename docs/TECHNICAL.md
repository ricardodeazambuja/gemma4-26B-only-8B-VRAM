# Technical write-up: running Gemma 4 26B-A4B (QAT) on an 8 GB GPU with `pi`

This document explains **how** the setup in this repo works, **why** each decision was made,
and the **caveats** discovered along the way. It is the engineering companion to the
[README](../README.md) (quickstart) and [MEMORY.md](../MEMORY.md) (terse notes).

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

Two terms worth defining:

- **QAT (Quantization-Aware Training):** the model was fine-tuned with quantization simulated in
  the forward pass, so the 4-bit weights keep much more quality than naive post-training
  quantization. This is what makes a 4-bit 26B model genuinely usable.
- **UD-Q4_K_XL:** unsloth's "Unsloth Dynamic" quant — different tensors get different bit-widths
  (important ones kept higher) rather than a uniform Q4.

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

`pi` is often described as "ollama-based", but in practice it just speaks the **OpenAI
chat-completions API**. Its provider config (`~/.pi/agent/models.json`) points at any
OpenAI-compatible endpoint. We register a `llamacpp` provider with `baseUrl
http://127.0.0.1:8080/v1` and `api: openai-completions`. From `pi`'s perspective, llama.cpp's
`llama-server` is indistinguishable from ollama — it just needs the endpoint.

This indirection is the key architectural insight: **the inference engine is decoupled from the
agent.** We can run Vulkan or a custom CUDA build behind the same endpoint and `pi` never changes.

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

Crucially, `--cpu-moe` is **backend-agnostic** — it is a tensor-placement directive, so the
RAM/VRAM split is identical whether the GPU backend is CUDA or Vulkan. That fact is what let us
swap backends freely (§6).

**Measured VRAM** at `CTX=16384`: `--cpu-moe` (0 expert layers on GPU) ≈ 3.5 GB; `NCMOE=22`
(8 expert layers on GPU) ≈ 6.8 GB. On an 8 GB card, `NCMOE=22` is about the ceiling.

---

## 5. Why not Ollama

`pi` ships configured for ollama, and the box already had ollama 0.21.2. But **ollama cannot do
`--cpu-moe`.** It performs only automatic *whole-layer* GPU offload — it cannot keep a layer's
attention on the GPU while putting that same layer's experts on the CPU. There is no environment
variable, Modelfile parameter, or CLI flag for per-expert placement. (Tracked upstream:
[ollama#11772](https://github.com/ollama/ollama/issues/11772),
[ollama#14579](https://github.com/ollama/ollama/issues/14579).)

So we run **llama.cpp's `llama-server`** instead and point `pi` at it. Because `pi` only needs the
OpenAI endpoint (§3), this is a drop-in swap — same agent, different engine.

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
- **Context:** the OP recipe used `-c 248000`. That is fantasy on 8 GB — KV cache for 248K tokens
  does not fit. We use `-c 16384`; raise it only if VRAM allows (it competes with experts).
- **`--no-mmap`:** with CPU tensor overrides, llama.cpp warns that mmap is slower; we load fully
  into RAM (the box has 31 GB).
- **`-fa auto`:** flash attention reduces KV-cache footprint.

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

---

## 11. Caveats & gotchas

A consolidated list of the non-obvious things that bit us — useful if you adapt this.

**Backend / CUDA**
- The conda `cuda129`/`cuda130` builds crash on drivers older than their CUDA (`device kernel
  image is invalid`). Use Vulkan or build from source against the driver's CUDA. (§6)
- `--list-devices` is **not** a validity test — the broken build passes it; only a real decode
  loads a kernel. Smoke-test with `llama-bench -n 8`, not device enumeration.
- In `llama-server`, `-ngl 0` / `--device none` does **not** force CPU-only: the CUDA backend
  still registers and grabs warmup compute, re-triggering the crash. To benchmark true CPU-only,
  use `llama-bench --device none -ngl 0` (its scheduler honors it). A pure-CPU *server* would need
  the conda `cpu_mkl` build instead.
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
- There was no `gguf` Python module in the env; GGUF metadata was read by parsing the binary
  header directly (little-endian: magic, u32 version, u64 tensor_count, u64 kv_count, typed KVs).

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

### References

- Model: <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- pi: <https://github.com/badlogic/pi-mono>
- Ollama MoE-offload requests: [#11772](https://github.com/ollama/ollama/issues/11772),
  [#14579](https://github.com/ollama/ollama/issues/14579)

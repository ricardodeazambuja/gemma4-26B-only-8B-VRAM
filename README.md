# Gemma 4 26B-A4B (QAT) locally on an 8 GB GPU, driven by `pi`

![model](https://img.shields.io/badge/Gemma%204-26B--A4B%20QAT-4c8bf5)
![GPU](https://img.shields.io/badge/GPU-RTX%202070%208%E2%80%AFGB-76b900)
![CUDA](https://img.shields.io/badge/CUDA%20build-~23.5%20tok%2Fs-2ea043)
![Vulkan](https://img.shields.io/badge/Vulkan-~4.9%20tok%2Fs-f0883e)
![CPU](https://img.shields.io/badge/CPU%20only-~1.9%20tok%2Fs-8b949e)

Run Google's **Gemma 4 26B-A4B** mixture-of-experts model (26B total / **4B active**), in the
unsloth **QAT** 4-bit GGUF (~14 GB), on a laptop with an **8 GB** NVIDIA GPU — served by llama.cpp
and driven by [`pi`](https://github.com/badlogic/pi-mono), a local coding agent.

The trick is llama.cpp's **`--cpu-moe`** flag: it pins the heavy MoE expert weights to system RAM
while keeping the attention layers, the **KV cache (your context)**, and — VRAM permitting — a few
expert layers on the GPU. A 14 GB model that can't possibly fit in 8 GB of VRAM then runs
comfortably, because only ~4B params are active per token.

<p align="center"><img src="docs/speed.svg" alt="Token generation speed by backend: CPU 1.9, Vulkan 4.9, CUDA 23.5 tok/s" width="680"></p>

> ⚡ **Want the most tokens/sec? Build the CUDA backend** — it's **~5× faster** than the zero-build
> Vulkan default (~23.5 vs ~4.9 tok/s here). `scripts/build-llama-cuda.sh` builds it against your
> *own* driver's CUDA; the run scripts then **auto-select** it. The launch banner shows which is
> live: 🟢 CUDA / 🟡 Vulkan / 🔴 CPU.

<p align="center"><img src="docs/architecture.svg" alt="pi talks over HTTP to llama-server (OpenAI API on :8080); the GPU's 8 GB VRAM holds attention layers, the KV cache (your context), and N expert layers, while system RAM holds the remaining MoE expert FFN weights via --cpu-moe" width="700"></p>

And it's more than a model server. A suite of **[Gemma-tuned `pi` extensions](#the-pi-extensions)**
turns a fallible small model into a usable, fully-offline coding agent — covering its weaknesses
(broken edits, retry loops, lost context, no cross-session memory) with deterministic code.

> 📖 The full engineering story — architecture, the CUDA-version wall, the build, the performance
> analysis, multimodal, and every caveat — is in **[docs/TECHNICAL.md](docs/TECHNICAL.md)**. Every
> script and env-var knob is in **[scripts/README.md](scripts/README.md)**.

---

## Prerequisites

- **mamba / conda** ([Miniforge](https://github.com/conda-forge/miniforge)) — the scripts build all
  other dependencies into isolated envs.
- An **NVIDIA GPU** with a working driver (it provides the Vulkan ICD), **~14 GB free disk** for the
  model, and **≥16 GB system RAM**.
- [`pi`](https://github.com/badlogic/pi-mono) for the agent front-end
  (`npm i -g @mariozechner/pi-coding-agent`) — not required just to run the server.
- The optional CUDA build also needs `git`, ~4 GB disk, and ~20 min (the toolchain is auto-installed
  into an isolated conda env — nothing system-wide).

---

## Quick start

**Fastest (recommended) — CUDA backend (~23.5 tok/s):**

```bash
BACKEND=cuda bash scripts/setup.sh    # env + model + build llama.cpp vs your CUDA (~20 min, once)
bash scripts/configure-pi.sh          # register the pi provider (once)
BACKEND=cuda bash scripts/start.sh    # server + pi
```

**Zero-build — Vulkan backend (~4.9 tok/s), works on any driver:**

```bash
bash scripts/setup.sh                 # env + model (~14 GB), no compile
bash scripts/configure-pi.sh          # register the pi provider (once)
bash scripts/start.sh                 # server + pi
```

> **Don't want to memorize env vars?** `bash scripts/start.sh --menu` walks you through backend,
> auto-tune, context, KV-cache quantization, sampling and images, then launches — and syncs pi's
> context window to the server's automatically. Every knob is also a plain env var.

Once the CUDA backend is built the run scripts **auto-select** it (no `BACKEND=cuda` needed each
time); override with `BACKEND=cuda|vulkan|cpu`. Prefer to run the two halves yourself instead of
`start.sh`:

```bash
BACKEND=cuda bash scripts/run-server.sh   # terminal A: the server (drop BACKEND for Vulkan)
bash scripts/run-pi.sh                      # terminal B: chat through pi
bash scripts/stop-server.sh                 # when done
```

The model downloads automatically on first `setup.sh` (the public unsloth QAT repo — no Hugging
Face account needed). **Full knob reference** — context size, `KVQUANT`, sampling, images, MTP,
`NCMOE`, a different quant, the raw OpenAI/curl API — is in **[scripts/README.md](scripts/README.md)**.

---

## Tested configuration

| | |
|---|---|
| **Model** | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` → `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (~14 GB) |
| **GPU** | NVIDIA RTX 2070 Max-Q, **8 GB VRAM**, driver **535** (CUDA 12.2) |
| **RAM** | 31 GB (16 GB is enough) |
| **CPU** | Intel Core i7-8750H |
| **Backend** | llama.cpp — **Vulkan** (zero-build) or **CUDA** (built from source, ~5–6× faster) |
| **Result** | reasoning + tool-calls work; **~4.9 tok/s on Vulkan, ~23.5 tok/s on CUDA** |

---

## The pi extensions

A 4-bit 26B-A4B MoE on a ~120K window is a capable but **fallible** coding agent — it ships broken
edits, repeats failing tool calls, re-reads whole files to find one line, forgets everything across
sessions. Cloud harnesses paper over this with a bigger model; here the local model is the *only*
intelligence at runtime, so [`pi-extensions/`](pi-extensions/) covers each weakness with
**deterministic code**: 17 extensions for verified edits, code outlines, loop-breaking, task plans,
cross-session memory, web search + page fetch, autonomous goals, think-time grounding, interrupt
and compaction notices, an external advisor, and more — all built to keep the KV-cache prefix
byte-stable so the harness stays cheap on a laptop.

```bash
cd pi-extensions && ./setup.sh    # npm install → symlink into ~/.pi/agent/extensions → (optional) embeddings
```

See **[pi-extensions/README.md](pi-extensions/README.md)** for the full list, the six design rules,
and install options, and [TECHNICAL.md §15](docs/TECHNICAL.md#15-the-harness-layer-pi-extensions)
for the engineering rationale — which weakness each one targets, and why those shapes and not others.

---

## Documentation

| Looking for… | See |
|---|---|
| **Scripts & every env-var knob** — context, sampling, `KVQUANT`, images, MTP, a different quant, the OpenAI/curl API | [`scripts/README.md`](scripts/README.md) |
| **The engineering write-up** — architecture, the CUDA-version wall, performance analysis, the tuning method, multimodal, running bigger models | [`docs/TECHNICAL.md`](docs/TECHNICAL.md) |
| **The pi extensions** — what each does, the six design rules, install | [`pi-extensions/README.md`](pi-extensions/README.md) |
| **The MTP speculative-decoding study** — measured gains + measurement-noise analysis | [`docs/mtp-benchmark.md`](docs/mtp-benchmark.md) |
| **Inspecting a GGUF** — architecture / modality / tensors | [`utils/README.md`](utils/README.md) |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA error: device kernel image is invalid` | Your driver is older than the build's CUDA version. Use the default Vulkan backend (don't set `BACKEND=cuda`). |
| Server picks the wrong / integrated GPU | Check `mamba run -n llamacpp llama-server --list-devices`; set `BACKEND=vulkan` (auto-detects NVIDIA) or pass the device explicitly. |
| Out of VRAM at startup | Lower `CTX`. (Note: `NCMOE` moves experts *onto* the GPU, so it uses *more* VRAM, not less.) |
| Empty `content` in API responses | Increase `max_tokens` — reasoning is eating the budget. |
| `mamba: command not found` | Activate your conda base env or add mamba to PATH before running the scripts. |

More symptoms and the deeper caveats are in [TECHNICAL.md §11](docs/TECHNICAL.md#11-caveats--gotchas).

---

## Repository layout

```
.
├── README.md                 # this file — overview + quick start + the doc map
├── environment.yml           # conda/mamba runtime deps (Vulkan path)
├── environment-build.yml     # conda/mamba build toolchain (native CUDA build)
├── scripts/
│   ├── README.md             # all scripts + every env-var knob (the operational reference)
│   ├── setup.sh              # env + model download
│   ├── start.sh              # all-in-one: server (if needed) + pi
│   ├── run-server.sh         # launch llama-server (auto CUDA/Vulkan; --image for vision)
│   ├── build-llama-cuda.sh   # build llama.cpp against the local CUDA (optional, ~5-6× faster)
│   ├── benchmark-config.sh   # probe NCMOE/CTX configs, recommend the fastest that fits (optional)
│   └── …                     # configure-pi, run-pi, stop-server, run-server-mtp, benchmark-mtp, helpers
├── pi-extensions/            # pi extensions tuned for local Gemma (hub README + one README per extension)
│   ├── README.md             # design rules, the 16 extensions, install
│   ├── setup.sh              # one-shot: npm install + link into ~/.pi/agent/extensions
│   └── <name>/               # one dir per extension: index.ts + test.mjs + README.md
├── config/
│   └── pi-provider.json      # pi provider definition
├── utils/                    # inspect-gguf (architecture / modality / tensors) — see utils/README.md
├── docs/
│   ├── TECHNICAL.md          # engineering write-up (architecture, perf, multimodal, harness)
│   ├── mtp-benchmark.md      # speculative-decoding study
│   └── speed.svg             # backend speed comparison chart (in this README)
├── models/                   # downloaded GGUF lives here (gitignored)
└── vendor/                   # llama.cpp source + CUDA build (gitignored)
```

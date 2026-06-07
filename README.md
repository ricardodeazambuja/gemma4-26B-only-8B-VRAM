# Gemma 4 26B-A4B (QAT) locally on an 8 GB GPU, driven by `pi`

Run Google's **Gemma 4 26B-A4B** mixture-of-experts model (26B total / **4B active**
parameters), in the unsloth **QAT** 4-bit GGUF (~14 GB), on a laptop with an **8 GB**
NVIDIA GPU — and use it as the backend for [`pi`](https://github.com/badlogic/pi-mono)
(a local coding agent).

The trick is llama.cpp's **`--cpu-moe`** flag: it pins the heavy MoE expert weights to
system RAM while keeping the attention layers and KV cache on the GPU. A 14 GB model
that can't possibly fit in 8 GB of VRAM then runs comfortably, because only ~4B params
are active per token.

```
            ┌──────────────── llama-server (OpenAI API :8080) ────────────────┐
   pi  ───▶ │  GPU (8 GB):  attention layers + KV cache                        │
            │  RAM (≥16 GB): all MoE expert FFN weights  (via --cpu-moe)       │
            └─────────────────────────────────────────────────────────────────┘
```

---

## TL;DR

```bash
# 1. install llama.cpp (+ model download, ~14 GB) into a conda/mamba env
bash scripts/setup.sh

# 2. register the server as a provider in pi (once)
bash scripts/configure-pi.sh

# 3. terminal A: start the model server (leave running)
bash scripts/run-server.sh

# 4. terminal B: chat through pi
pi --provider llamacpp --model gemma-4-26b-a4b-qat
```

---

## Tested configuration

| | |
|---|---|
| **Model** | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` → `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (~14 GB) |
| **GPU** | NVIDIA RTX 2070 Max-Q, **8 GB VRAM**, driver **535** (CUDA 12.2) |
| **RAM** | 31 GB (16 GB is enough) |
| **CPU** | Intel Core i7-8750H |
| **Backend** | llama.cpp (conda-forge build) on the **Vulkan** backend |
| **Result** | ~3.1 GB VRAM used, reasoning works, ~3–4 tok/s |

---

## Two findings that shaped this setup

### 1. Ollama cannot do `--cpu-moe`
`pi` is "ollama-based" only in that it speaks the OpenAI-compatible API. But **Ollama has
no per-expert CPU offload** — it only does automatic *whole-layer* GPU offload (tracked in
ollama issues [#11772](https://github.com/ollama/ollama/issues/11772),
[#14579](https://github.com/ollama/ollama/issues/14579)). To get the `--cpu-moe`
behavior we run **llama.cpp's `llama-server`** instead and point `pi` at it. `pi` doesn't
care which engine is behind the endpoint.

### 2. The conda-forge CUDA build won't run on an older driver — use Vulkan
The conda-forge `llama.cpp` package ships CUDA **12.9** / **13.0** builds. On a driver that
only supports CUDA 12.2 (e.g. driver 535), the CUDA kernels fail at load with:

```
CUDA error: device kernel image is invalid
```

CUDA *minor-version compatibility* covers the runtime **API**, but the compiled **SASS
kernels** from 12.9 are too new for the 12.2 driver to load. **Fix:** run the **Vulkan**
backend (`--device VulkanN`), which uses the driver's own ICD and has no CUDA-version wall.
`--cpu-moe` is backend-agnostic, so the RAM/VRAM split is identical. `scripts/run-server.sh`
defaults to Vulkan and auto-detects the NVIDIA Vulkan device.

> If your driver is new enough for the build's CUDA version (driver ≥ ~575 for CUDA 12.9),
> you can run `BACKEND=cuda bash scripts/run-server.sh` for more speed. Or build llama.cpp
> from source against CUDA 12.2 (`mamba install -c conda-forge cuda-toolkit=12.2`, then
> `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`).

---

## Prerequisites

- **mamba / conda** ([Miniforge](https://github.com/conda-forge/miniforge)). All other
  dependencies (llama.cpp, the Vulkan loader, `huggingface_hub`) are installed into a
  dedicated env by `scripts/setup.sh`.
- An NVIDIA GPU with a working driver (the driver provides the Vulkan ICD).
- ~14 GB free disk for the model, and ≥16 GB system RAM.
- [`pi`](https://github.com/badlogic/pi-mono) if you want the agent front-end
  (`npm i -g @mariozechner/pi-coding-agent`). Not required just to run the server.

---

## Scripts

| Script | What it does |
|---|---|
| `scripts/setup.sh` | Creates the `llamacpp` conda env (llama.cpp + huggingface_hub) and downloads the GGUF into `models/`. Idempotent. |
| `scripts/run-server.sh` | Launches `llama-server` with `--cpu-moe`, Vulkan, `--no-mmap`, `-c 16384`, `--jinja`, on `127.0.0.1:8080`. |
| `scripts/configure-pi.sh` | Adds the `llamacpp` provider to `~/.pi/agent/models.json` from `config/pi-provider.json`. |
| `config/pi-provider.json` | The pi provider definition (copy into `models.json` manually if you prefer). |

All scripts accept env-var overrides — see the header comment in each.

### Useful overrides

```bash
NCMOE=22   bash scripts/run-server.sh     # FASTEST on 8 GB: keep first 22 layers' experts
                                          # on CPU, put the last 8 layers' experts on the GPU
CTX=32768  bash scripts/run-server.sh     # larger context (more VRAM for KV cache)
BACKEND=cuda bash scripts/run-server.sh   # use CUDA (only if driver matches the build)
PORT=9000  bash scripts/run-server.sh     # different port
```

### Performance & tuning

The model is **30 layers, 128 experts/layer, top-8 routing**; in this quant each layer's
experts are ~0.45 GB. `--n-cpu-moe N` (env `NCMOE=N`) keeps the **first N** layers' experts
on CPU and puts the rest on the GPU — fewer on CPU = faster but more VRAM.

Measured on the RTX 2070 (8 GB) at `CTX=16384`, Vulkan:

| Setting | Experts on GPU | VRAM used | Eval speed |
|---|---|---|---|
| `--cpu-moe` (default) | 0 layers | ~3.5 GB | ~3.7 tok/s |
| `NCMOE=22` | 8 layers | ~6.8 GB | **~4.9 tok/s** (+34%) |

`NCMOE=22` leaves ~1.2 GB VRAM headroom — about the limit on an 8 GB card once the desktop
is using some VRAM. If you hit out-of-memory, raise `NCMOE` (e.g. 24) or lower `CTX`. For a
bigger jump, build llama.cpp against your driver's CUDA version and run `BACKEND=cuda`.

---

## Using it without pi

`llama-server` exposes a standard OpenAI API, so any client works:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26b-a4b-qat",
       "messages":[{"role":"user","content":"What is 17 times 23? Answer briefly."}],
       "max_tokens":600}' | python3 -m json.tool
```

**Reasoning note:** Gemma 4 thinks before answering. llama.cpp returns the chain-of-thought
in a separate `reasoning_content` field and the final answer in `content`. Give it a
generous `max_tokens` — too small and the thinking consumes the whole budget, leaving
`content` empty. `pi` handles this automatically.

There's also a built-in web UI at <http://127.0.0.1:8080>.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA error: device kernel image is invalid` | Your driver is older than the build's CUDA version. Use the default Vulkan backend (don't set `BACKEND=cuda`). |
| Server picks the wrong / integrated GPU | Check `mamba run -n llamacpp llama-server --list-devices`; set `BACKEND=vulkan` (auto-detects NVIDIA) or pass the device explicitly. |
| Out of VRAM at startup | Lower `CTX`. (Note: `NCMOE` moves experts *onto* the GPU, so it uses *more* VRAM, not less.) |
| Empty `content` in API responses | Increase `max_tokens` — reasoning is eating the budget. |
| `mamba: command not found` | Activate your conda base env or add mamba to PATH before running the scripts. |

---

## Repository layout

```
.
├── README.md                 # this file
├── MEMORY.md                 # working notes / decisions log
├── scripts/
│   ├── setup.sh              # env + model download
│   ├── run-server.sh         # launch llama-server (Vulkan + --cpu-moe)
│   └── configure-pi.sh       # register provider in pi
├── config/
│   └── pi-provider.json      # pi provider definition
└── models/                   # downloaded GGUF lives here (gitignored)
```

# Scripts — launch, build, benchmark, and the env-var knobs

The operational reference for everything under `scripts/`: the launchers, the optional
native-CUDA build, the benchmark tools, and every env-var knob they accept. For the
*why* behind the tuning knobs — measured throughput, the CUDA-version wall, the
auto-tune method, multimodal — see [`../docs/TECHNICAL.md`](../docs/TECHNICAL.md).

> The 3-command quick start lives in the [main README](../README.md#quick-start). This
> file is what you reach for once you want to change a knob.

## Scripts

| Script | What it does |
|---|---|
| `setup.sh` | **(once)** Creates the `llamacpp` conda env (llama.cpp + huggingface_hub) and downloads the GGUF into `models/`. `BACKEND=cuda` also builds the native CUDA backend. Idempotent. |
| `configure-pi.sh` | **(once)** Adds the `llamacpp` provider to `~/.pi/agent/models.json` from `../config/pi-provider.json`. |
| `start.sh` | **All-in-one:** starts the server (with the CUDA/Vulkan/CPU banner), waits for it to load, then launches pi. Pass **`--menu`** for a guided walk through every knob; otherwise set knobs as env vars (`BACKEND`/`NCMOE`/`CTX`/`KVQUANT`/`TEMP`/`MTP`/`NMAX`/`--image`) and other args pass to pi. On a first fresh launch it offers to **auto-tune** and remembers the result. Cleans up the server it started on exit or Ctrl-C; leaves an already-running one alone. (Full behavior in [TECHNICAL.md §9](../docs/TECHNICAL.md#9-tuning).) |
| `run-server.sh` | Launches `llama-server` with `--cpu-moe`, `--no-mmap`, `-c 32768`, `--jinja`, on `127.0.0.1:8080`. Auto-selects CUDA if built, else Vulkan; prints a color-coded backend banner at launch. Override with `BACKEND=cuda\|vulkan\|cpu`. `KVQUANT=q8_0` quantizes the KV cache (long-context lever). Pass `--image` to enable vision (loads the `mmproj`). |
| `run-server-mtp.sh` | **(optional)** Interactive launcher for the server **with MTP speculative decoding** — prompts for each knob with safe defaults, then hands off to `run-server.sh`. `DRY_RUN=1` previews the command. |
| `run-pi.sh` | Launches pi against the local server (`--provider llamacpp --model gemma-4-26b-a4b-qat`). Extra args pass through to pi. |
| `stop-server.sh` | Stops the server by the port it listens on (default 8080). |
| `build-llama-cuda.sh` | **(optional, ~5–6× faster)** Builds llama.cpp from source against your driver's CUDA version into a `llamacpp-cuda` env. Auto-detects CUDA + GPU arch, smoke-tests the result. |
| `benchmark-config.sh` | **(optional)** Finds the fastest `NCMOE`/`CTX` for *your* GPU by probing real configs on an isolated port. Pin one context with `CTX=` or sweep `CTX_LIST=`. (Method and noise handling: [TECHNICAL.md §9](../docs/TECHNICAL.md#9-tuning).) |
| `benchmark-mtp.sh` | **(optional)** Measures decode tok/s (and MTP draft acceptance) with vs without MTP for one config — used to produce [`../docs/mtp-benchmark.md`](../docs/mtp-benchmark.md). `SPEC_TYPE`/`DRAFT_MODEL` also allow other draft methods. |
| `_banner.sh` | Shared backend banner + resolution (sourced by the above). |
| `_tuning.sh` | Shared auto-tune cache: best `NCMOE` per backend+context+KV, plus your picked context & KV quant. |
| `make-speed-chart.py` | Regenerate `../docs/speed.svg` from measured numbers. |
| `../config/pi-provider.json` | The pi provider definition (copy into `models.json` manually if you prefer). |

All scripts accept env-var overrides — see the header comment in each, or run any of them with
`-h`/`--help` to print that header.

## Useful overrides

```bash
NCMOE=22    bash scripts/run-server.sh    # FASTEST on 8 GB: 8 expert layers on GPU (default is all-CPU experts)
CTX=65536   bash scripts/run-server.sh    # bigger context (see the ceiling table below)
BACKEND=cuda bash scripts/run-server.sh   # force native CUDA backend (auto-selected once built)
BACKEND=cpu bash scripts/run-server.sh    # no GPU offload — slow, benchmark baseline only
PORT=9000   bash scripts/run-server.sh    # different port
CTX=131072 KVQUANT=q8_0 bash scripts/run-server.sh   # quantize the KV cache to fit long context
bash scripts/run-server.sh --image        # enable image input (CLI flag, not an env var)
```

These env vars work on **`run-server.sh`, `start.sh`, and `build-llama-cuda.sh`** alike (e.g.
`CTX=65536 NCMOE=27 BACKEND=cuda bash scripts/start.sh`).

## A different quant, or your own file

`setup.sh` fetches the QAT 4-bit GGUF by default. The QAT repo ships **only this one 4-bit file**
(QAT is trained for 4-bit). For other precisions, use the **non-QAT** repo
[`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main),
which has the full Q2–Q8 range (Q5 ≈ 21 GB, Q6 ≈ 23 GB, Q8 ≈ 27 GB).

```bash
# a higher-precision quant of the SAME model (better quality, more RAM, SLOWER; non-QAT repo)
MODEL_REPO=unsloth/gemma-4-26B-A4B-it-GGUF \
MODEL_FILE=gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf bash scripts/setup.sh

# already have a GGUF somewhere? skip the download and point the server at it
MODEL=/path/to/model.gguf bash scripts/run-server.sh
```

(A bigger quant scales up the GPU-resident tensors too, so fewer expert layers fit in VRAM and the
RAM-resident ones are heavier — expect *slower* generation, not the same ~23 tok/s. What fits in
your RAM, and whether a bigger Gemma 4 is worth it, is in
[TECHNICAL.md §13](../docs/TECHNICAL.md#13-running-bigger-models).)

## Context size

The context window is `CTX` (**default 32768** = 32K; model max **262144**). The context is the
KV cache, and it lives in **VRAM**, not RAM (only the expert weights are offloaded) — so it shares
the 8 GB with the attention weights and on-GPU experts. **More context means fewer experts on the
GPU, which means slower** — context trades directly against speed:

| Goal | Setting | Max context on 8 GB | Speed |
|---|---|---|---|
| **Full speed** (recommended on 8 GB) | `NCMOE=22` | **~32K** | ~23 tok/s |
| Balance | `NCMOE=27`, etc. | ~64K | medium |
| **Max context** (script default) | `--cpu-moe` (omit `NCMOE`) | **~128K** (≈160K ceiling) | slower (all experts on CPU) |

(The KV cache itself is tiny — ~0.6 GB at 16K — thanks to flash attention and Gemma's
sliding-window layers, and its size is **independent of the quant**. 256K, the model's trained max,
does not fit on 8 GB.) If you change `CTX`, pass the **same** value to `configure-pi.sh`:

```bash
# e.g. a 128K window (slower): all experts to RAM, big context
CTX=131072 bash scripts/start.sh
CTX=131072 bash scripts/configure-pi.sh   # keep pi's contextWindow in sync with the server's -c
```

## KV-cache quantization (`KVQUANT`)

You can quantize the **KV cache itself** to shrink it in VRAM, freeing room for more context or more
on-GPU experts. `KVQUANT=q8_0` (near-lossless) roughly halves it; `q5_1`/`q4_0` go further but cost
quality. It's a **long-context lever** — at 32K the KV cache is already ~0.6 GB so it barely matters,
but past ~64K the freed VRAM fits a faster expert split, so the win is both **bigger usable context**
and **higher tok/s** (e.g. +38% at 128K here).

```bash
CTX=131072 KVQUANT=q8_0 bash scripts/start.sh   # 128K context with a quantized KV cache
```

(Best on CUDA; on Vulkan, flash-attention + KV-quant can be slow — the server warns.) The measured
gains, the full table, and why it forces flash attention on are in
[TECHNICAL.md §9](../docs/TECHNICAL.md#kv-cache-quantization-kvquant).

## Sampling (temperature & friends)

Set these on the **server** via env vars; they become the defaults for every request. The defaults
follow [unsloth's Gemma 4 recommendation](https://unsloth.ai/docs/models/gemma-4/qat): `TEMP=1.0`,
`TOP_P=0.95`, `TOP_K=64`.

```bash
TEMP=0.7 bash scripts/start.sh                              # more deterministic
TEMP=1.0 TOP_P=0.95 TOP_K=64 bash scripts/start.sh          # the defaults, explicit
EXTRA_ARGS="--min-p 0.01 --repeat-penalty 1.1 --seed 42" bash scripts/start.sh   # anything else
```

(`pi` doesn't expose sampling flags, so the server is where you set them. An OpenAI client that
*does* send `temperature`/`top_p` overrides the server default for that request.)

## Images (vision)

Gemma 4 is natively multimodal, but the text GGUF doesn't carry the vision encoder — llama.cpp
needs a **separate multimodal projector** (`mmproj`). Download it once (~1.2 GB, BF16), then start
the server with `--image`:

```bash
curl -L -o models/gemma4-26b-a4b-qat/mmproj-BF16.gguf \
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf
bash scripts/run-server.sh --image        # add NCMOE=22 BACKEND=cuda as usual
bash scripts/start.sh --image             # or all-in-one: --image is forwarded to the server
```

Now send images through the standard OpenAI vision format (`image_url` with a `data:image/...;base64,`
URI). Notes:

- **Vision only.** This projector is `gemma4v` — images, **not** audio. (Gemma 4 *can* do audio, but
  this BF16 file has no audio conformer; that needs a different/unified projector.)
- **Projector runs on the CPU** (`--no-mmproj-offload`): on 8 GB there's no VRAM left for a 1.2 GB
  BF16 tower beside the experts + KV. Image encoding is therefore CPU-bound, but decode stays ~full
  speed. Override the path with `MMPROJ=…` if you keep the file elsewhere.
- Without `--image` the server is **text-only** — the projector is never loaded.

Full multimodal write-up: [TECHNICAL.md §14](../docs/TECHNICAL.md#14-multimodal-images-via-the-mmproj).

## Speculative decoding (MTP)

Gemma 4 ships a **Multi-Token Prediction** head that llama.cpp uses for *lossless* self-speculative
decoding: a draft head proposes the next few tokens and the full model verifies them in one batched
pass — identical output, potentially faster generation. The 0.25 GB QAT head ships in our own model
repo and fits at the same `NCMOE`.

```bash
MTP=1 TEMP=0.7 bash scripts/start.sh           # MTP on (n-max 2), at a coding temperature
bash scripts/run-server-mtp.sh                 # interactive: pick every MTP knob with safe defaults
```

**It only pays off at low temperature** — **+15–30 % at greedy/coding temperature**, but **no
measurable gain at the default `TEMP=1.0`** (the gain falls within this rig's ±13 % run-to-run
noise). So enable it for coding work and keep the temperature low; for chat at `TEMP=1.0` it's not
worth it. The full study, the measurement-noise analysis, and why EAGLE3 doesn't work here are in
[`../docs/mtp-benchmark.md`](../docs/mtp-benchmark.md).

## Performance & tuning

The model is **30 layers, 128 experts/layer, top-8 routing**. `--n-cpu-moe N` (env `NCMOE=N`) keeps
the **first N** layers' experts on CPU and puts the rest on the GPU — fewer on CPU = faster, more
VRAM. On this 8 GB card `NCMOE=22` (8 layers' experts on the GPU) is the fast default and leaves
~1.2 GB headroom; if you hit out-of-memory, raise it (e.g. 24) or lower `CTX`.

**Build the CUDA backend — it's ~5–6× faster.** Vulkan is only the zero-build fallback; on this RTX
2070 the Vulkan MoE path (not RAM bandwidth) was the bottleneck, and CUDA took generation from
~4.9 → ~23.5 tok/s. Why MoE makes 8 GB feasible at all, plus the full llama-bench tables, are in
[TECHNICAL.md §8](../docs/TECHNICAL.md#8-performance-analysis).

```bash
bash scripts/build-llama-cuda.sh        # auto-detects your CUDA + GPU arch, ~20 min
NCMOE=22 BACKEND=cuda bash scripts/run-server.sh
```

**Don't want to guess `NCMOE`/`CTX`?** `benchmark-config.sh` probes real configs on *your* GPU and
reports the fastest that fits per context. Easier still: on a first fresh launch `start.sh` offers to
run that measurement once and **remembers** the result, so later launches reuse it instantly.
`bash scripts/start.sh --menu` is a guided front-end to all of it (backend, context, `NCMOE`,
`KVQUANT`, sampling, image). The full method — the median-of-runs noise handling, the auto-tune
cache, the menu steps — is in [TECHNICAL.md §9](../docs/TECHNICAL.md#9-tuning).

**Keep pi in sync.** The menu runs `configure-pi.sh` for you. On the env-var path it isn't
automatic — if you launch with e.g. `CTX=131072`, also run `CTX=131072 bash scripts/configure-pi.sh`
once, or pi caps the usable context at its previously configured window (32768 by default).

## Using it without pi

`llama-server` exposes a standard OpenAI API, so any client works:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26b-a4b-qat",
       "messages":[{"role":"user","content":"What is 17 times 23? Answer briefly."}],
       "max_tokens":600}' | python3 -m json.tool
```

**Reasoning note:** Gemma 4 thinks before answering. llama.cpp returns the chain-of-thought in a
separate `reasoning_content` field and the final answer in `content`. Give it a generous
`max_tokens` — too small and the thinking consumes the whole budget, leaving `content` empty. `pi`
handles this automatically. There's also a built-in web UI at <http://127.0.0.1:8080>.

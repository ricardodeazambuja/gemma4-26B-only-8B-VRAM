# Qwen on 8 GB VRAM — Qwen3.6 / Qwen3.8 bake-off

**Date:** 2026-10-16 · **Branch:** `qwen_on_8gb_vram` · **Status:** smoke tests done; distill
tried in pi by the author ("not bad"); side-by-side real-task comparison still open
**Scripts:** [`download-qwen-bakeoff.sh`](../scripts/download-qwen-bakeoff.sh) ·
[`run-qwen.sh`](../scripts/run-qwen.sh) · [`start-qwen.sh`](../scripts/start-qwen.sh) ·
[`qwen-bench.py`](../scripts/qwen-bench.py) · [`qwen-codecheck.py`](../scripts/qwen-codecheck.py)

Can a Qwen model replace Gemma4-26B-A4B on an 8 GB laptop GPU + 32 GB RAM? Three
candidates, served with llama.cpp (MoE experts or dense layers partly in system RAM) and
driven by the [pi](https://github.com/badlogic/pi-mono) coding agent.

**Test machine**

| | |
|---|---|
| GPU | NVIDIA RTX 2070 Max-Q, **8 GB** (Turing, sm_75), driver 580.178.04 (reports CUDA 13.0) |
| CPU / RAM | Intel i7-8750H (6 cores / 12 threads), **32 GB DDR4** |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138 |
| Engine | llama.cpp **`b9606`** (`88a39274ecf88ba11686acd357b59685b1cbf03d`, 2026-06-12), CUDA build via [`build-llama-cuda.sh`](../scripts/build-llama-cuda.sh) |
| Toolchain | conda-forge CUDA toolkit 12.2.2 (nvcc 12.2.140), gcc/g++ 12.4, cmake 4.2.3, ninja 1.13.2 — exact lock: [`envs/llamacpp-cuda.linux-64.lock`](../envs/llamacpp-cuda.linux-64.lock) |
| Baseline | Gemma4-26B-A4B QAT, ~22–23.5 tok/s ([`mtp-benchmark.md`](mtp-benchmark.md); its ±13 % run-to-run noise floor applies here too) |

## TL;DR

- **Qwen3.6-35B-A3B works and is the practical pick.** Same speed as Gemma (22.8 tok/s,
  single run) at **4× the context (128K)**, clean tool calls, and Qwen's own card puts it
  far ahead of Gemma4-26B-A4B on agentic coding (SWE-bench Verified 73.4 vs 17.4 —
  vendor-run, see caveats).
- **Qwen3.8-27B (dense) is too slow for agent loops on mainline llama.cpp: 2.5 tok/s.**
  The CPU half is **compute-bound** (6 cores at 600 %, GPU at 20 %) because the
  GSQ-RCO file is mostly IQ1/IQ2/IQ3 i-quants, which are costly to decode on a CPU.
  More GPU layers, hyperthreads and MTP don't fix it (details below).
- **ik_llama.cpp doesn't help:** same file is slower there (1.75 vs 2.63 tok/s at NGL 30),
  CPU-only it's no faster (0.93 vs 1.01).
- **The empero-ai Qwen3.8-35B-A3B distill is the fastest and tersest:** 26.2 tok/s at
  128K (`NCMOE=34`), tool calls ✓, and it answers in ~6× fewer tokens than Qwen3.6 on
  the same coding task (557 vs 3,558) — both produced a correct function (a second distill
  run: 784 tokens, also correct). Smoke tests, not a quality verdict.
- **Decision (2026-10-16): keep Qwen3.6 + the GSQ 27B + the distill; stop chasing 27B
  speed** (the K-quant test was cancelled before it ran).

## Run these models on your own 8 GB machine

### Requirements

- **NVIDIA GPU with 8 GB VRAM** (tested on Turing; newer should be fine). Linux.
- **32 GB system RAM** — the MoE files are 17.7–19.6 GB and mostly live in RAM. With
  16 GB, only the 27B (10.4 GB) is realistic, and it is slow (see below).
- **~52 GB free disk** for all three models + vision projectors (each model is optional).
- [Miniforge](https://github.com/conda-forge/miniforge) (`mamba`), `git`, an NVIDIA driver,
  and Node.js for [pi](https://github.com/badlogic/pi-mono) (`npm i -g @mariozechner/pi-coding-agent`; optional — any
  OpenAI-compatible client works against `http://127.0.0.1:8080/v1`).

### Steps

```bash
git clone https://github.com/ricardodeazambuja/gemma4-26B-only-8B-VRAM && cd gemma4-26B-only-8B-VRAM
git switch qwen_on_8gb_vram

# 1. The exact build toolchain: 220 pinned conda-forge packages (CUDA toolkit 12.2.2,
#    nvcc 12.2.140, gcc/g++ 12.4) — URLs + md5 in envs/llamacpp-cuda.linux-64.lock.
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock

# 2. Build llama.cpp at the tested release b9606 (= 88a39274ecf88ba11686acd357b59685b1cbf03d)
#    into vendor/llama.cpp (~20 min). CUDA_VER=12.2 matters: without it the script picks
#    your driver's CUDA version for the toolkit. The env from step 1 is reused as-is.
CUDA_VER=12.2 LLAMA_REF=b9606 bash scripts/build-llama-cuda.sh

# 3. A small env for the Hugging Face CLI used by the download script.
mamba create -y -n llamacpp -c conda-forge huggingface_hub

# 4. Download + sha256-verify the models at the pinned repo revisions
#    (any subset of: qwen36 qwen38 distill dflash).
bash scripts/download-qwen-bakeoff.sh distill qwen36

# 5. Serve one and open pi on it (pi gets the right model label and context window).
QMODEL=distill bash scripts/start-qwen.sh            # or qwen36 / qwen38
#    ...or just the server, for any OpenAI-compatible client:
QMODEL=distill bash scripts/run-qwen.sh

# 6. Smoke-test whatever is being served (speed, tool call, 15.6K-token prefill; code check).
python3 scripts/qwen-bench.py
python3 scripts/qwen-codecheck.py
```

Notes:
- **Driver:** CUDA 12.2 kernels need an NVIDIA driver that supports CUDA ≥ 12.2
  (`nvidia-smi` → "CUDA Version"; driver ≥ 535). The tested driver reports 13.0, which
  runs 12.2-built kernels fine. With an older driver, use `CUDA_VER=<your version>` and an
  unpinned toolchain (`bash scripts/build-llama-cuda.sh` creates one) — untested here.
- **GPU arch** is auto-detected (`sm_75` here); override with `CUDA_ARCH=86` etc.
- **Other names / locations:** `ENV_NAME=` and `SRC_DIR=` for the build script,
  `ENV_NAME=` for the download script, `CUDA_ENV=` / `CUDA_BIN=` for `run-qwen.sh`.
- **Newer llama.cpp** (`LLAMA_REF` unset = latest release) should work but is untested here.
- Downloads use the **pinned revisions** in the script, so you get the exact bytes tested
  here even if a repo is updated later; `FOLLOW_MAIN=1` opts into the latest uploads.

### Reproduction verified (clean room, 2026-10-16)

The steps above were run verbatim in a fresh clone of this branch, with fresh conda envs
created from the lock (`qwen-repro-cuda`) and from conda-forge (`qwen-repro-hf`,
huggingface_hub 1.29.0), and a `PATH` stripped of user tools (no `hf` on it):

| Step | Result |
|---|---|
| 1. env from lock | created (36 s, warm package cache) |
| 2. build `b9606` | built in 9 min 50 s at `88a39274…`; CUDA 12.2 toolkit, gcc 12, `sm_75` auto-detected |
| 4. pinned download (`dflash`, 1.1 GB) | fetched at `2d9571f`, **sha256 OK** |
| 5. serve `QMODEL=distill` | default `NCMOE=34`, loaded in 20 s from the clean build |
| 6. `qwen-bench.py` | **29.0 tok/s**, tool call ✓, prefill 197 tok/s (15.6K tokens) |
| 6. `qwen-codecheck.py` | 555 tokens, doctests pass, **0/500 wrong** |

Same behaviour as the original build (26.2 tok/s / 167 tok/s prefill), within the rig's
run-to-run spread. The 19.6 GB distill file was linked in rather than re-downloaded
(identical bytes, same pinned sha256).

### Tuning for *your* card — the values above fit *this* laptop

The defaults (`NCMOE=32` qwen36, `NCMOE=34` distill, `NGL=30` qwen38, 128K / 128K / 32K
context) were found by trial on a card that also drives a desktop. Your free VRAM differs, so:

1. **Check free VRAM** with `nvidia-smi` before starting (desktop, browser and other apps
   take some).
2. **MoE (`qwen36`, `distill`):** `NCMOE` = how many of the 40 layers keep their experts in
   RAM. Lower = faster but more VRAM. Start safe with `NCMOE=40` (or `NCMOE=` = all in
   RAM, ~4.5 GB VRAM at 128K), then step down by 2.
3. **Dense (`qwen38`):** `NGL` = layers on the GPU (of 64). Higher = faster, more VRAM.
4. **After every change, send a real request** (`python3 scripts/qwen-bench.py`), not just
   a load: the 27B at `NGL=36` loaded fine and then died with `CUDA error: out of memory`
   on the first prompt. Back off one step from the first setting that fails, and keep
   ~0.5 GB headroom.
5. **Out of VRAM with nothing left to move?** Lower the context (`CTX=65536`), or the KV
   type is already `q8_0` — leave it there.
6. **Threads:** use your physical core count; hyperthreads were slower here at every setting.
7. **"Killed" after a clean load** is the host-RAM OOM killer (check `journalctl -k`), not VRAM.

### Exact files (Hugging Face LFS sha256 — what `download-qwen-bakeoff.sh` verifies)

| File | Size | sha256 |
|---|---:|---|
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | 17.73 GB | `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3` |
| `mmproj-F16.gguf` (Qwen3.6) | 0.90 GB | `8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f` |
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | 10.44 GB | `63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262` |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 0.93 GB | `13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16` |
| `Qwen3.8-35B-A3B-IQ4_XS.gguf` (distill) | 19.63 GB | `b645af45431ef9b41f43cae51c9323b2d0ca84f23031d483d2105c643ae58d65` |
| `mmproj-Qwen3.8-35B-A3B-F16.gguf` (distill) | 0.90 GB | `4381cb5110074396c2c7b39221fffae0c31886aa95b674de8e103d97edf58b94` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (optional drafter) | 1.14 GB | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |

Pinned repo revisions (in `download-qwen-bakeoff.sh`): unsloth/Qwen3.6-35B-A3B-GGUF
`a483e9e`, ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF `d562806`,
empero-ai/Qwen3.8-35B-A3B-Distill-GGUF `b1f9d1d`, z-lab/Qwen3.8-27B-DFlash2-GGUF `2d9571f`.

## Candidates

| `QMODEL` | Model / file | Size | Shape | Source |
|---|---|---:|---|---|
| `qwen36` | Qwen3.6-35B-A3B `UD-IQ4_XS` | 17.7 GB | MoE, ~3B active | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| `qwen38` | Qwen3.8-27B GSQ-RCO `IQ3_XXS-mtp` | 10.4 GB | dense, + MTP head | [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) |
| `distill` | Qwen3.8-35B-A3B-Distill `IQ4_XS` | 19.6 GB | MoE, ~3B active | [empero-ai/Qwen3.8-35B-A3B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF) |

Plus each model's vision projector, and the DFlash2 drafter for the 27B
(`z-lab/Qwen3.8-27B-DFlash2-GGUF`, `Q4_K_M`, 1.1 GB). **Every file was sha256-verified
against the HF LFS oid** (hashes above). Total download ≈ 52 GB.

**Why these.** [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Qwen's own table, same harness for all models):

| | Gemma4-26B-A4B | Qwen3.6-35B-A3B |
|---|---:|---:|
| SWE-bench Verified | 17.4 | 73.4 |
| SWE-bench Pro | 13.8 | 49.5 |
| Terminal-Bench 2.0 | 34.2 | 51.5 |
| MCPMark | 14.2 | 37.0 |
| LiveCodeBench v6 | 77.1 | 80.4 |

Caveat: vendor-run, Qwen's scaffold at 200K context. Gemma4-31B scores 52 on the same
row, so the 17.4 may partly reflect the scaffold, not only the model. The gap is
consistent across every agentic row; knowledge/maths rows are close.

[Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B): SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0, LiveCodeBench v6 90.3
(vs Qwen3.6-27B 53.5 / 63.4 / 83.9). Not directly comparable with the 3.6 card's numbers.

[GSQ-RCO card](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (ISTA-DASLab, vs BF16 = GPQA-D 89.9 / LCB 85.7): `IQ3_XXS` 88.9 / 84.6;
`IQ2_S` 86.4 / 82.3; unsloth `UD-Q2_K_XL` 86.9 / 82.3; unsloth `UD-IQ2_S` 76.3 / 72.0.

## Sizing the context (KV cache, from each model's `config.json` on Hugging Face)

| Model | full-attention layers | KV heads × dim | 32K q8_0 | 128K q8_0 |
|---|---:|---|---:|---:|
| Qwen3.6-35B-A3B (and the distill) | 10 of 40 | 2 × 256 | 0.36 GB | **1.42 GB** |
| Qwen3.8-27B | 16 of 64 | 4 × 256 | **1.14 GB** | 4.55 GB |

The other layers are Gated DeltaNet (fixed-size recurrent state, no KV). Hence the
defaults in `run-qwen.sh`: **128K for the MoEs** (Qwen advises ≥128K for thinking),
**32K for the dense 27B**.

## Measurements

All runs: CUDA build, `-fa on`, q8_0 KV, `--parallel 1 -b 512 -ub 512`, sampling from the
model cards (temp 0.6 for qwen36/distill, 1.0 for qwen38; top-p 0.95, top-k 20).
Single runs unless stated — read against the ±13 % noise floor.

### Qwen3.6-35B-A3B

| Config | VRAM | Load | Gen tok/s | Prefill (15.6K tok) | Tool call |
|---|---:|---:|---:|---:|---|
| `--cpu-moe` (all experts RAM), 128K | 4.5 GB | 72 s | 15.7 ¹ | — | ✓ |
| `NCMOE=32` (8 layers' experts on GPU), 128K | **7.6 GB** | — | **22.8** | **221 tok/s** (70 s) | ✓ |

¹ measured while the ~50 GB download was running (contention).

- Thinks a lot: both 1500-token speed runs ended in `finish=length` with no answer yet.
- `NCMOE=32` leaves ~0.5 GB VRAM headroom; `NCMOE=34` is the fallback if the desktop grabs VRAM.
- Used interactively through pi by the author: works fine.

### Qwen3.8-35B-A3B-Distill (empero-ai) IQ4_XS

| Config | VRAM | Load | Gen tok/s | Prefill (15.6K tok) | Tool call |
|---|---:|---:|---:|---:|---|
| `NCMOE=32`, 128K | — | — | — | — | **OOM at load** (`cudaMalloc failed` on a 545 MiB buffer) |
| `NCMOE=34`, 128K | 7.2–7.3 GB | 10 s ² | **26.2** | 167 tok/s (93 s) | ✓ |

² warm page cache. Its experts are bigger than Qwen3.6's (file 19.6 vs 17.7 GB), hence
`NCMOE=34`. Speed run finished in 759 tokens (`finish=stop`, short thinking).

### Head-to-head code check (same prompt, `merge_intervals` + 5 doctests, max 6000 tokens)

The function was executed: its own doctests, plus 500 random inputs against a
reference implementation.

| | tokens | gen tok/s | own doctests | 500 random vs reference | time |
|---|---:|---:|---|---|---:|
| distill (`NCMOE=34`) | **557** | 25.9 | 5/5 | **0 wrong** | ~22 s |
| distill, 2nd run (`qwen-codecheck.py`) | 784 | 25.2 | 5/5 | **0 wrong** | 31 s |
| Qwen3.6 (`NCMOE=32`) | 3,558 | 17.8 ³ | 4/5 (its own expected value wrong: `[[1, 6]]` for `[[1, 7]]`) | **0 wrong** | ~200 s |

³ lower than the earlier 22.8 single run — within the rig's run-to-run spread (thermals);
not re-measured. **n = 2 (distill) / 1 (Qwen3.6)**: shows the distill's terse thinking is
real, not that it is the better coder. Reproduce with
[`qwen-codecheck.py`](../scripts/qwen-codecheck.py) (seeded random inputs).

### Qwen3.8-27B GSQ-RCO IQ3_XXS (dense), mainline llama.cpp

| Config | VRAM | Gen tok/s | Notes |
|---|---:|---:|---|
| `NGL=36`, 32K | 7.65 GB idle | — | **crashed on first request: `CUDA error: out of memory`** (cuBLAS workspace) |
| `NGL=30`, 32K (server) | 6.9 GB | **2.5** | tool call ✓ (30 s); answer completed; prefill **167 tok/s** (15.6K in 93 s) |
| `NGL=30`, 32K, `MTP=1` (n_max 3) | 7.2 GB | **2.0** | draft acceptance 57 % (436/771) at temp 1.0 → slower |

`llama-bench` sweep (`-p 0 -n 48 -fa 1 -ctk q8_0 -ctv q8_0 -r 2`; tiny KV, so high NGL
fits here but **not** in the 32K server):

| NGL | 6 threads | 12 threads |
|---:|---:|---:|
| 30 | 2.63 | 2.34 |
| 34 | 2.85 | 2.14 |
| 38 | 3.11 | 2.94 |
| 42 | 3.53 | 3.29 |

Linear fit: per-token time ≈ **0.110 s + 0.0080 s per CPU-resident layer**
(→ ~3.0 tok/s at NGL 36, ~9 tok/s only if all 64 layers were on the GPU).
Hyperthreads always hurt — keep `-t 6`.

**Diagnosis (why it is slow).** During generation `llama-server` sits at **600 % CPU**
(all 6 physical cores) with the **GPU at 20 %**. At 2.5 tok/s the CPU half streams only
~14 GB/s — well under DDR4 bandwidth — so it is **compute-bound, not bandwidth-bound**.
The file's tensor allocation (`tensor-allocation/*IQ3_XXS-mtp.rco-allocation.txt`):
IQ1_M 4, IQ1_S 4, IQ2_S 72, IQ2_XS 32, IQ2_XXS 30, IQ3_S 97, IQ3_XXS 83, IQ4_XS 33,
Q2_K 28, Q4_K 19, Q6_K 8, BF16 96, F32 360 — i.e. mostly i-quants, whose CPU decode
is expensive. **My pre-test estimate (6–8 tok/s from bandwidth) was wrong by ~3× for
exactly this reason.** MTP hurts because verifying k drafted tokens costs the CPU ~k×
when it is compute-bound.

### Qwen3.8-27B on ik_llama.cpp (`cea2d61`, built 2026-10-16) — no help

Built with the same toolchain/flags as mainline:

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp vendor/ik_llama.cpp   # tested at cea2d61ffdac38ee2958ba88d120fa07cb67703a
P="$(mamba run -n llamacpp-cuda printenv CONDA_PREFIX)"
mamba run -n llamacpp-cuda cmake -S vendor/ik_llama.cpp -B vendor/ik_llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_HOST_COMPILER="$(mamba run -n llamacpp-cuda printenv CXX)" -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_RPATH="$P/lib" -DCMAKE_INSTALL_RPATH="$P/lib"
mamba run -n llamacpp-cuda cmake --build vendor/ik_llama.cpp/build -j 10 --target llama-server llama-bench   # ~40 min
```

Same file, same flags (`-fa 1 -ctk q8_0 -ctv q8_0 -t 6 -p 0 -n 48`):

| | mainline | ik_llama.cpp | ik + `-rtr 1` (212 tensors repacked) |
|---|---:|---:|---:|
| NGL 30 | 2.63 | **1.75** | 1.77 |
| NGL 38 | 3.11 | 2.02 | — |

CPU-only isolation (`-ngl 0 -t 6 -p 128 -n 16`, one rep):

| | tg16 | pp128 |
|---|---:|---:|
| mainline | 1.01 | 57.0 |
| ik_llama.cpp | 0.93 | 56.1 |

**Verdict:** ik_llama.cpp's CPU i-quant kernels are *not* faster than this mainline build
on this file, and its hybrid path is ~1/3 slower. Repacking doesn't change it. Spec
decoding on ik is therefore not worth testing for this file (CPU cost still scales with
drafted tokens). Mainline CPU-only reads the 9.7 GB file at ~10 GB/s — far below DDR4
bandwidth — which re-confirms the i-quant decode cost and motivates the K-quant test.

## Research log — making the 27B faster (2026-10-16)

Verified against primary sources (HF cards / GitHub) unless marked *unverified*.

| Lead | Verdict |
|---|---|
| **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** | **Tested — slower** (see above). Claimed: README: "better CPU performance"; faster CPU kernels for IQ1_M (PR 327), IQ2_XS (PR 312); "much faster CPU prompt processing" (batched → may make speculative verification cheap); supports `qwen35`/`qwen35moe` with fused delta-net (PR 1315/1333/1362); `--fit` auto-fit; `-rtr` CPU repacking (README warns against it only for MoE hybrid); speculative `--spec-type mtp` and `draft` (+ DFlash impl in `common/speculative-dflash-impl.h`), `--spec-autotune`. |
| **DFlash2** drafter ([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) | Merged in mainline llama.cpp **2026-08-27** ([PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)) — our `b9606` build predates it. Drafter downloaded (1.1 GB, verified) but **not tested**: the 27B's CPU half is compute-bound, so drafting more tokens costs ~proportionally more (MTP already measured slower). |
| **K-quant build** ([unsloth `UD-Q2_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), 9.8 GB) | **Not tested** (cancelled by decision). The one remaining lever: K-quants are cheaper to decode on CPU. Unverified whether the UD mix still contains i-quants. |
| **[Swift-1.5 Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)** (UkisAI; [GSQ-RCO GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)) | Fewer thinking tokens: GPQA −42 % mean, LiveCodeBench −24.5 %, but **Terminal-Bench 2.1 only −16 % mean / ~0 % median** → little gain for agent loops. License: free non-commercial; commercial terms above a $1M revenue threshold. Cards disagree on headline speed-up (1.95× vs 9.18×). |
| [Whittle-Qwen-3.8-35B-A3B](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B) (logic65) | Skip. One-person research preview, trained on 4K-token rows only; its own table doesn't beat stock Qwen3.6-35B-A3B. |
| [Qwen3.8-Whittle-16B](https://huggingface.co/logic65/Qwen3.8-Whittle-16B) (27B depth+width pruned) | Skip. Card: "**Non-starter for agent and long multi-turn use: repetition looping**". |
| [Qwen3.8-14B-Instruct-Turbo](https://huggingface.co/mradermacher/Qwen3.8-14B-Instruct-Turbo-GGUF) (mradermacher quants) | Skip. Source repo `ewinregirgojr/...` no longer accessible — unverifiable. |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) [REAP-288](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF) / [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Skip. Smallest builds 58–66 GB+ — exceed 32 GB RAM + 8 GB VRAM. |
| Web search claim: spec decoding gives 2–3× when the target is partly on CPU | *Unverified*, and **contradicted by our MTP measurement** on this file (it assumes a bandwidth-bound CPU). ik_llama.cpp's CPU kernels turned out no faster, so it doesn't apply there either. |

## Next steps

1. ~~ik_llama.cpp vs mainline~~ — done, slower (2026-10-16).
2. ~~ik_llama.cpp spec decoding~~ — skipped: CPU kernels no faster, so verification still scales.
3. ~~unsloth `UD-Q2_K_XL`~~ — cancelled by decision (keep the current files). Still the
   one untested lever if the 27B speed matters again.
4. ~~Test the distill~~ — done (above).
5. **Real-task comparison through pi: Qwen3.6 vs the distill.** 2026-10-16: the author
   tried the distill interactively in pi — verdict "not bad". A side-by-side on the same
   tasks against Qwen3.6 is still to do.
   Original note: (the numbers above are
   smoke tests, not quality). The distill's card only reports MMLU/ARC, and its training
   traces were math/code-weighted — watch tool use and long sessions specifically.

## Reproduce the measurements

```bash
bash scripts/download-qwen-bakeoff.sh                 # ~52 GB, sha256-verified
QMODEL=qwen36 bash scripts/run-qwen.sh                # NCMOE=32 default; then, in another shell:
python3 scripts/qwen-bench.py                         # speed + tool call + 15.6K prefill
python3 scripts/qwen-codecheck.py                     # head-to-head code check
QMODEL=distill bash scripts/run-qwen.sh               # NCMOE=34 default (32 OOMs)
QMODEL=qwen38 bash scripts/run-qwen.sh                # NGL=30 default (36 OOMs at 32K)
mamba run -n llamacpp-cuda vendor/llama.cpp/build/bin/llama-bench \
  -m models/qwen38-27b-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  -ngl 30,34,38,42 -t 6,12 -p 0 -n 48 -fa 1 -ctk q8_0 -ctv q8_0 -r 2
```

## References

Models (Hugging Face)
- Qwen3.6-35B-A3B — https://huggingface.co/Qwen/Qwen3.6-35B-A3B · GGUF: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- Qwen3.8-27B — https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B GSQ-RCO GGUF (ISTA-DASLab) — https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
- Qwen3.8-35B-A3B-Distill (empero-ai) — https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill · GGUF: https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF
- Qwen3.8-27B DFlash2 drafter — https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF

Methods / papers
- GSQ (Gumbel-Softmax Quantization) — https://arxiv.org/abs/2604.18556 · code https://github.com/IST-DASLab/GSQ
- RCO (Riemannian Constrained Optimization) — https://arxiv.org/abs/2605.00649 · code https://github.com/IST-DASLab/RCO
- DFlash — https://arxiv.org/abs/2602.06036 · https://github.com/z-lab/dflash · DFlash 2 blog https://inco.ai/blog/dflash2/

Software
- llama.cpp — https://github.com/ggml-org/llama.cpp (tested at tag `b9606`); DFlash2 support: https://github.com/ggml-org/llama.cpp/pull/27342
- ik_llama.cpp — https://github.com/ikawrakow/ik_llama.cpp (tested at `cea2d61`)
- pi coding agent — https://github.com/badlogic/pi-mono
- Miniforge (mamba) — https://github.com/conda-forge/miniforge

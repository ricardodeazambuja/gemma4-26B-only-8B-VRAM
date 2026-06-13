# MTP (Multi-Token Prediction) benchmark — measured on this rig

**Date:** 2026-06-12 · **Branch:** `feat/mtp-benchmark` · **Script:** [`scripts/benchmark-mtp.sh`](../scripts/benchmark-mtp.sh)

Self-speculative decoding via Gemma 4's MTP head, measured with vs without MTP on the
actual 8 GB rig across sampling temperature, context length (32k vs 64k), `n-max`, and
the `--spec-draft-p-min` lever. Settles [`TECHNICAL.md` §13](TECHNICAL.md) with numbers
— and quantifies the measurement noise so the numbers mean something.

## TL;DR

- **MTP is lossless and free** (0.25 GB QAT head ships in our repo; CUDA build already
  supports it; loads at the same `NCMOE`). It's a **clear win at greedy / low
  temperature (+15–30 %)** and **no measurable gain at the default temp 1.0**.
- **64k context is essentially free vs 32k** (~the same tok/s) — Gemma 4's
  sliding-window attention (`n_swa=1024`) caps most layers' KV, so the bigger context
  barely costs VRAM or speed. **Use 64k.**
- **`--spec-draft-p-min` is a footgun here:** `p_min>0` with temperature sampling drove
  the output into **degenerate loops** (`fmt-fmt-fmt…`) in two independent configs. Its
  big-looking speedups are garbage. Leave it at 0.
- **EAGLE3 was tried — it doesn't work here.** We converted the Gemma 4 EAGLE3 draft
  (`RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`), rebuilt llama.cpp at `88a3927`, and
  measured it: only ~42 % draft acceptance (vs MTP's ~80 %) ⇒ **slower than baseline even at
  greedy** (−19 % at n-max 3), **degenerates at temp>0**, and **segfaults on the chat-template
  path** (real use via pi). The fresh eagle3 support in this build is too immature — revisit
  only on a much newer llama.cpp. **So there is no temp-1.0 spec-decoding win on this rig today.**

## The measurement-noise floor (read before trusting any single number)

RTX 2070 Max-Q laptop GPU + CPU-resident MoE experts over DDR4 → thermally throttled and
bandwidth-noisy. Across **many baseline runs that are all the *same* config**, trimmed-mean
throughput was **≈ 21.9 ± 1.4 tok/s** (±6 % between configs; individual reps spanned
14.9–25 tok/s). **⇒ any speedup under ≈ ±13 % is not resolvable** and is reported as
"within noise." Controls used everywhere below: a baseline interleaved **immediately
before** each MTP run (same thermal state), **trimmed means** (drop slowest+fastest of
5–6 reps), mean ± std.

## Setup

| | |
|---|---|
| GPU | RTX 2070 Max-Q, 8 GB · llama.cpp `vendor/` @ `04eb4c4` (#23398, supports `draft-mtp`) |
| target / head | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14 GB) · `mtp-gemma-4-26B-A4B-it.gguf` (**0.25 GB**, in our own repo) |
| fixed | `-ngl 99 --n-cpu-moe 27 --no-mmap -ctk q8_0 -ctv q8_0 -fa on`; 256-token gen; one LRU-cache coding prompt; seed 42 |

**Method.** Throwaway `llama-server` per variant; tok/s = `timings.predicted_per_second`
from `/completion`; acceptance scraped from the server log. Greedy is lossless ⇒ baseline
and MTP generate identical tokens (true same-work comparison).

## 1. MTP speedup vs sampling temperature (n-max 2, p_min 0)

| temp | context | baseline | MTP | speedup | accept | verdict |
|---|---|---:|---:|---:|---:|---|
| **greedy** | 32k | 22.9 | 27.1 | **+19 %** | 79 % | **real** |
| **greedy** | 32k (n-max 1) | 18.9 | 24.6 | **+31 %** | 88 % | real |
| **greedy** | 32k (n-max 4) | 21.5 | 27.5 | **+28 %** | 84 % | real |
| **greedy** | 64k | 21.2 | 24.0 | **+13 %** | 79 % | real (noisy) |
| temp 1.0 | 32k | 23.1 | 24.0 | +4 % | 74 % | within noise |
| temp 1.0 | 64k | 23.9 | 25.9 | +8 % | 74 % | within noise |

**Read:** a real win at greedy across both contexts; **no measurable gain at temp 1.0**.
The limiter is draft acceptance, which falls with temperature. (Intermediate temps 0.3/0.6
were measured but are *not* a usable curve — single prompt + single seed means each temp
generates different text, and acceptance came out *higher* at 0.6 than greedy, which is
impossible for the same content. Content luck, not a temperature law.)

## 2. Context: 64k vs 32k (baseline, NCMOE sweep)

| context | NCMOE 25 | NCMOE 27 | NCMOE 29/31 |
|---|---:|---:|---:|
| 32k | 21.9 | 22.7 | — |
| 64k | 25.4 | 21–24 | ~21 |

All NCMOE values **fit at 64k** (no OOM). Within the ±13 % noise, **64k ≈ 32k** — the
larger context is nearly free because Gemma 4's **sliding-window attention** (`n_swa=1024`)
bounds most layers' KV. Decode speed is governed by **NCMOE (expert placement), not
context length** (256-token generations keep the live KV tiny). Lower NCMOE trends faster
(more experts on GPU) but fine NCMOE distinctions are mostly inside the noise/thermal band.
**Conclusion: run 64k; it does not cost meaningful tok/s.**

## 3. `--spec-draft-p-min` — a footgun (do not use at temp > 0)

`p_min` is a *draft-side confidence floor* (`common/speculative.cpp:706`: the head stops
drafting once its top-token prob drops below `p_min`) — **not** a relaxed target-accept
rule. Two configs, both at temp 1.0, both **degenerated**:

| config (temp 1.0, 64k) | tok/s | accept | **output** |
|---|---:|---:|---|
| n-max 6, p_min 0 | 24.0 | 58 % | coherent — but **no gain** (= baseline) |
| n-max 6, p_min 0.7 | 29.4 | 85 % | ⚠️ **degenerate** (`fmt-1.0-1.0-…-0-0-0`) |
| n-max 4, p_min 0.75 | (≈32) | 100 % | ⚠️ **degenerate** (`fmt-fmt-fmt…`) |

The eye-catching speedups are an artifact: a trivial repeating loop is perfectly
MTP-predictable (hence the 85–100 % acceptance) and fast — but it's garbage. `p_min=0` on
the same seed produced coherent code. **At greedy, p_min is safe** (lossless ⇒ coherent;
n-max 6 / p_min 0.7 gave a normal +19 %). The takeaway: in this build, `p_min>0` +
`draft-mtp` + temperature sampling is **not output-safe** — this is the config a Twitter
post suggested (`--spec-draft-n-max 6 --spec-draft-p-min 0.7 -c 80000`); it looks fast only
because the text collapses. *(Plausibly a llama.cpp correctness bug; worth verifying on a
newer build before trusting any p_min>0 result.)*

## 4. Mechanism correction

`TECHNICAL.md` had argued MTP can't help a `--cpu-moe` MoE ("verifying K tokens activates
the *union* of experts ⇒ more RAM traffic"). The greedy win **refutes** that: batched
verification *amortizes* expert-weight streaming across accepted tokens. **The limiter is
draft acceptance, not RAM bandwidth** — and acceptance falls with temperature, which is why
the temp-1.0 gain vanishes.

## Recommendation

- **Enable MTP for low-temperature / coding work** (`--spec-type draft-mtp --spec-draft-n-max 2`,
  `p_min` left at 0): +15–30 %, lossless, free, fits at the current NCMOE.
- **At default temp 1.0 it buys nothing measurable** — don't expect a chat speedup.
- **Run 64k** — it costs ~nothing vs 32k.
- **Do not use `--spec-draft-p-min > 0`** until the temp>0 degeneration is confirmed fixed.
- **EAGLE3 doesn't help (tried).** On `88a3927` it's slower than baseline (~42 % acceptance),
  degenerates at temp>0, and segfaults on the chat path — not usable; revisit on a much newer
  llama.cpp. For a faster temp-1.0 experience the lever is *lowering the sampling temperature*
  for coding, not speculative decoding.

## Reproduce / raw data

`NMAX=2 TEMP=0 bash scripts/benchmark-mtp.sh`. Raw per-run tok/s + acceptance in
`/tmp/mtp_bench/{r2..r7}/` and `out-r*.log`; consolidated re-analysis `analyze.py`.
Degenerate-output check: `NMAX=6 PMIN=0.7 TEMP=1.0 bash /tmp/mtp_bench/inspect.sh`.

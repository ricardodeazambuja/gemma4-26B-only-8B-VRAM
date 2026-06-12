# MTP (Multi-Token Prediction) benchmark — measured on this rig

**Date:** 2026-06-12 · **Branch:** `feat/mtp-benchmark` · **Script:** [`scripts/benchmark-mtp.sh`](../scripts/benchmark-mtp.sh)

Self-speculative decoding via Gemma 4's Multi-Token Prediction head, measured
with vs without MTP on the actual 8 GB rig. This settles the open question in
[`TECHNICAL.md` §13](TECHNICAL.md) ("worth measuring") with numbers.

## TL;DR

MTP is **lossless, free, and loads at the same `NCMOE`** as baseline (the draft
head is only 0.25 GB and ships inside our own repo). The throughput gain is
**strongly acceptance-limited, hence sampling-temperature-dependent**:

| sampling | draft acceptance | speedup vs baseline |
|---|---|---|
| **greedy** (temp 0) | 79–88 % | **+18 % … +31 %** |
| **temp 1.0** (rig default) | 66–74 % | **+4 % … +7 %** (noisy, near break-even) |

So the *greedy* number is a misleading ceiling. At the rig's real sampling
(temp 1.0, top-p 0.95, top-k 64) the win shrinks to a marginal few percent.

**Correction to the prior reasoning.** `TECHNICAL.md` had argued MTP can't help a
`--cpu-moe` MoE because "verifying K draft tokens activates the *union* of experts
⇒ more RAM traffic." The greedy result (+25 %) **refutes** that: batched
verification *amortizes* expert-weight streaming across accepted tokens
(consecutive tokens route to overlapping experts), which is exactly why
speculative decoding helps a memory-bound MoE. The real limiter is **acceptance
rate**, which collapses under temperature sampling — not RAM traffic.

## Setup

| | |
|---|---|
| GPU | RTX 2070 Max-Q, 8 GB |
| llama.cpp | local CUDA build `vendor/llama.cpp` @ `04eb4c4` ("llama : add Gemma4 MTP", #23398) — **already supports `--spec-type draft-mtp`, no rebuild needed** |
| target | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14 GB) |
| draft head | `mtp-gemma-4-26B-A4B-it.gguf` (**0.25 GB**, smart-4bit, shipped *inside* `unsloth/gemma-4-26B-A4B-it-qat-GGUF`) |
| placement | `-ngl 99 --n-cpu-moe 27` (MTP fits at the **same** NCMOE — no expert-placement penalty) |
| context / KV | `-c 32768 -ctk q8_0 -ctv q8_0 -fa on` |
| workload | 256-token generation, fixed seed, an LRU-cache coding prompt (representative of the agent's real use) |

**Method.** A throwaway `llama-server` per variant; decode throughput =
`timings.predicted_per_second` from the native `/completion` endpoint; draft
acceptance scraped from the server log (the JSON omits it). To cancel Max-Q
thermal drift, a baseline is run **immediately before** each MTP variant and the
two are compared as a pair. Each number is the **trimmed mean of 5 reps**
(slowest/cold-start run dropped). Greedy (temp 0) is lossless, so baseline and MTP
generate the *identical* 256 tokens — a true same-work comparison.

## Results

### Greedy (temp 0) — acceptance ceiling

| n-max | baseline (tok/s) | MTP (tok/s) | speedup | acceptance |
|---:|---:|---:|---:|---:|
| 1 | 19.3 | 25.3 | **+31 %** | 87.5 % |
| 2 | 23.2 | 27.4 | **+18 %** | 79.2 % |
| 4 | 22.1 | 28.2 | **+28 %** | 84.1 % |

All three pairs positive; distributions barely overlap. MTP ran in the *hotter*
slot (after baseline each round), so thermal works **against** MTP — the win is if
anything conservative.

### temp 1.0 (top-p 0.95, top-k 64) — real-usage sampling

| n-max | baseline (tok/s) | MTP (tok/s) | speedup | acceptance |
|---:|---:|---:|---:|---:|
| 2 | 23.2 | 24.7 | **+6.5 %** | 73.7 % |
| 4 | 24.0 | 24.9 | **+3.7 %** | 65.7 % |

Acceptance drops ~15 points vs greedy and the gain mostly evaporates. At n-max=4
some requests dipped to ~17 tok/s (rejected drafts waste a verify batch), so
**n-max=2 is the safer default** under sampling — less downside, similar mean.
Baseline anchor reproduced the documented **~23 tok/s** throughout, validating the
comparison.

## Recommendation

- **It never makes output worse** (lossless) and costs only a 0.25 GB head that
  fits at the current NCMOE — so enabling it is low-risk.
- **The payoff is real but temperature-gated.** For default temp-1.0 chat it's a
  marginal few percent. For lower-temperature / coding work (where acceptance is
  higher) it trends toward the greedy ceiling and is worth turning on.
- If enabled, use **`--spec-type draft-mtp --spec-draft-n-max 2`** (Unsloth's
  starting point; least downside under sampling). Reproduce with:
  `NMAX=2 bash scripts/benchmark-mtp.sh`.

## Raw data

`/tmp/mtp_bench/{r2,r3}/results.tsv` and `server-*.log` (per-run tok/s and
per-request draft-acceptance lines). Round 1 (a first, noisier non-interleaved
sweep) corroborates the greedy direction (n1≈26, n4≈28 vs baseline≈21–22).

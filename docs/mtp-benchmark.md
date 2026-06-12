# MTP (Multi-Token Prediction) benchmark — measured on this rig

**Date:** 2026-06-12 · **Branch:** `feat/mtp-benchmark` · **Script:** [`scripts/benchmark-mtp.sh`](../scripts/benchmark-mtp.sh)

Self-speculative decoding via Gemma 4's Multi-Token Prediction head, measured with
vs without MTP on the actual 8 GB rig. Settles the open question in
[`TECHNICAL.md` §13](TECHNICAL.md) with numbers — and, just as importantly,
quantifies the measurement noise so the numbers mean something.

## TL;DR

MTP is **lossless, free, and loads at the same `NCMOE`** as baseline (the 0.25 GB
QAT head ships inside our own repo; the CUDA build already supports it — no rebuild).
The throughput gain is **acceptance-limited**, and acceptance collapses under
temperature sampling:

| sampling | draft acceptance | MTP speedup | verdict |
|---|---|---|---|
| **greedy** (temp 0) | 79–88 % | **+19 % … +31 %** | **real** (well above noise) |
| **temp 1.0** (rig default) | 66–74 % | +2 % … +7 % | **within measurement noise** |

So: a **clear win at greedy / low temperature**, and **no measurable gain at the
default temp 1.0**. The `--spec-draft-p-min` "relaxed drafting" lever did **not**
help (see below — its one big-looking number was a degenerate-output artifact).

> **Still open (Experiment 3):** all of this is at **CTX 32k**. The rig's target is
> **64k**, where the larger KV cache forces more experts onto the CPU — likely
> slower, and possibly changing MTP's payoff. The 64k↔32k decision is measured
> separately.

## The measurement-noise floor (read this before trusting any single number)

This rig is an **RTX 2070 Max-Q laptop GPU** with CPU-resident MoE experts streaming
over DDR4 — both thermally throttled and bandwidth-noisy. Across **8 baseline runs
that are all the *same* configuration** (sampling temperature does not change
baseline decode speed), the trimmed-mean throughput was:

> **21.9 ± 1.4 tok/s** (±6 % between configs; individual reps spanned **14.9–24.9**,
> worst single-config excursion −14 %).

**⇒ any speedup smaller than ≈ ±13 % is not resolvable on this rig** and must be
reported as "within noise," not as a number. This is why the table below has *few*
rows we trust rather than many we don't. To beat the noise we use: a baseline run
**interleaved immediately before** each MTP run (same thermal state), **trimmed
means** (drop slowest+fastest of 6 reps), and we report **mean ± std**.

## Setup

| | |
|---|---|
| GPU | RTX 2070 Max-Q, 8 GB |
| llama.cpp | local CUDA build `vendor/llama.cpp` @ `04eb4c4` ("llama : add Gemma4 MTP", #23398) — already supports `--spec-type draft-mtp` |
| target | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14 GB) |
| draft head | `mtp-gemma-4-26B-A4B-it.gguf` (**0.25 GB**, smart-4bit, inside `unsloth/gemma-4-26B-A4B-it-qat-GGUF`) |
| placement | `-ngl 99 --n-cpu-moe 27` (MTP fits at the **same** NCMOE — no expert-placement penalty) |
| context / KV | `-c 32768 -ctk q8_0 -ctv q8_0 -fa on` |
| workload | 256-token generation, fixed seed (42), one LRU-cache coding prompt |

**Method.** A throwaway `llama-server` per variant; decode throughput =
`timings.predicted_per_second` from `/completion`; draft acceptance scraped from the
server log (the JSON omits it). Greedy is lossless ⇒ baseline and MTP generate the
*identical* 256 tokens (a true same-work comparison); at temp > 0 they diverge.

## Results (trimmed mean ± std, paired baseline immediately before each MTP)

| config | baseline tok/s | MTP tok/s | speedup | draft accept | verdict |
|---|---:|---:|---:|---:|---|
| greedy, n-max 1 | 18.9 ± 0.6 | 24.6 ± 2.3 | **+31 %** | 87.5 % | real |
| greedy, n-max 2 | 22.9 ± 0.8 | 27.1 ± 1.0 | **+19 %** | 79.2 % | real |
| greedy, n-max 4 | 21.5 ± 2.6 | 27.5 ± 2.7 | **+28 %** | 84.1 % | real |
| temp 1.0, n-max 2 | 23.1 ± 0.0 | 24.0 ± 2.3 | +4 % | 73.7 % | **within noise** |
| temp 1.0, n-max 4 | 23.7 ± 0.6 | 24.2 ± 2.9 | +2 % | 65.7 % | **within noise** |

At greedy, output is identical both sides, so these are the cleanest rows — and the
win (+19–31 %) clears the ±13 % floor comfortably. At temp 1.0 the gain sits *inside*
the noise band; the honest read is **no measurable speedup at the default sampling**.

## What did *not* survive triage

- **Intermediate temps (0.3, 0.6) are not a usable curve.** Measured at n-max 2 they
  read +15 % and +34 % — but acceptance came out *higher* at temp 0.6 (90 %) than at
  greedy (79 %), which is impossible for the same content. Cause: **single prompt +
  single seed** ⇒ each temperature generates *different text*, and acceptance is
  content-dependent. These points measure content luck, not a temperature law.
- **`--spec-draft-p-min` did not yield a real gain.** It is a *draft-side confidence
  floor* (`common/speculative.cpp:706`: the MTP head stops drafting once its top-token
  probability drops below `p_min`) — **not** a relaxed target-accept rule, so it cannot
  "rescue" good drafts the target would reject. Swept at temp 1.0 / n-max 4:
  `p_min=0.5` → −3 % (within noise); `p_min=0.75` → an eye-catching **+52 % / 100 %
  acceptance** that turned out to be a **degenerate-output artifact** — the generation
  collapsed to `fmt-fmt-fmt…` (a trivial loop is perfectly MTP-predictable and fast,
  but it's garbage). `p_min=0` on the same seed produced coherent code. Net: p_min is
  not a usable throughput lever here, and high p_min can co-occur with degenerate samples.

## Mechanism correction

`TECHNICAL.md` had argued MTP can't help a `--cpu-moe` MoE because "verifying K draft
tokens activates the *union* of experts ⇒ more RAM traffic." The greedy win **refutes**
that: batched verification *amortizes* expert-weight streaming across accepted tokens
(consecutive tokens route to overlapping experts), which is exactly why speculative
decoding helps a memory-bound MoE. **The limiter is draft acceptance, not RAM bandwidth** —
and acceptance falls with temperature, which is why the temp-1.0 gain vanishes.

## Caveat

Acceptance is reported from a **single prompt + single seed**, so it is one content
sample, not an average — fine for the coarse greedy-vs-temp-1.0 contrast, not for
fine-grained per-temperature claims (hence the triage above).

## Recommendation

- **Lossless ⇒ enabling it never hurts output**, and it costs only a 0.25 GB head that
  fits at the current NCMOE — low risk.
- **At the default temp 1.0 it buys nothing measurable.** Its value is at **greedy /
  low-temperature** work (coding), where +20–30 % is real.
- If enabled, use **`--spec-type draft-mtp --spec-draft-n-max 2`**; leave `p_min` at 0.
- Reproduce: `NMAX=2 TEMP=0 bash scripts/benchmark-mtp.sh`.

## Raw data

`/tmp/mtp_bench/{r2,r3,r4}/` (`results.tsv`, `server-*.log`) + the round driver logs
`out-r2/r3/r4.log`. Consolidated re-analysis (one trim policy over all rounds):
`/tmp/mtp_bench/analyze.py`.

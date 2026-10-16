# Qwen bake-off — can a Qwen model replace Gemma 4 on this rig?

**Started:** 2026-10-16 · **Branch:** `feat/test-qwen-bakeoff` · **Status:** smoke tests done; real-task comparison pending
**Scripts:** [`download-qwen-bakeoff.sh`](../scripts/download-qwen-bakeoff.sh) ·
[`run-qwen.sh`](../scripts/run-qwen.sh) · [`start-qwen.sh`](../scripts/start-qwen.sh) ·
[`qwen-bench.py`](../scripts/qwen-bench.py)

Rig: RTX 2070 Max-Q **8 GB**, i7-8750H (6C/12T), **32 GB DDR4**, vendored llama.cpp `88a3927`
(2026-06-12, CUDA build). Baseline to beat: Gemma4-26B-A4B QAT, ~22–23.5 tok/s
(see [`mtp-benchmark.md`](mtp-benchmark.md); its ±13 % noise floor applies here too).

## TL;DR (so far)

- **Qwen3.6-35B-A3B works and is the practical pick.** Same speed as Gemma (22.8 tok/s,
  single run) at **4× the context (128K)**, clean tool calls, and Qwen's own card puts it
  far ahead of Gemma4-26B-A4B on agentic coding (SWE-bench Verified 73.4 vs 17.4 —
  vendor-run, see caveats).
- **Qwen3.8-27B (dense) is too slow for agent loops on mainline llama.cpp: 2.5 tok/s.**
  The CPU half is **compute-bound** (6 cores at 600 %, GPU at 20 %) because the
  GSQ-RCO file is mostly IQ1/IQ2/IQ3 i-quants, which are costly to decode on a CPU.
  More GPU layers, hyperthreads and MTP don't fix it (details below).
- **ik_llama.cpp doesn't help:** same file is slower there (1.75 vs 2.63 tok/s at NGL 30),
  CPU-only it's no faster (0.93 vs 1.01). **Next:** a K-quant build (unsloth `UD-Q2_K_XL`).
- **The empero-ai Qwen3.8-35B-A3B distill is the fastest and tersest:** 26.2 tok/s at
  128K (`NCMOE=34`), tool calls ✓, and it answers in ~6× fewer tokens than Qwen3.6 on
  the same coding task (557 vs 3,558) — both produced a correct function. One sample
  each: a smoke test, not a quality verdict.
- **Decision (2026-10-16): keep Qwen3.6 + the GSQ 27B + the distill; stop chasing 27B
  speed** (the K-quant test was cancelled before it ran).

## Candidates

| `QMODEL` | Model / file | Size | Shape | Source |
|---|---|---:|---|---|
| `qwen36` | Qwen3.6-35B-A3B `UD-IQ4_XS` | 17.7 GB | MoE, ~3B active | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| `qwen38` | Qwen3.8-27B GSQ-RCO `IQ3_XXS-mtp` | 10.4 GB | dense, + MTP head | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` |
| `distill` | Qwen3.8-35B-A3B-Distill `IQ4_XS` | 19.6 GB | MoE, ~3B active | `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` |

Plus each model's vision projector, and the DFlash2 drafter for the 27B
(`z-lab/Qwen3.8-27B-DFlash2-GGUF`, `Q4_K_M`, 1.1 GB). **Every file was sha256-verified
against the HF LFS oid.** Total download ≈ 52 GB; disk (`/dev/sda1`) had 327 GB free.

**Why these.** Qwen3.6-35B-A3B model card (Qwen's own table, same harness for all models):

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

Qwen3.8-27B card: SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0, LiveCodeBench v6 90.3
(vs Qwen3.6-27B 53.5 / 63.4 / 83.9). Not directly comparable with the 3.6 card's numbers.

GSQ-RCO card (ISTA-DASLab, vs BF16 = GPQA-D 89.9 / LCB 85.7): `IQ3_XXS` 88.9 / 84.6;
`IQ2_S` 86.4 / 82.3; unsloth `UD-Q2_K_XL` 86.9 / 82.3; unsloth `UD-IQ2_S` 76.3 / 72.0.

## Sizing the context (KV cache, from the HF `config.json`)

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
- Used interactively through pi by the user: works fine.

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
| Qwen3.6 (`NCMOE=32`) | 3,558 | 17.8 ³ | 4/5 (its own expected value wrong: `[[1, 6]]` for `[[1, 7]]`) | **0 wrong** | ~200 s |

³ lower than the earlier 22.8 single run — within the rig's run-to-run spread (thermals);
not re-measured. **n = 1 per model**: shows the distill's terse thinking is real, not that it
is the better coder.

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
| **ik_llama.cpp** (`ikawrakow/ik_llama.cpp`) | **Tested — slower** (see above). Claimed: README: "better CPU performance"; faster CPU kernels for IQ1_M (PR 327), IQ2_XS (PR 312); "much faster CPU prompt processing" (batched → may make speculative verification cheap); supports `qwen35`/`qwen35moe` with fused delta-net (PR 1315/1333/1362); `--fit` auto-fit; `-rtr` CPU repacking (README warns against it only for MoE hybrid); speculative `--spec-type mtp` and `draft` (+ DFlash impl in `common/speculative-dflash-impl.h`), `--spec-autotune`. |
| **DFlash2** drafter (`z-lab/Qwen3.8-27B-DFlash2-GGUF`) | Merged in mainline llama.cpp **2026-08-27** (PR #27342) — our vendored build predates it. Drafter downloaded (1.1 GB, verified). Test on ik_llama.cpp. |
| **K-quant build** (unsloth `UD-Q2_K_XL`, 9.8 GB) | Untested fallback: K-quants are cheaper to decode on CPU. Unverified whether the UD mix still contains i-quants. |
| **Swift-1.5 Qwen3.8-27B** (UkisAI, also in GSQ-RCO) | Fewer thinking tokens: GPQA −42 % mean, LiveCodeBench −24.5 %, but **Terminal-Bench 2.1 only −16 % mean / ~0 % median** → little gain for agent loops. License: free non-commercial; commercial terms above a $1M revenue threshold. Cards disagree on headline speed-up (1.95× vs 9.18×). |
| Whittle-Qwen-3.8-35B-A3B (logic65) | Skip. One-person research preview, trained on 4K-token rows only; its own table doesn't beat stock Qwen3.6-35B-A3B. |
| Qwen3.8-Whittle-16B (27B depth+width pruned) | Skip. Card: "**Non-starter for agent and long multi-turn use: repetition looping**". |
| Qwen3.8-14B-Instruct-Turbo (mradermacher quants) | Skip. Source repo `ewinregirgojr/...` no longer accessible — unverifiable. |
| Qwen3.8-Flash-Next REAP-288 / GSQ-RCO | Skip. Smallest builds 58–66 GB+ — exceed 32 GB RAM + 8 GB VRAM. |
| Web search claim: spec decoding gives 2–3× when the target is partly on CPU | *Unverified*, and **contradicted by our MTP measurement** on this file (it assumes a bandwidth-bound CPU). May hold on ik_llama.cpp — that's what we test. |

Process note: the Antigravity web searches repeatedly paused to ask to run commands.
Allowed: URL reads, `mkdir`, `grep`. Declined: Python scripts (one URL-resolver ran
briefly before the decline arrived; it only fetched URLs).

## Next steps

1. ~~ik_llama.cpp vs mainline~~ — done, slower (2026-10-16).
2. ~~ik_llama.cpp spec decoding~~ — skipped: CPU kernels no faster, so verification still scales.
3. ~~unsloth `UD-Q2_K_XL`~~ — cancelled by decision (keep the current files). Still the
   one untested lever if the 27B speed matters again.
4. ~~Test the distill~~ — done (above).
5. **Real-task comparison through pi: Qwen3.6 vs the distill** (the numbers above are
   smoke tests, not quality). The distill's card only reports MMLU/ARC, and its training
   traces were math/code-weighted — watch tool use and long sessions specifically.

## Reproduce

```bash
bash scripts/download-qwen-bakeoff.sh                 # ~52 GB, sha256-verified
QMODEL=qwen36 bash scripts/run-qwen.sh                # NCMOE=32 default; then, in another shell:
python3 scripts/qwen-bench.py
QMODEL=distill bash scripts/run-qwen.sh               # NCMOE=34 default (32 OOMs)
QMODEL=qwen38 bash scripts/run-qwen.sh                # NGL=30 default (36 OOMs at 32K)
mamba run -n llamacpp-cuda vendor/llama.cpp/build/bin/llama-bench \
  -m models/qwen38-27b-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  -ngl 30,34,38,42 -t 6,12 -p 0 -n 48 -fa 1 -ctk q8_0 -ctv q8_0 -r 2
```

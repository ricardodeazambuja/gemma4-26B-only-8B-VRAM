# Gemma 4 12B Coder (fable5-composer2.5) — quant evaluation for 8 GB VRAM + 31 GB RAM

Evaluation of [`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)
on this box (RTX 2070 Max-Q, **8 GB VRAM**; **31 GB** system RAM; CUDA backend built in `vendor/`).

**Status: COMPLETE** (2026-06-15). All 4 quants downloaded + tested. Recommendation below.

---

## TL;DR / recommendation

**Use `Q4_K_M` with `-ngl 42` (or 40 for headroom), 32K context, `q8_0` KV cache, CUDA + `--jinja`.**
It's the best fit for this 8 GB + 31 GB box: ~**17 tok/s** with excellent, coherent coding output,
because at Q4 only **6 of 48 layers** spill to CPU RAM. Run it with:

```bash
bash scripts/run-12b-coder.sh                 # Q4_K_M, ngl 40, 32K ctx, q8_0 KV
NGL=42 bash scripts/run-12b-coder.sh          # squeeze max speed (~17 t/s, tighter VRAM)
```

| Quant | Best `-ngl` | tok/s | Quality | Verdict |
|---|---|---|---|---|
| **Q4_K_M** | **42** | **~17** | excellent | ✅ **recommended** — the fit |
| Q6_K | 32 | ~7 | a touch higher | only if you want max fidelity & accept 2.4× slower |
| Q8_0 | 24 | ~4.5 | near-lossless | diminishing returns; 3.9× slower |
| Q2_K | 99 (fits!) | ~19 | **broken** | ❌ degenerate output — do not use |

**The trick for THIS model:** it's *dense*, so `--cpu-moe`/`NCMOE` (the repo's 26B-A4B trick) does
**nothing** here. The dense levers are `-ngl` (partial layer offload to RAM) + `q8_0` KV + Gemma's
built-in SWA (only every 6th layer does full attention, so the KV cache stays tiny — ~0.5 GB at
32K). The model card's own cheat-sheet calls 8 GB / Q4_K_M "tight ~2–4K ctx" — but that's
**VRAM-only**; with `-ngl` offload to your 31 GB RAM we comfortably ran **32K ctx at ~17 tok/s**.

⚠️ **Safety:** safe to download/run (data-only GGUF, sandboxed template, clean repo, checksums
verified — see §2). The one real caveat is the model's own card: *"not safety-aligned, reduced
refusals"* — it's a task-focused coder, so review the code it writes (you would anyway).

---

## 1. What this model is — and why the repo's headline trick does NOT apply

| | |
|---|---|
| Repo | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` |
| Architecture (GGUF metadata) | `gemma4` |
| Base model | `google/gemma-4-12B-it` |
| Params | ~11.9 B — **dense**, not MoE |
| Native context | 131072 |
| Chat template | custom, with thinking (`<\|channel>thought`) + tool-call channels (needs `--jinja`) |
| Likes / downloads | 615 / ~20k (as of 2026-06-15) |

**Key consequence:** this repo's whole reason for existing — llama.cpp's `--cpu-moe` /
`--n-cpu-moe` (`NCMOE`) flag, which pins the 26B-A4B's **expert** FFN weights to RAM while keeping
attention + KV on the GPU — **does not apply to a dense model.** There are no experts to offload.

For a *dense* 12B the levers that fit it into 8 GB VRAM are different:

1. **`-ngl N` — partial layer offload.** Put the first N transformer layers on the GPU and run the
   rest on the CPU (in RAM). This is the primary VRAM lever for a dense model. (`run-server.sh`
   hardwires `-ngl 99` = all layers on GPU, which is correct for the MoE-on-RAM design but will OOM
   a dense 12B that doesn't fully fit — see §4 for the dense launch recipe.)
2. **KV-cache quantization** (`KVQUANT=q8_0`) — shrinks the per-token context cost in VRAM, freeing
   room for more layers. Same lever as the MoE path; near-lossless at q8_0.
3. **Context sizing** (`-c`) — smaller context = smaller KV cache = more room for weights.
4. **Flash attention** (`-fa on`) — required for V-cache quant; also saves KV memory.

The available quants:

| Quant | File size | Fits fully in 8 GB VRAM? (weights only) |
|---|---|---|
| Q2_K | 4.83 GB | yes, with room for KV + context |
| Q4_K_M | 7.38 GB | no — needs partial `-ngl` offload |
| Q6_K | 9.79 GB | no — substantial RAM offload |
| Q8_0 | 12.67 GB | no — mostly RAM (near-lossless) |

All four fit in 31 GB RAM with room to spare.

---

## 2. Safety assessment

**Verdict: safe to download and run, with one ordinary caveat (review generated code).**

The user explicitly asked to confirm the model is safe. Findings:

- **File format is data-only.** GGUF is a tensor/metadata container parsed by llama.cpp. Unlike
  PyTorch `.bin`/`.pt`/`.ckpt` (Python *pickle* → arbitrary code execution on load), **loading a
  GGUF does not deserialize or execute code.** The classic "malicious weights run code when you load
  them" supply-chain vector does not apply to GGUF.
- **The repo contains only data.** Full file listing: 4× `.gguf`, `README.md`, `.gitattributes`.
  **No `.py`, `.sh`, no pickle, no executable** — nothing that runs on your machine.
- **The chat template is sandboxed.** The embedded Jinja template is rendered by llama.cpp's *minja*
  engine, a restricted template evaluator with no filesystem/network/`os` access. Reviewed the
  template: it is elaborate (tool-call + thinking-channel formatting) but does only string
  manipulation — no injection path to the host.
- **Checksums verified on download.** `hf download` validates each file's SHA-256 against the
  repo's LFS metadata; a corrupted/tampered transfer fails the download.
- **Parser-CVE exposure is low.** Early-2024 llama.cpp had heap-overflow CVEs in GGUF metadata
  parsing; those are mitigated by a recent build (this box's `llama-server` is a Jun-2026 build).
- **Uploader signal is positive-ish.** Public, non-gated, `license: gemma`, 615 likes / ~20k
  downloads. Not a guarantee, but not a throwaway account.

**Residual caveat (applies to *any* model):** a coder fine-tune could emit subtly insecure or
backdoored code suggestions. This is not detectable from the file — mitigate the normal way: read
the code it produces before running it. Nothing about *this* model's packaging raises that risk
above baseline.

---

## 3. Method

- Backend: locally-built **CUDA** binary (`vendor/llama.cpp/build/bin/`), per `.gemma4-menu`.
- Held constant across quants: **context = 32768**, **KV cache = q8_0**, **flash-attn on**.
- For each quant: find the **max `-ngl`** (layers on GPU) that loads inside 8 GB at that context,
  then measure prompt (pp) and generation (tg) tok/s with `llama-bench`, and record VRAM headroom.
- Coherence check: a real coding prompt through `llama-server` on the leading candidates (speed
  alone can't pick a winner — Q2_K may be fast but too degraded to code with).
- Hang-safety (running unattended): servers are backgrounded with timeouts; VRAM is verified freed
  (`nvidia-smi`) between quants; slow quants (Q6_K, Q8_0) run last.

---

## 4. Dense launch recipe (the deliverable — `models/` is gitignored)

A convenience launcher is committed at **`scripts/run-12b-coder.sh`** (dense-correct flags, env
knobs, CUDA/Vulkan auto-select, refuses Q2_K). Download a quant and run:

```bash
# one-time: fetch the recommended quant (checksum-verified by hf)
mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF \
    gemma4-coding-Q4_K_M.gguf --local-dir models/gemma4-12b-coder

bash scripts/run-12b-coder.sh                 # Q4_K_M, ngl 40, 32K ctx, q8_0 KV, :8080
NGL=42 bash scripts/run-12b-coder.sh          # max speed (~17 t/s, ~0.4 GB headroom)
TEMP=0 bash scripts/run-12b-coder.sh          # deterministic coding
CTX=65536 KVQUANT=q4_0 NGL=38 bash scripts/run-12b-coder.sh   # longer context
QUANT=Q6_K bash scripts/run-12b-coder.sh      # higher fidelity, ~7 t/s
```

Equivalent raw command (what the script runs), for reference:

```bash
mamba run --no-capture-output -n llamacpp-cuda vendor/llama.cpp/build/bin/llama-server \
  -m models/gemma4-12b-coder/gemma4-coding-Q4_K_M.gguf --alias gemma-4-12b-coder \
  --device CUDA0 -ngl 42 --no-mmap -c 32768 \
  -ctk q8_0 -ctv q8_0 -fa on --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 --host 127.0.0.1 --port 8080
```

Notes:
- **`--jinja` is required** — the model ships a custom thinking/tool-call chat template. The
  standard OpenAI `/v1/chat/completions` path works and cleanly splits thinking into
  `reasoning_content` and the answer into `content`. (`pi` uses exactly this path.)
- **Do not** pass `--cpu-moe`/`-ngl 99` from the 26B `run-server.sh` — the first OOMs/garbles, the
  second is a no-op here (no experts). This is why there's a separate launcher.
- Known llama.cpp roughness: the server's `peg-gemma4` output parser can throw on *malformed*
  channel output (only observed with the broken Q2_K). With Q4_K_M+ it parses fine.
- To point `pi` at it, register a provider on `http://127.0.0.1:8080/v1` (alias `gemma-4-12b-coder`)
  the same way `scripts/configure-pi.sh` does for the 26B.

---

## Results

_(appended incrementally as each quant completes, so a dropped session keeps finished work)_

| Quant | Size | Max `-ngl` @ 32k/q8_0 | VRAM used | pp tok/s | tg tok/s | Coherent? |
|---|---|---|---|---|---|---|
| Q2_K | 4.83 GB | 99 (fits fully) | 6.3 GB | 103 | 19.5 | **NO — degenerate/unusable** |
| **Q4_K_M** | 7.38 GB | **42** (44+ OOM) | 7.67 GB | 66.9 | **17.4** | **YES — excellent** |
| Q6_K | 9.79 GB | 32 (34+ OOM) | 7.78 GB | 30.0 | 7.2 | YES |
| Q8_0 | 12.67 GB | 24 (26+ OOM) | 7.75 GB | 12.9 | 4.5 | YES |

All numbers: CUDA backend, **ctx 32768, q8_0 KV, flash-attn on**, greedy gen of 200 tokens; VRAM
"used" includes the ~0.5 GB desktop. Q4_K_M/Q6_K/Q8_0 verified with no download contention. The
max `-ngl` per quant leaves only ~0.3–0.5 GB headroom — the per-quant *safe* defaults in
`run-12b-coder.sh` (40 / 30 / 22) back off a couple layers so a VRAM spike from the desktop won't
OOM an unattended server.

**Quality spot-check (Q4_K_M, greedy):** correct `is_prime` (edge cases + sqrt bound + even-skip),
correct one-liners, and a correct **O(1) LRU cache** (hashmap + doubly-linked list with dummy
head/tail nodes) with sound reasoning in the thinking channel. Genuinely usable for Python coding.
Note: it's a *thinking* model — give it generous `max_tokens` (the thinking goes to
`reasoning_content`; with a tiny budget `content` comes back empty, the same gotcha as the 26B).

### Diagnostics log (the hard part)

The model **loads** as arch `gemma4` and runs at full speed, but **Q2_K output is degenerate**
(greedy/temp-0 yields `ed.py.y…<channel|><channel|>…` / CJK noise). Ruled out, step by step:

- **Not the chat template / parser.** Raw `/completion` (no template, no parser) at temp 0 is also
  garbage. (Separately, the server's `peg-gemma4` output parser *crashes* on this model's
  `<\|channel>` output — a real llama.cpp robustness bug, but downstream of the garbage, not its
  cause.)
- **Not the vendor CUDA build, not CUDA, not flash-attn.** The conda-forge build (different commit
  `3c585b3`) on **CPU** garbles identically. Two independent builds, same failure → points at the
  file/config, not one binary.
- **Not a missing arch update.** `LLM_ARCH_GEMMA4` exists and `gemma4.cpp` even has a
  `gemma4_unified` logits-bias path. And **zero** model-execution commits exist between our build
  (88a3927, Jun 12) and today's master — so rebuilding to latest would not change model output.
- Load warnings are benign (token-type overrides for `</s>`, `<\|tool_response>`).

**RESOLVED — two independent causes, the model itself is fine:**

1. **Q2_K is genuinely over-degraded / unusable.** Tested three ways — raw `/completion`, jinja
   `/v1/chat/completions` (temp 0.7), and the native `<\|turn>` format — all degenerate (`l. l. l.`,
   CJK noise, or rambling to the token limit with empty content). Q2_K is the most aggressive quant
   of a 12B with unusual attention dims; it collapses. The card only *recommends* Q4_K_M and lists
   Q2_K as "tiniest, runs almost anywhere" with no quality promise. **Skip Q2_K on this model.**

2. **Q4_K_M (and up) work perfectly — but the prompt must be chat-formatted.** Q4_K_M via the
   standard `/v1/chat/completions` + `--jinja` path (exactly what `pi` uses) produces clean,
   correct code with the thinking split into `reasoning_content` and the answer in `content`, **no
   parser crash.** Feeding a *raw* unstructured prompt (no `<\|turn>` chat structure) drives even
   Q4_K_M out of distribution into garbage — that is expected for an instruct/thinking model and is
   not a defect. `enable_thinking` defaults to thinking-on through the OpenAI path; no special kwarg
   needed.

**Net:** the model is good. Use **Q4_K_M or larger**, always through the chat template. The early
garbage was Q2_K + raw-prompt testing, not a llama.cpp/arch problem.

# VibeThinker-1.5B — safety check, setup, and smoke test

WeiboAI's **VibeThinker-1.5B** is a 1.5B reasoning model fine-tuned from
**Qwen2.5-Math-1.5B** (paper: *Tiny Model, Big Logic*, arXiv:2511.06221). It targets
**competition mathematics and algorithmic coding** — it is *not* a general chatbot. This
repo's Gemma plumbing reuses almost wholesale; the launcher is
[`scripts/run-vibethinker.sh`](../scripts/run-vibethinker.sh).

⚠️ **Safety:** safe to download/run (data-only GGUF from a reputable quantizer, sandboxed
ChatML template, clean repo, checksums verified — see §2). The only caveat is the ordinary
one: it's a narrow research reasoning model, not safety-tuned for general chat — review any
code/math it emits, as you would for any model.

---

## 1. What this model is — and why the repo's headline tricks do NOT apply

Verified from the GGUF metadata (`utils/inspect-gguf.sh` + a metadata dump):

| Property | Value | Consequence |
|---|---|---|
| architecture | `qwen2` | standard llama.cpp support; ChatML |
| params / layers | 1.5B / **28 dense layers** | **no `--cpu-moe`/`NCMOE`** — it isn't MoE |
| context_length | **131072** (rope freq_base 640000, no scaling) | `CTX=40960` default is safe; native 128K |
| embedding_length | 1536 | tiny — fits 8 GB VRAM many times over |
| chat_template | ChatML, **no `<think>` tags** | CoT is **inline in `content`** |
| default system prompt | *"Please reason step by step, and put your final answer within `\boxed{}`."* | answer arrives in `\boxed{}` |
| modality | text only | **no mmproj / `--image`** |
| MTP head | none | **no speculative decoding** |

So the only Gemma lever that carries over is the dense `-ngl` one — and here it's trivial:
**`-ngl 99` puts all 28 layers on the GPU** (measured **3.7 GB VRAM at 40 K context**), so there
is no offload tuning to do at all.

**Output-format gotcha (differs from Gemma):** Gemma 4 on llama.cpp splits the chain-of-thought
into a separate `reasoning_content` field. VibeThinker does **not** — the entire response
(reasoning + the final `\boxed{}` answer) comes back in **`content`**, with no `reasoning_content`
key. A client that only reads `reasoning_content` would see nothing. Parse the `\boxed{...}` out of
`content` for the answer.

## 2. Safety assessment

**Verdict: safe to download and run, with one ordinary caveat (review generated code/math).**

- **File format is data-only.** GGUF is a tensor/metadata container parsed by llama.cpp. Unlike
  PyTorch `.bin`/`.pt`/`.ckpt` (Python *pickle* → arbitrary code execution on load), **loading a
  GGUF does not deserialize or execute code.** The classic "malicious weights run code when you
  load them" supply-chain vector does not apply to GGUF.
- **The source repo contains only data.** `mradermacher/VibeThinker-1.5B-GGUF` full non-GGUF file
  listing (HF API): `.gitattributes`, `README.md`. **No `.py`, `.sh`, no pickle, no executable** —
  nothing that runs on your machine. License `mit`, non-gated.
- **Reputable uploader.** `mradermacher` is a well-known community quantizer with thousands of GGUF
  repos. This one: non-gated, MIT, 432 downloads / 29 likes. The upstream `WeiboAI/VibeThinker-1.5B`
  is also MIT and non-gated.
- **The chat template is sandboxed.** The embedded ChatML/Qwen2.5 template is rendered by llama.cpp's
  *minja* engine — a restricted evaluator with no filesystem/network/`os` access. Reviewed it: it
  does ChatML role framing plus optional `<tool_call>` JSON formatting — only string manipulation,
  no injection path to the host.
- **Checksums verified on download.** `hf download` validates each file's SHA-256 against the repo's
  LFS metadata; a corrupted/tampered transfer fails the download.
- **Parser-CVE exposure is low.** Early-2024 llama.cpp had heap-overflow CVEs in GGUF metadata
  parsing; those are mitigated by a recent build (this box's `llama-server` is a 2026 build).

**Residual caveat (applies to *any* model):** VibeThinker is a research reasoning model focused on
math/code, explicitly *not* tuned for general-assistant safety. It can also emit subtly wrong math
or insecure code — not detectable from the file. Mitigate the normal way: check the answers/code it
produces. Nothing about *this* model's packaging raises risk above baseline.

## 3. Launch recipe (the deliverable — `models/` is gitignored)

```bash
# one-time download (Q8_0 ≈ 1.8 GB, near-lossless; f16 ≈ 3 GB if you want zero quant)
mamba run -n llamacpp hf download mradermacher/VibeThinker-1.5B-GGUF \
  VibeThinker-1.5B.Q8_0.gguf --local-dir models/vibethinker-1.5b

# serve — all on GPU, ChatML via --jinja, the card's sampling, big token budget for long CoT
bash scripts/run-vibethinker.sh                    # Q8_0, -ngl 99, 40k ctx, temp 0.6
TEMP=1.0 bash scripts/run-vibethinker.sh           # the card's other recommended temp
QUANT=f16 bash scripts/run-vibethinker.sh          # truly lossless
CTX=131072 KVQUANT=q8_0 bash scripts/run-vibethinker.sh   # full native context
```

Sampling defaults follow the model card: **temp 0.6 (or 1.0), top_p 0.95, top-k disabled.** The
devs warn against temp 0 — greedy collapses the path exploration the RL training relies on.

## 4. Smoke test (verified on this box, RTX 2070 Max-Q 8 GB)

| Check | Result |
|---|---|
| Load | model + 40 K KV cache loads, all 28 layers on GPU, **3.7 GB VRAM**, ready in ~2 s |
| Speed | **~100–103 tok/s** decode at Q8_0 (fully GPU-resident) |
| `7^2025 mod 100` | `\boxed{7}` ✓ correct; clean `finish_reason: stop`; cross-checked with Euler's theorem in-CoT |
| `least n with n! ÷ 2024` | converges on **23** ✓ correct (= largest prime factor of 2024) |
| Output format | response in `content` only, **no `reasoning_content`**; CoT inline, answer in `\boxed{}` |

**Token budget matters.** The "simple" `7^2025 mod 100` problem used **4809 completion tokens** of
chain-of-thought before boxing the answer; a first run capped at 4000 tokens got truncated mid-reasoning
(`finish_reason: length`) and never reached `\boxed{}`. The launcher therefore defaults to
`NPREDICT=38912` — keep generation budgets large.

## 5. Using it with pi

pi speaks the OpenAI API to the same `llama-server`, so it drives VibeThinker too. A
`vibethinker-1.5b` model entry is registered under the existing `llamacpp` provider
(`config/pi-provider.json` → `scripts/configure-pi.sh`).

Easiest — one command starts the server, waits for it, launches pi, and offers to stop
the server when pi exits (a minimal sibling of `start.sh`):

```bash
bash scripts/configure-pi.sh                          # once, registers the pi model
bash scripts/start-vibethinker.sh                     # server + pi, then "Stop the server? [y/N]"
bash scripts/start-vibethinker.sh -p "Write a ..."    # one-shot prompt
STOP_ON_EXIT=1 bash scripts/start-vibethinker.sh      # auto-stop the server on exit
```

It reuses a server already on the port (and leaves that one running); only a server it
started itself is offered for shutdown. Or drive the pieces by hand:

```bash
bash scripts/run-vibethinker.sh                                   # terminal 1: the server
MODEL_ID=vibethinker-1.5b bash scripts/run-pi.sh                  # terminal 2: interactive
pi --provider llamacpp --model vibethinker-1.5b -p "Write a Python function ..."  # one-shot
```

**Verified (one-shot):** asked for an O(n log n) `longest_increasing_subsequence`, it produced
a correct patience-sorting solution (`bisect_left`, strict-increasing) and self-checked it on
three examples in its reasoning.

**Caveats — it's a reasoner, not an agent.**
- **Best for self-contained problems** ("write a function / solve this"), *not* multi-file agentic
  work. It reasons and emits code; it does not reliably drive pi's read/edit/bash tool loop.
- **It dumps a long chain-of-thought first.** Under pi's coding system prompt it wraps the CoT in an
  inline `<think>…</think>` block, then the answer. (On bare math prompts with the model's default
  system prompt it instead reasons in plain text and boxes the answer — the format follows the
  prompt.) The provider entry sets `reasoning: false`, so pi shows that block inline rather than
  folding it. (To fold it as a thinking section, serve with `--reasoning-format deepseek` and flip
  the entry to `reasoning: true`.)
- Keep generation budget large (the launcher's `NPREDICT=38912`) — it thinks a lot.

## 6. Method
- Server backgrounded with a health-poll wait; VRAM verified freed (`nvidia-smi`) after the run.
- Known-answer problems chosen so the `\boxed{}` result is checkable against a Python computation
  (`23` = least n with n!÷2024; `7` = 7^2025 mod 100).

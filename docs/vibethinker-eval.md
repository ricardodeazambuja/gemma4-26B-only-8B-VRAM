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

## 6. Letting the model test its own code (`scripts/selfcheck.py`)

**It can't drive pi's tools.** Verified directly: with a `tools` payload it emits **no** tool
calls — not under `tool_choice:"auto"`, not under `"required"`, not even with `--reasoning-budget 0`.
It always slides into a `<think>` block, reasons, and boxes an answer. (Root cause: llama.cpp
doesn't recognise this Qwen template's `<think>` as a reasoning channel — `reasoning_content` stays
empty and the reasoning-budget lever never engages — so the tool-call grammar is never reached.)
So an agentic "model calls a run-code tool" loop is a dead end for this model.

**Instead, close the loop outside the model.** `selfcheck.py` is a Reflexion-style harness — *it*
runs the code and feeds failures back:

```
ask -> extract the ```python -> RUN it against a verifier -> feed the concrete
failure (smallest counterexample) back -> repeat up to N rounds
```

```bash
python scripts/selfcheck.py scripts/selfcheck_problems/distinct_substrings.py --rounds 3
```

A problem module defines `TASK`, `ENTRY`, and `verify(ns) -> (ok, feedback)`. Candidates run in a
subprocess under `RLIMIT_AS` (an O(n²) attempt OOMs cleanly into feedback instead of crashing the
box) plus a wall-clock timeout. Token budget matters a lot: `VT_MAXTOK=16000` truncated every round
mid-`<think>` before the model reached its final code fence — use `≥32000` and give the server CTX
headroom (`CTX=65536`) for the growing multi-round prompt.

**Result on the suffix-automaton task (honest):** the loop works end-to-end — the model finishes
(`finish_reason=stop`), emits code, the harness catches every broken version with a counterexample,
and the model revises in response. But it did **not converge** in 3 rounds: `banana` → 18, then 20,
then 18 (correct 15). It oscillates near the answer without localising the bug. Takeaway: a 1.5B
reasoner is a strong **draft generator** but a weak **self-debugger** on hard data-structure code —
one I/O counterexample isn't enough signal to pin a suffix-automaton bug. (Richer feedback — multiple
counterexamples, or which substrings are mis-counted — might help; not yet tried.)

## 7. VibeThinker-3B (Qwen2.5-Coder-3B)

WeiboAI also shipped a **3B** model, built on **Qwen2.5-Coder-3B** and post-trained for
competitive math, coding, STEM, and *instruction-following with explicit constraints*. Same
plumbing — the launcher and pi wiring are size-parameterised:

```bash
mamba run -n llamacpp hf download mradermacher/VibeThinker-3B-GGUF \
  VibeThinker-3B.Q8_0.gguf --local-dir models/vibethinker-3b      # ~3.3 GB, same safety profile as §2
SIZE=3B bash scripts/run-vibethinker.sh                            # serve it (alias vibethinker-3b)
SIZE=3B bash scripts/start-vibethinker.sh                          # server + pi, all-in-one
pi --provider llamacpp --model vibethinker-3b -p "..."             # or drive pi directly
```

Verified metadata: arch `qwen2`, **36 layers**, native context 131072 (rope base 1e6, no
scaling), Q8_0 ≈ 3.3 GB, **5.4 GB VRAM** at 40k ctx (`-ngl 99`, all on GPU). Default system
prompt is the generic *"You are a helpful assistant."* (the 1.5B's is the math `\boxed{}` nudge),
so the 3B reads more like a normal assistant.

**Result — markedly stronger at coding.** Given the suffix-automaton task the 1.5B failed (buggy
`last` update) and then could **not** self-correct over 3 selfcheck rounds, the **3B solved it
correctly in ONE shot through pi**: a clean O(n) suffix automaton that passed all 509 oracle cases
and the 100000-char scale check (0.15 s). This matches the model card's coding focus
(LiveCodeBench / LeetCode).

**Tool use — still none.** Re-probed the 3B with a `run_python` tool: no `tool_calls` under
`tool_choice` auto *or* required (same `<think>`-and-answer behaviour as the 1.5B). So in pi the
3B is a much better one-shot coder, but still **not** a tool-driving agent — the §6 external
verify-retry loop remains the way to "let it test its own code" (and it now passes round 1).

## 8. Making llama.cpp recognise the `<think>` reasoning (the template fix)

By default llama.cpp does **not** treat this model's `<think>…</think>` as a reasoning channel:
`reasoning_content` stays empty, the CoT sits inline in `content`, and `--reasoning-budget` does
nothing. Root cause, traced in this build's source (`common/chat.cpp` → the auto-parser in
`common/chat-diff-analyzer.cpp:423`): reasoning detection is **template-driven**. The analyzer
renders the template with a probe `reasoning_content` field and diffs to find the wrapping tags
(`compare_reasoning_presence`). VibeThinker's stock Qwen2.5 template **never renders
`reasoning_content`**, so the diff finds nothing → `reasoning.mode = NONE` → the `<think>` the model
emits at inference passes through as content. The server logs this as `chat template, thinking = 0`.

**Fix (no rebuild): a patched chat template**, in [`config/vibethinker-1.5b-think.jinja`](../config/vibethinker-1.5b-think.jinja)
and [`config/vibethinker-3b-think.jinja`](../config/vibethinker-3b-think.jinja) — the stock template plus
two edits: (1) the assistant branch renders `reasoning_content` wrapped in `<think>…</think>` (so the
analyzer detects the tags), and (2) the generation prompt force-opens `<think>` (the model continues
inside it). `run-vibethinker.sh` serves it by default (`THINK=1`) via `--chat-template-file`.

**Verified on the 3B** (`thinking = 1` after the fix):

| | stock template (`THINK=0`) | patched template (`THINK=1`) |
|---|---|---|
| `reasoning_content` | empty | **1441 chars** (the CoT) |
| `content` for "17*23" | CoT wall + answer | clean **`391`** |
| `--reasoning-budget 500` | ignored | **honoured** — 505 tokens total, thinking capped, answer still emitted |

This fixes the two real pain points: pi now folds the CoT (provider entries flipped to
`reasoning: true`), and **`REASONING_BUDGET` bounds the runaway thinking** that truncated the
selfcheck loop (§6). What it does *not* fix is tool use — the OpenAI `tools`/`tool_choice` path still
yields no `tool_calls` (§7); that's a model-training limit, not a parser one. (Curiosity: under pi's
*text-based* tool protocol the 3B did emit a `<function>` bash call — worth a deeper look, but not the
rigorous OpenAI path.)

Alternative not taken: a `--reasoning-tags "<think>,</think>"` **source patch** near
`chat.cpp:2392` to force the tags without a per-model template (no such CLI flag exists in this
build). The template route needs no rebuild, so it wins for now.

## 9. Making VibeThinker call tools (the recovery shim)

§6 found VibeThinker won't drive tools natively. A closer look shows *why*, and that it's
recoverable. The model **reliably produces correct tool intent** — the right tool name and valid
JSON arguments — but wraps it in a hallucinated tag (`<fulfil>`, `<function call>`, `<xml>`, …)
instead of the `<tool_call>` llama.cpp greps for, and labels the name with either `"name"` (OpenAI
tools framing) or `"function"` (pi's framing). So llama.cpp never populates `tool_calls` and clients
see text. Prompt-anchoring to the canonical `<tool_call>` form does **not** fix emission (0/5 — the
RL-baked dialect ignores the instruction), so the fix has to be downstream.

**Two things make it work:**
1. **Reasoning budget** (`REASONING_BUDGET`, the §8 lever). On a genuinely tool-requiring task
   (e.g. a hash it can't fake), *without* a budget the model ruminates past the token limit and
   emits **nothing**. *With* a budget it stops thinking and emits the call — measured **5/6** on
   sha256/md5 tasks.
2. **[`scripts/vibethinker-toolshim.py`](../scripts/vibethinker-toolshim.py)** — an OpenAI-compatible
   recovery proxy (`client → shim:8088 → llama-server:8080`). When a response has no native
   `tool_calls` but its content carries a recoverable `{name|function, arguments}` object whose name
   is one of the *declared* tools, it rewrites the response into proper `tool_calls`. Invented names
   (the model sometimes makes one up) are rejected. Works for streaming and non-streaming; misses are
   logged to stderr.

One command brings up the whole chain (server with budget + shim + pi routed through it,
all cleaned up on exit):

```bash
TOOLS=1 SIZE=3B bash scripts/start-vibethinker.sh                 # interactive
TOOLS=1 SIZE=3B bash scripts/start-vibethinker.sh -p "<task>"     # one-shot
```

Or by hand:

```bash
REASONING_BUDGET=600 SIZE=3B bash scripts/run-vibethinker.sh   # server WITH a budget (required)
python scripts/vibethinker-toolshim.py                          # shim on :8088
# register a pi provider 'vtshim' with baseUrl http://127.0.0.1:8088/v1, then:
pi --provider vtshim --model vibethinker-3b "<task that needs a tool>"
```

**Verified:**
- Recovery into `tool_calls`: non-streaming and streaming (SSE carries the `tool_calls` delta +
  `finish_reason: tool_calls`), confirmed via curl and the **OpenAI Python SDK** (the library pi uses).
- **Full multi-turn loop closes**, streaming, on sha256 (a task the model cannot fake): the shim
  recovers the `run_python` call → it executes → the result is fed back → the model returns the
  **correct digest** (`STREAMING MULTI-TURN LOOP CLOSED: True`).
- **pi end-to-end (one command).** `TOOLS=1 SIZE=3B bash scripts/start-vibethinker.sh -p "Use the
  bash tool to compute 7**13 …"` brought up server+shim+pi, pi executed the shim-recovered `bash`
  call (`echo $((7**13))`), and returned the correct **96889010407** — then cleaned up. The model
  calls pi's real tools (`bash`, `plan_set`) correctly; the shim recovers them.

**Honest ceiling.** This is ~80%, not 100%: ~1/6 of the time on tool-requiring tasks the model emits
nothing recoverable or goes fully off-format (nested XML, invented tool name), and it can wrap the
final answer in junk (e.g. a stray `<html>` around the right number). It depends on the reasoning
budget being set. The shim makes VibeThinker a *usable* tool-caller, not a reliable agent — pair it
with verification, and prefer a tool-trained model for heavy agentic work.

## 10. Method
- Server backgrounded with a health-poll wait; VRAM verified freed (`nvidia-smi`) after the run.
- Reasoning fix verified by diffing `reasoning_content`/`content` and the `thinking = 0|1` server log
  with vs without `--chat-template-file`, and confirming `--reasoning-budget` caps completion tokens.
- Known-answer problems chosen so the `\boxed{}` result is checkable against a Python computation
  (`23` = least n with n!÷2024; `7` = 7^2025 mod 100).
- Tool-use probed with a `run_python` tool across `tool_choice` auto/required and `--reasoning-budget 0`.
- Self-check verifier checks against a brute-force `set()` oracle (≈500 random + hand-picked strings)
  plus a 100000-char scale/timing check.

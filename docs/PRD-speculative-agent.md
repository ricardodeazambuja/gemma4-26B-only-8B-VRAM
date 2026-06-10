# PRD — Speculative Execution & Branch Prediction for the Gemma↔Opus agent

> **Living document.** This is both the spec *and* the progress tracker. Update the
> **Status board** (§7) and the **Progress log** (§9) as work happens. The first thing to do
> when resuming is: read the Status board, find the first `TODO`/`DOING`/`BLOCKED` item, and
> continue from there. Keep this file truthful — a half-done task is `DOING`, not `DONE`.

| | |
|---|---|
| **Branch** | `feat/spec-exec-branch-prediction` |
| **Owner** | ricardodeazambuja |
| **Status** | 🟢 Building — M0–M3 + MI(images) done, M4 next |
| **Last updated** | 2026-06-10 |
| **Pending live tests** | Need `llama-server` up: predict.sh latency (R1), image OCR (MI), background predict+draft hit (M3). |
| **Host surface** | Claude Code (hooks → plugin) |
| **Backends** | local `llama-server` (Gemma 4 26B-A4B QAT, :8080) + Claude Code / Opus |

---

## 1. Summary

Build a **two-tier speculative agent** inside Claude Code that mirrors CPU speculative
execution and branch prediction:

- **Gemma 4 (local, cheap, fast)** acts as the **draft / branch-predictor tier**. It always
  goes *first*: it drafts an answer, classifies difficulty, and — between turns — speculatively
  pre-computes the *likely next step*.
- **Opus / Claude Code (the "big")** acts as the **verifier / target tier**. It either
  rubber-stamps Gemma's draft (cheap "accept") or supersedes it (a "misprediction" → it does the
  real work).
- After the big model produces output, the **local agent helps**: it reviews outputs,
  pre-fetches, or prepares the next branch.

The win is the same as token-level speculative decoding, lifted to the *agent-turn* level:
turn an expensive Opus **generate** into a cheap Opus **accept/reject**.

## 2. The mapping (why this is genuinely "speculative execution")

| CPU / speculative-decoding concept | This system |
|---|---|
| Draft model (cheap, guesses ahead) | **Gemma 4** on llama-server (:8080) |
| Target model (authoritative, verifies) | **Opus / Claude Code** |
| Speculate ahead of the critical path | Gemma pre-computes the likely next step in the **background** |
| Branch predictor | Gemma predicting *which* request comes next |
| Misprediction flush | Wrong guess → discard cached speculative result |
| "Accept the draft" (the win) | Opus does a cheap **verify** instead of a full **generate** |
| Predictor accuracy | **Hit rate** logged to `stats.jsonl`, viewed via `/spec-stats` |
| Cheap pre-processing before the expensive unit | **Image → text** by Gemma so Opus pays no image tokens (G6) |

## 3. Goals / Non-goals

### Goals
- G1. Gemma *always runs first* on each prompt, automatically, without Opus having to choose to delegate.
- G2. Speculative work happens **off the critical path** (background) so it never blocks the user.
- G3. Mispredictions are cheap and safe: discard-on-miss, never auto-apply destructive work.
- G4. The branch-predictor **hit rate is measurable** and visible (`/spec-stats`).
- G5. Ships first as raw hooks (fast iteration), then graduates to a distributable **plugin**.
- **G6. (MUST) Multimodal image offload.** Gemma 4 is multimodal; any image is OCR'd/described by
  the *local* tier and only the resulting **text** reaches Opus, so Opus never spends image tokens.
  Enforced automatically; degrades safely (Gemma down → normal read, image not stranded).

### Non-goals
- N1. Replacing Opus from a hook (impossible — hooks feed/​gate the turn, they don't pre-empt it).
- N2. Auto-applying speculative file writes. Speculation is read-only / discardable only.
- N3. Token-level llama.cpp `--model-draft` speculative decoding — *orthogonal bonus*, tracked
  separately in §8, not required for v1.

## 4. Why hooks, not a skill (the packaging decision)

A **skill** only runs *after* Opus decides to invoke it — by then an Opus turn is already paid
for, which violates G1 ("start with A"). The only Claude Code surface that runs **automatically,
in the harness, around the big model** is the **hook system**. Therefore:

- **Mechanism** = hooks (`UserPromptSubmit`, `Stop`, optional `PostToolUse`).
- **Distribution** = a **plugin** wrapping those hooks + a `/spec-stats` command + an optional
  *deliberate* "ask-Gemma-first" skill.
- A skill alone is insufficient; a plugin without hooks is insufficient.

## 5. Architecture

### 5.1 Hooks
- **`UserPromptSubmit` → `predict.sh`** (synchronous, must be *fast*): the "start with A" entry.
  1. Hash the prompt, check `cache/` for a warm speculative hit.
  2. Tiny Gemma call: classify `easy|hard` + (on cache miss) a short draft.
  3. Inject Gemma's draft/answer + cache status into Opus's context via stdout.
  Opus then **verifies** (cheap accept) or **supersedes** (misprediction).
  ⚠️ This call is on the critical path — keep it to a few tokens.
- **`Stop` → `speculate.sh`** (background/detached): the speculative-execution + branch-prediction
  engine. When Opus finishes a turn, predict the *next* user request and pre-compute a **read-only**
  result into `cache/`, keyed by predicted-prompt hash. Heavy work lives here, off the critical path.
- **`PostToolUse` → `review.sh`** (optional): after Opus writes/runs, Gemma reviews/summarizes/flags
  ("check the outputs from the big model and help").

### 5.2 File layout (prototype stage)
```
.claude/
  settings.local.json     # registers the hooks
  spec/
    gemma.sh              # curl -> http://localhost:8080/v1/chat/completions
    predict.sh           # UserPromptSubmit: cache-lookup + cheap classify -> inject
    speculate.sh         # Stop: predict next prompt, pre-compute in BACKGROUND -> cache/
    review.sh            # PostToolUse (optional)
    lib.sh               # hashing, cache paths, stats logging helpers
    cache/               # warm speculative results, keyed by prompt hash  (gitignored)
    stats.jsonl          # hit/miss log -> predictor accuracy               (gitignored)
```

### 5.3 Plugin layout (graduation stage)
```
gemma-spec-plugin/
  .claude-plugin/plugin.json
  hooks/hooks.json
  commands/spec-stats.md        # /spec-stats  -> hit rate, last predictions
  skills/gemma-draft/SKILL.md   # deliberate "ask Gemma first"
  scripts/{gemma,predict,speculate,review,lib}.sh
```

## 6. Constraints, risks & open questions

- **R1 — Critical-path latency.** Gemma ~23 tok/s. The synchronous `UserPromptSubmit` call must be
  tiny (classify + cache lookup). Mitigation: all heavy drafting → background `Stop` job. *(open: budget a ms target)*
- **R2 — Opus can't be pre-empted.** Hooks feed/​gate, never replace the turn. Accept this; savings
  come from cheap *verify* turns, not from skipping Opus.
- **R3 — Cache safety / staleness.** Speculate only read-only, discardable work. Flush on miss.
  Never auto-apply writes. *(open: cache key — prompt hash only, or hash+cwd+git-HEAD?)*
- **R4 — Server availability.** Hooks must degrade gracefully when `llama-server` is down (no draft,
  no error spam — just fall through to normal Opus).
- **R5 — Image offload must never strand an image (G6).** If Gemma is down, the encode fails, or the
  file isn't an image, the `PreToolUse(Read)` hook must **allow** the normal read. Only intercept when
  Gemma can actually return text. *(open: also offload tool-produced images, e.g. screenshots, via
  `PostToolUse` — deferred past v1.)*
- **Q1** — Should `predict.sh` ever short-circuit trivial prompts entirely (Gemma answers, Opus skipped)?
  Default v1: no (N1). Revisit after measuring.
- **Q2** — How to measure "accept vs misprediction" objectively without a human label? *(open)*

## 7. Status board

Legend: `TODO` · `DOING` · `DONE` · `BLOCKED` · `DROPPED`

### Milestone M0 — Scaffolding & branch  ✅ DONE
- [x] `DONE` Create branch `feat/spec-exec-branch-prediction`
- [x] `DONE` Write this PRD and commit as the living tracker

### Milestone M1 — Gemma client & graceful degradation  ✅ DONE
- [x] `DONE` `gemma.sh`: curl wrapper to `/v1/chat/completions`, configurable host/port/model
- [x] `DONE` Health check + graceful no-op when server down (R4) — exits 3 silently
- [x] `DONE` `lib.sh`: prompt hashing, cache paths, `stats.jsonl` append helpers
- [x] `DONE` `.gitignore`: ignore `cache/` and `stats.jsonl` (moved up from M5)

### Milestone M2 — Synchronous path (the "start with A" entry)  ✅ DONE
- [x] `DONE` `predict.sh`: cache lookup → cheap classify/draft → inject via stdout
- [x] `DONE` Register `UserPromptSubmit` hook (in committed `.claude/settings.json`, not local — repo is the demo)
- [x] `DONE` Verify hit / miss_offline / empty-prompt paths; latency stamped to stats (`ms`) for R1
  - Note: live latency measurement vs a running server is pending until the server is up (carried in M3 test).

### Milestone MI — Multimodal image offload (MUST · G6)  ✅ DONE (live test pending server)
- [x] `DONE` `gemma.sh --image <path>`: base64 + OpenAI vision content → Gemma returns OCR/description
- [x] `DONE` `describe.sh` `PreToolUse(Read)` hook: image + Gemma up → **deny raw read**, hand Opus the text
- [x] `DONE` Safe degrade (R5): Gemma down / not an image / non-Read / encode fails → **allow** normal Read (verified)
- [x] `DONE` Register the `PreToolUse(Read)` hook; `image_offload` stat logs path + bytes saved
- [x] `DONE` Tested: degrade-to-allow paths + deny-JSON payload shape valid
- [ ] `TODO` LIVE test (needs server up): real image → denied + correct OCR text reaches Opus; bytes logged

### Milestone M3 — Background speculation (branch prediction)  ✅ DONE (live test pending server)
- [x] `DONE` `speculate.sh`: predicts next prompt + drafts it → `cache/` + `last_prediction.json` (detached via setsid)
- [x] `DONE` Register `Stop` hook; verified it returns instantly (~12ms) and never blocks the user
- [x] `DONE` Wire **hit** detection into `predict.sh`: lexical-Jaccard match (`SPEC_MATCH_MIN`, default 34%) → `hit_predicted`
- [x] `DONE` Tested: similarity 100/42/0, fuzzy-hit injection, no false hit on unrelated prompt
- [ ] `TODO` LIVE test (needs server up): worker actually predicts + drafts next turn; real hit on a follow-up

### Milestone M4 — Observability
- [ ] `TODO` `stats.jsonl` records predict/hit/miss/accept per turn
- [ ] `TODO` `/spec-stats` command (or script) shows hit rate + recent predictions

### Milestone M5 — Optional helper & graduation
- [ ] `TODO` `review.sh` `PostToolUse` hook (Gemma reviews Opus output)
- [ ] `TODO` Graduate hooks → `gemma-spec-plugin/` and a deliberate `gemma-draft` skill
- [ ] `TODO` Add `.gitignore` entries for `cache/` and `stats.jsonl`

### Orthogonal (not blocking v1)
- [ ] `TODO` (§8) Token-level llama.cpp `--model-draft` speculative decoding in `start.sh`

## 8. Orthogonal bonus — token-level speculative decoding

Independent of the agent harness: llama.cpp can run a small **draft GGUF** (`--model-draft`,
`--draft-max`) to speed up Gemma itself (draft+verify at the *token* level). One server-flag +
a second small GGUF. Speeds up every Gemma call above. Tracked, not required for v1.

## 9. Progress log

Newest first. One line per meaningful change; reference commits/tags.

- `2026-06-10` — M3 done. `speculate.sh` Stop hook detaches a setsid worker that predicts the next request and pre-drafts it into `cache/` + `last_prediction.json`; hook returns ~12ms (non-blocking). `predict.sh` matches the real next prompt via lexical Jaccard (`spec_similarity`, threshold `SPEC_MATCH_MIN`) → `hit_predicted` with score. Verified with mocks. Next: M4 observability.
- `2026-06-10` — MI done (multimodal image offload, MUST/G6, added per user request). `gemma.sh --image` (OpenAI vision), `describe.sh` PreToolUse(Read) hook denies image reads and hands Opus Gemma's OCR/text so Opus pays 0 image tokens. Safe-degrade (R5) verified for server-down/non-image/non-Read; deny-JSON shape verified. Live OCR test pending server. Next: M3.
- `2026-06-10` — M2 done. `predict.sh` UserPromptSubmit hook: warm-cache HIT injects draft, cache MISS does a short inline classify+draft (server up), server-down/empty → silent no-op. Registered in committed `.claude/settings.json`. All paths log to stats.jsonl with latency. Next: M3 background speculation.
- `2026-06-10` — M1 done. `lib.sh` + `gemma.sh` under `.claude/spec/`; graceful degradation verified (server down → exit 3, no noise); stats logging works. `.gitignore` updated. Next: M2 synchronous path.
- `2026-06-09` — M0 done. Branch created, PRD authored as living tracker. Status: planning → ready to build M1.

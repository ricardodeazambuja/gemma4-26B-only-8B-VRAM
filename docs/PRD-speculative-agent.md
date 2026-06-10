# PRD — Speculative Execution & Branch Prediction for the Gemma↔Opus agent

> **Living document.** This is both the spec *and* the progress tracker. Update the
> **Status board** (§7) and the **Progress log** (§9) as work happens. The first thing to do
> when resuming is: read the Status board, find the first `TODO`/`DOING`/`BLOCKED` item, and
> continue from there. Keep this file truthful — a half-done task is `DOING`, not `DONE`.

| | |
|---|---|
| **Branch** | `feat/spec-exec-branch-prediction` |
| **Owner** | ricardodeazambuja |
| **Status** | 🟢 Building — M0–M1 done, M2 next |
| **Last updated** | 2026-06-10 |
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

## 3. Goals / Non-goals

### Goals
- G1. Gemma *always runs first* on each prompt, automatically, without Opus having to choose to delegate.
- G2. Speculative work happens **off the critical path** (background) so it never blocks the user.
- G3. Mispredictions are cheap and safe: discard-on-miss, never auto-apply destructive work.
- G4. The branch-predictor **hit rate is measurable** and visible (`/spec-stats`).
- G5. Ships first as raw hooks (fast iteration), then graduates to a distributable **plugin**.

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

### Milestone M2 — Synchronous path (the "start with A" entry)
- [ ] `TODO` `predict.sh`: cache lookup → cheap classify/draft → inject via stdout
- [ ] `TODO` Register `UserPromptSubmit` hook in `settings.local.json`
- [ ] `TODO` Verify draft actually lands in Opus context; measure added latency (R1)

### Milestone M3 — Background speculation (branch prediction)
- [ ] `TODO` `speculate.sh`: predict next prompt, pre-compute read-only result → `cache/` (detached)
- [ ] `TODO` Register `Stop` hook; confirm it does not block the user
- [ ] `TODO` Wire cache **hit** detection back into `predict.sh` (close the loop)

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

- `2026-06-10` — M1 done. `lib.sh` + `gemma.sh` under `.claude/spec/`; graceful degradation verified (server down → exit 3, no noise); stats logging works. `.gitignore` updated. Next: M2 synchronous path.
- `2026-06-09` — M0 done. Branch created, PRD authored as living tracker. Status: planning → ready to build M1.

# PRD — Speculative Execution & Branch Prediction for the Gemma↔Opus agent

> **Living document.** This is both the spec *and* the progress tracker. Update the
> **Status board** (§7) and the **Progress log** (§9) as work happens. The first thing to do
> when resuming is: read the Status board, find the first `TODO`/`DOING`/`BLOCKED` item, and
> continue from there. Keep this file truthful — a half-done task is `DOING`, not `DONE`.

| | |
|---|---|
| **Branch** | `feat/spec-exec-branch-prediction` |
| **Owner** | ricardodeazambuja |
| **Status** | 🟢 v2: M0–M5, MI, MA, MO, MM, MX — all live-verified · MH (post-review hardening) done |
| **Last updated** | 2026-06-10 (overnight session) |
| **Live-verified** | Auto-start (cuda/65536/NCMOE=22/--image, 38s, non-blocking 113ms); predict draft 3.7s; image OCR exact; branch-prediction hit 85% in 186ms. |
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
  `PostToolUse` — deferred past v1.)* **MH addition:** "strand" also covers OCR that *succeeds but
  is insufficient* (dense diagram, layout matters) — now a Read with offset/limit passes the raw image
  through (same escape hatch as logs, advertised in the deny reason), and `SPEC_IMAGE_OFFLOAD=0` is a
  kill switch. Offload stays the default (G6 unchanged).
- **R6 — Reasoning must be OFF for the draft tier (solved).** Gemma 4 QAT ships with thinking on
  (`--jinja`, `thinking=1`); under small token budgets the whole budget goes to `reasoning_content` and
  `message.content` comes back EMPTY → every draft was a no-op. Fixed in `gemma.sh` by sending
  `chat_template_kwargs:{enable_thinking:false}` by default (re-enable with `--think`/`SPEC_THINK=1`),
  plus a `reasoning_content` fallback. `reasoning_effort:none` does NOT work for this template.
- **R7 — Inline draft latency (mitigated in MO).** Was ~3.7s on every prompt; now: long prompts skip
  the call (instant), hard prompts ~0.8s (no draft body), easy prompts ~2s (and those inject a usable
  answer). Lever remains `SPEC_PREDICT_MAX` / `SPEC_INLINE_MAXCHARS`.
- **R8 — Injection is a COST, not a free win (found live, fixed in MO).** Everything the hooks print
  becomes the big model's input tokens. A wrong draft (Gemma's "hard" guesses, "I can't access files"
  pleas) is worse than nothing. Rules: inject only easy drafts / verified cache hits, terse wrappers,
  drafts written for an agent with tool access, consume-once + TTL so stale content can't recur.
- **R9 — Log digests must stay lossless-reachable.** The digest keeps exact head/tail + grep'd error
  lines, and any Read with offset/limit bypasses the offload entirely — exact bytes are always one
  call away. Never extend this pattern to code/source files (quality constraint, see MX rejections).
- **R10 — Pre-executed commands are a FIXED hard-coded set (security review fix).** The first version let
  Gemma propose commands from an allowlist; that was a prompt-injection → arbitrary-file-disclosure vector
  (`head /etc/passwd` passes a naive allowlist, output lands in Claude's context). **Now the model chooses
  nothing**: the worker runs exactly `git status --short --branch`, `git diff --stat`, `git log --oneline -8`
  (only inside a git work tree), by direct exec (no shell), 5s timeout, 1200-char cap. No transcript-derived
  text ever reaches a shell, so there is no injection surface. Verified: a transcript instructing
  `head /etc/passwd` / `cat ~/.ssh/id_rsa` produced only git output, no leakage.
- **Q1** — Should `predict.sh` ever short-circuit trivial prompts entirely (Gemma answers, Opus skipped)?
  Default v1: no (N1). Revisit after measuring.
- **Q2** — How to measure "accept vs misprediction" objectively without a human label?
  *(crudely answered in MH)*: the Stop worker compares the injected draft against the assistant's
  final text with word **containment** (≥ `SPEC_ACCEPT_MIN`, default 60% → `accepted`, else
  `superseded`) and logs an `outcome` event; `/spec-stats` reports the acceptance rate. Heuristic,
  clearly labeled — a real signal would need the big model to self-report. Still open for a better
  measure.

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
- [x] `DONE` LIVE: inline draft correct (`DIFFICULTY: easy` + answer); measured **~3.7s** on the critical path (R1)

### Milestone MI — Multimodal image offload (MUST · G6)  ✅ DONE (live test pending server)
- [x] `DONE` `gemma.sh --image <path>`: base64 + OpenAI vision content → Gemma returns OCR/description
- [x] `DONE` `describe.sh` `PreToolUse(Read)` hook: image + Gemma up → **deny raw read**, hand Opus the text
- [x] `DONE` Safe degrade (R5): Gemma down / not an image / non-Read / encode fails → **allow** normal Read (verified)
- [x] `DONE` Register the `PreToolUse(Read)` hook; `image_offload` stat logs path + bytes saved
- [x] `DONE` Tested: degrade-to-allow paths + deny-JSON payload shape valid
- [x] `DONE` LIVE: PNG with text → read denied + exact OCR (`Invoice #42 Total: $128.50 / Status: PAID`) reached Opus as text

### Milestone M3 — Background speculation (branch prediction)  ✅ DONE + live-verified
- [x] `DONE` `speculate.sh`: predicts next prompt + drafts it → `cache/` + `last_prediction.json` (detached via setsid)
- [x] `DONE` Register `Stop` hook; verified it returns instantly (~12ms) and never blocks the user
- [x] `DONE` Wire **hit** detection into `predict.sh`: lexical-Jaccard match (`SPEC_MATCH_MIN`, default 34%) → `hit_predicted`
- [x] `DONE` Tested: similarity 100/42/0, fuzzy-hit injection, no false hit on unrelated prompt
- [x] `DONE` LIVE: worker predicted+drafted next turn; near-identical follow-up scored an 85% hit in 186ms (no model call)

### Milestone M4 — Observability  ✅ DONE
- [x] `DONE` `stats.jsonl` records predict/hit/miss + speculate + image_offload per turn (M1–M3)
- [x] `DONE` `stats.sh` computes overall + online branch-prediction hit rate, avg match %, image KB saved
- [x] `DONE` `/spec-stats` slash command (`.claude/commands/spec-stats.md`) runs it
- [x] `DONE` Tested on empty + synthetic stream (3/5 overall, 3/4 online, 768 KB saved)

### Milestone M5 — Optional helper & graduation  ✅ DONE (packaging deferred by decision)
- [x] `DONE` `review.sh` `PostToolUse(Write|Edit|MultiEdit)` hook (Gemma second opinion); OFF unless `SPEC_REVIEW=1`
- [x] `DONE` Deliberate `gemma-draft` skill (`.claude/skills/gemma-draft/SKILL.md`) — manual "ask Gemma first" / OCR
- [x] `DONE` `.gitignore` entries for `cache/` and `stats.jsonl` (landed in M1)
- [x] `DROPPED→DEFERRED` Repackage into a standalone `gemma-spec-plugin/`. **Decision:** the committed
  `.claude/` hooks+skill+command already provide all behavior and ship with the repo; a separate plugin
  would only duplicate the 5 scripts to gain marketplace install. Revisit if we want to distribute it.
  *Graduation steps when needed:* create `gemma-spec-plugin/.claude-plugin/plugin.json`, move
  `.claude/spec/*` → `plugin/scripts/`, translate `settings.json` hooks → `plugin/hooks/hooks.json`
  (use `${CLAUDE_PLUGIN_ROOT}`), move command + skill under the plugin, publish to a marketplace.

### Milestone MA — Auto-start the optimal server from the hooks  ✅ DONE + live-verified
- [x] `DONE` `ensure-server.sh`: single-flight (lock dir), detached (setsid) worker, **non-blocking** (hook returns instantly)
- [x] `DONE` Reuse "optimal" config — backend via `resolve_backend`, CTX/NCMOE/KV from `.gemma4-tuning` via `_tuning.sh` (no reinvention)
- [x] `DONE` Launch with `--image` when the mmproj exists, so the image offload (G6) works on the auto-started server
- [x] `DONE` Triggered fire-and-forget from `predict.sh` + `speculate.sh`; kill switch `SPEC_AUTOSTART=0`; `SPEC_AUTOSTART_DRYRUN=1`
- [x] `DONE` `mamba` PATH fallback for stripped hook env; truncates `$SPEC_SERVER_LOG` on fresh launch; logs `autostart` stat
- [x] `DONE` LIVE: a prompt with the server down auto-launched cuda/65536/NCMOE=22/--image; healthy in 38s; prompt returned in 113ms

### Milestone MO — Token-efficiency review pass  ✅ DONE + live-verified
Driven by a live failure: the hook injected a wrong "hard" draft into the big model's context —
costing tokens instead of saving them. The big tier is now **Fable 5** ("Opus" elsewhere in this
doc = "the big Claude model"); every injected token must earn its place.
- [x] `DONE` Inject inline drafts ONLY when Gemma classifies `easy` (wrong hard drafts = pure cost, R8); hard → log `miss_hard`, no injection. Side-benefit: hard path 3.7s → ~0.8s (no draft body generated)
- [x] `DONE` Length gate: prompts > `SPEC_INLINE_MAXCHARS` (240) skip the inline call entirely (`miss_skipped_long`)
- [x] `DONE` Consume-once: exact-cache entries and `last_prediction.json` are deleted on use — a hit can't re-fire stale content or inflate the hit rate
- [x] `DONE` TTL: cache entries expire after `SPEC_CACHE_TTL_MIN` (120 min) — speculative results go stale (R3)
- [x] `DONE` Fix transcript parsing: last `type=="user"` entry is often a tool_result, not the user's prompt — now takes the last entry with non-empty TEXT
- [x] `DONE` Draft quality: speculation drafts are written FOR a coding agent with tool access (never "I can't access files") and get the session context — live: "Run the benchmark tests" → "`pytest tests/test_bench.py`"
- [x] `DONE` Terse injection wrappers (~60 → ~20 boilerplate tokens per injection)

### Milestone MM — Async multimodal: Gemma consumes images while Claude works  ✅ DONE + live-verified
- [x] `DONE` OCR cache keyed by `path+mtime` (`cache/img_*.json`) — an edited image re-OCRs naturally
- [x] `DONE` `describe.sh --prewarm <path>`: background OCR into the cache (no hook output)
- [x] `DONE` Hook mode is cache-first: prewarmed image → **instant** deny+text (162ms vs 10–60s synchronous)
- [x] `DONE` Trigger 1: image paths mentioned in the prompt → fire-and-forget prewarm (`predict.sh`)
- [x] `DONE` Trigger 2: recently-modified images in the project (≤30 min, ≤8 MB, max 2) → prewarm during idle (`speculate.sh` worker)
- [x] `DONE` Image-mention prompts skip the inline text draft (useless + GPU contention with the prewarm): 6.3s → 110ms
- [x] `DONE` LIVE: prewarm 4.9s in background → interception `cached:true` in 162ms; stats show prewarm/instant counts

### Milestone MX — Creative leverage (extended goal)  ✅ DONE + live-verified
Constraint: save big-model tokens **without reducing output quality** — every offload keeps either
exact data (head/tail/grep), a drill-down escape hatch, or is advisory-only.
- [x] `DONE` **Speculative read-only pre-execution** (true speculative execution): idle-time worker runs a
  FIXED set of repo-status commands (`git status --short --branch`, `git diff --stat`, `git log --oneline -8`),
  direct exec no-shell, 5s timeout, 1200-char cap, and embeds the output in the cached draft — a hit injects
  real repo state with zero tool round trips. **Security:** the model does NOT choose commands (an earlier
  allowlist version was a prompt-injection → file-disclosure vector; removed — see R10). Verified a malicious
  transcript cannot exfiltrate files.
- [x] `DONE` **Large-log offload**: Read of `*.log`/`*.out` ≥ `SPEC_LOG_MINKB` (64 KB) → instant deterministic
  digest (exact first/last 30 lines + `grep -in` error/warn lines + sizes) instead of a raw dump
  (347 KB ≈ 86K tokens → ~1K). Gemma pattern summary is computed in the **background** and appears on
  subsequent reads (async-first, 161–186ms). **Escape hatch:** Read with offset/limit passes through raw;
  small logs untouched; `SPEC_LOG_OFFLOAD=0` disables. Works even with Gemma down (digest is deterministic).
- [x] `DONE` **Token-savings estimator** in `/spec-stats` (clearly-labeled rough heuristics: logs bytes/4,
  ~1K/image, ~200/prediction hit).

#### Evaluated and deliberately deferred (so they aren't re-litigated)
- **PDF offload** (rasterize + per-page OCR): real savings but heavy latency on this GPU and quality risk
  on dense documents; the `pdf-to-markdown` skill already covers the heavy case. Revisit if PDFs recur.
- **Generic large-text-file summarization on Read**: REJECTED — Claude needs exact code/text; violates the
  quality constraint. (Logs are the exception: repetitive + escape hatch.)
- **Tool-output (Bash) compression**: not feasible via hooks — PostToolUse can only ADD context, it cannot
  shrink the tool result the big model already receives.
- **Session-start git brief**: redundant — Claude Code already injects gitStatus at session start.
- **Commit-message drafting by Gemma**: negligible savings, style risk.

### Milestone MH — Post-review hardening (balanced code review, 2026-06-10)  ✅ DONE
Five findings from a full-system review; all fixes respect the original goals (G2/G6/R1/R5/R9 patterns).
- [x] `DONE` Consume-once gap closed: an EXACT cache hit now also clears `last_prediction.json` when its
  key matches — the same draft can no longer re-inject via a later fuzzy match (or double-count as a hit)
- [x] `DONE` Verb gate on fuzzy hits: Jaccard overlap is blind to WHICH word differs ("delete the tests" vs
  "show the tests" = 66% ≥ 34% threshold — verified live). Now the lead word (the imperative verb, filler
  skipped) must also match; logged as `miss_verb_gate`; `SPEC_MATCH_VERB=0` restores old behavior
- [x] `DONE` Image offload kill switch + escape hatch (R5 parity with logs): `SPEC_IMAGE_OFFLOAD=0`
  disables; a Read with offset/limit passes the raw image through; deny reason advertises the hatch.
  Default unchanged (G6 still enforced automatically)
- [x] `DONE` Binary guard on the log digest: `*.out` also matches compiled binaries — a NUL byte in the
  first 4 KB now passes the read through untouched (text `.out` logs still digest)
- [x] `DONE` Draft outcome tracking (Q2): `predict.sh` stashes each injected draft (`pending_outcome.json`);
  the Stop worker (no server needed, off critical path per G2) judges accepted/superseded by word
  containment vs the assistant's answer (`SPEC_ACCEPT_MIN`, default 60%); `/spec-stats` shows acceptance
  rate. Repo-state pre-exec output now stored as a separate `obs` field so git noise can't skew the metric
- [x] `DONE` Tested: lead-word/containment units; exact-hit consumes both records + writes pending; verb
  gate blocks antonyms / passes same-verb / env-disableable; binary `.out` passes, 100 KB text `.out`
  digests; `SPEC_IMAGE_OFFLOAD=0` + offset hatch + server-down degrade all allow; outcome accepted(100%)
  / superseded(0%) logged; stats renders the new sections

### Orthogonal (not blocking v1)
- [ ] `TODO` (§8) Token-level llama.cpp `--model-draft` speculative decoding in `start.sh`

## 8. Orthogonal bonus — token-level speculative decoding

Independent of the agent harness: llama.cpp can run a small **draft GGUF** (`--model-draft`,
`--draft-max`) to speed up Gemma itself (draft+verify at the *token* level). One server-flag +
a second small GGUF. Speeds up every Gemma call above. Tracked, not required for v1.

## 9. Usage / enabling

All surfaces live under `.claude/` and are committed, so they activate on a fresh clone
(hooks reload at session start). They are **inert until `llama-server` is up** (R4/R5).

| Surface | Event | Effect | Toggle |
|---|---|---|---|
| `ensure-server.sh` | (called by hooks) | auto-launch optimal server if down (single-flight, non-blocking) | `SPEC_AUTOSTART=0` to disable |
| `predict.sh` | UserPromptSubmit | inject cache-hit / branch-predicted / easy-only inline draft; prewarm images named in the prompt | always on (no-op when server down) |
| `speculate.sh` | Stop | background: predict + pre-draft next turn, **pre-execute safe read-only commands**, pre-OCR recent images | always on |
| log offload | PreToolUse(Read) | `*.log`/`*.out` ≥64 KB → instant digest (exact head/tail + error-grep + async Gemma summary); offset/limit Reads pass raw | `SPEC_LOG_OFFLOAD=0` to disable |
| `describe.sh` | PreToolUse(Read) | image → Gemma OCR/text, deny raw read (0 image tokens) | always on |
| `review.sh` | PostToolUse(Write\|Edit) | Gemma second opinion on edited files | **off** — `export SPEC_REVIEW=1` |
| `gemma-draft` skill | manual | deliberately ask Gemma (draft / OCR) | invoke by intent |
| `/spec-stats` | command | branch-prediction hit rate + image KB saved | run on demand |

Key env knobs (all optional): `SPEC_HOST`/`SPEC_PORT`/`SPEC_MODEL` (server),
`SPEC_PREDICT_MAX` (inline draft tokens), `SPEC_MATCH_MIN` (hit threshold %),
`SPEC_MATCH_VERB=0` (drop the lead-word gate on fuzzy hits), `SPEC_ACCEPT_MIN`
(outcome containment %, default 60), `SPEC_IMAGE_OFFLOAD=0` (disable image offload;
offset/limit Reads always pass raw),
`SPEC_IMAGE_MAX`/`SPEC_IMAGE_TIMEOUT` (image OCR), `SPEC_REVIEW=1` (enable review),
`SPEC_THINK=1` (re-enable Gemma reasoning; off by default), `SPEC_AUTOSTART=0` (disable
auto-launch), `SPEC_AUTOSTART_DRYRUN=1` (show the resolved launch, don't launch),
`CTX`/`NCMOE`/`KVQUANT`/`BACKEND` (override the tuned pick). The auto-started server uses
your `.gemma4-tuning` optimal (e.g. cuda/65536/NCMOE=22) and loads the mmproj for images.

To run it live: just use Claude Code in this repo — the first prompt auto-starts the server
(or run `bash scripts/start.sh` yourself). Watch `/spec-stats` climb.

## 10. Progress log

Newest first. One line per meaningful change; reference commits/tags.

- `2026-06-10` — MH done (post-review hardening from a balanced full-system review). Exact-hit now consumes the fuzzy record too (no double-injection); verb gate on fuzzy hits (antonym prompts scored 50–66% Jaccard — verified — and would have injected wrong-direction drafts); image offload gets `SPEC_IMAGE_OFFLOAD=0` + offset/limit raw-pixel escape hatch (G6 default unchanged); NUL-check stops `*.out` binaries reaching the log digest; accepted/superseded outcome tracking via word containment answers Q2 crudely (`/spec-stats` shows acceptance rate; repo-state obs stored separately so it can't skew the metric). All paths tested offline.
- `2026-06-10` — SECURITY FIX (automated commit review, HIGH): the MX pre-execution let Gemma propose commands from an allowlist → prompt-injection arbitrary-file-disclosure (`head /etc/passwd` passed). Replaced with a FIXED hard-coded git-status command set; the model now chooses nothing and no transcript text reaches a shell. Verified a malicious transcript leaks nothing. R10 updated.
- `2026-06-10` — MX done (creative leverage, extended goal). Speculative read-only pre-execution (predicted "Commit the changes" → pre-ran git status+diff; allowlist unit-tested), large-log offload (347 KB → ~1K-token digest in 161ms, async Gemma summary, offset/limit escape hatch), token-savings estimator in /spec-stats. Evaluated-and-deferred ideas recorded so they aren't re-litigated. New R9/R10.
- `2026-06-10` — MM done (async multimodal, per user request). Background image pre-OCR (path+mtime cache): prompt-mention + recent-files triggers; interception now instant on prewarmed images (162ms); image-mention prompts skip the inline draft (6.3s→110ms).
- `2026-06-10` — MO done (token-efficiency review, prompted by a live wrong-draft injection). Easy-only + length-gated injection, consume-once + TTL caches, transcript-parse fix, agent-aware context-fed speculation drafts, terse wrappers. Hard path 3.7s→0.8s. New R8 (injection cost). Big tier now Fable 5.
- `2026-06-10` — MA done + everything verified LIVE. `ensure-server.sh` auto-launches the optimal server (single-flight lock, setsid-detached, non-blocking) reusing `.gemma4-tuning` (cuda/65536/NCMOE=22) + `--image`; triggered from predict.sh/speculate.sh; kill switch + dry-run. Live: auto-start healthy in 38s (prompt returned 113ms); image OCR exact; branch-prediction hit 85% in 186ms. Fixed R6 (Gemma thinking ate the token budget → empty drafts; now `enable_thinking:false` by default) and measured R7 (inline draft ~3.7s).
- `2026-06-10` — M5 done + v1 feature-complete. `review.sh` PostToolUse second-opinion hook (off unless `SPEC_REVIEW=1`); `gemma-draft` skill for deliberate use; added §9 Usage. Plugin repackaging deferred by decision (committed `.claude/` already ships the behavior). Remaining: live-server smoke tests only.
- `2026-06-10` — M4 done. `stats.sh` summarizes predictor accuracy (overall + online hit rate, avg fuzzy-match %, background speculations, image KB saved) from stats.jsonl; `/spec-stats` slash command wraps it. Verified on synthetic stream. Next: M5 (review.sh + plugin graduation).
- `2026-06-10` — M3 done. `speculate.sh` Stop hook detaches a setsid worker that predicts the next request and pre-drafts it into `cache/` + `last_prediction.json`; hook returns ~12ms (non-blocking). `predict.sh` matches the real next prompt via lexical Jaccard (`spec_similarity`, threshold `SPEC_MATCH_MIN`) → `hit_predicted` with score. Verified with mocks. Next: M4 observability.
- `2026-06-10` — MI done (multimodal image offload, MUST/G6, added per user request). `gemma.sh --image` (OpenAI vision), `describe.sh` PreToolUse(Read) hook denies image reads and hands Opus Gemma's OCR/text so Opus pays 0 image tokens. Safe-degrade (R5) verified for server-down/non-image/non-Read; deny-JSON shape verified. Live OCR test pending server. Next: M3.
- `2026-06-10` — M2 done. `predict.sh` UserPromptSubmit hook: warm-cache HIT injects draft, cache MISS does a short inline classify+draft (server up), server-down/empty → silent no-op. Registered in committed `.claude/settings.json`. All paths log to stats.jsonl with latency. Next: M3 background speculation.
- `2026-06-10` — M1 done. `lib.sh` + `gemma.sh` under `.claude/spec/`; graceful degradation verified (server down → exit 3, no noise); stats logging works. `.gitignore` updated. Next: M2 synchronous path.
- `2026-06-09` — M0 done. Branch created, PRD authored as living tracker. Status: planning → ready to build M1.

# The speculative agent — Gemma 4 drafts, the big model verifies

Hooks that put the **local Gemma 4 server in front of Claude Code** the way a CPU puts a
branch predictor in front of the pipeline: the cheap tier guesses ahead, the expensive tier
verifies. When the guess lands, an expensive **generate** turns into a cheap **accept** —
fewer big-model tokens for the same output quality.

Everything lives in this directory, ships with the repo, and is **inert until
`llama-server` is up** (and the hooks auto-start it for you). No install step.

> Full design history, constraints, and the decision log: [docs/PRD-speculative-agent.md](../../docs/PRD-speculative-agent.md).

---

## Quick start

1. **Prerequisites** — you've run the repo setup once (`bash scripts/setup.sh`, see the
   [main README](../../README.md)), so the model is downloaded and `jq`, `curl`, and the
   llama.cpp environment exist.
2. **Open Claude Code in this repo.** That's it. The hooks in
   [`.claude/settings.json`](../settings.json) register automatically at session start.
3. **Submit any prompt.** If the Gemma server is down, the first prompt auto-launches it in
   the background with your machine's tuned-optimal config (this turn just runs normally;
   the server is typically healthy in ~40 s). Subsequent prompts get drafts.
4. **Watch it work** — run `/spec-stats` after a few turns:

   ```
   speculative-agent stats  (12 prompts seen)
   ────────────────────────────────────────────
     cache hit (exact)        : 1
     branch predicted (fuzzy) : 2   avg match 78%
     inline draft, easy (miss): 3
     hard — left to big model : 4
     ...
     BRANCH-PREDICTION HIT RATE
       online  : 3/11 = 27%
     DRAFT OUTCOMES (containment heuristic)
       accepted : 2 · superseded : 1   (66% accepted)
   ```

5. **Turn things off** (optional): `SPEC_AUTOSTART=0` (no auto-launch),
   `SPEC_IMAGE_OFFLOAD=0` (read images normally), `SPEC_LOG_OFFLOAD=0` (read logs raw).
   With the server down and autostart off, every hook is a silent no-op.

---

## What runs when

| Hook event | Script | What it does |
|---|---|---|
| `UserPromptSubmit` | `predict.sh` | Inject a pre-computed or freshly-drafted answer into the big model's context (the "draft channel") |
| `Stop` (turn ends) | `speculate.sh` | In the background: judge the last draft, **predict your next request**, pre-draft it, pre-fetch repo state, pre-OCR recent images |
| `PreToolUse(Read)` | `describe.sh` | Images → Gemma OCR text instead of pixels (0 image tokens). Big logs → deterministic digest instead of a raw dump |
| `PostToolUse(Write\|Edit)` | `review.sh` | Optional second opinion on edited files (**off** unless `SPEC_REVIEW=1`) |

Plus two manual surfaces: the `/spec-stats` command and the `gemma-draft` skill
(deliberately ask the local tier first — "ask Gemma", "OCR this").

---

## How it works (technical)

### The CPU analogy, made literal

| Speculative execution | This system |
|---|---|
| Draft unit (cheap, guesses) | Gemma 4 26B-A4B on llama-server (:8080) |
| Target unit (authoritative) | The big Claude model |
| Branch predictor | Gemma predicting your *next* prompt from the last exchange |
| Speculative work | Drafting that predicted prompt's answer **while you're idle** |
| Misprediction flush | Wrong guess → cache entry expires or is superseded; cost ≈ 0 big-model tokens |
| The win | Big model **verifies** a correct draft instead of generating from scratch |

### Prompt time (`predict.sh`, synchronous — must be fast)

Checked in order; first match injects and stops:

1. **Exact cache hit** — your prompt hashes to a pre-computed draft from the background
   speculator. Injected instantly (no model call). Consumed on use.
2. **Fuzzy hit** — your prompt ≈ the last *predicted* prompt: word-set (Jaccard) overlap
   ≥ 34% **and** the same lead verb ("delete the tests" vs "show the tests" overlaps 66%
   but must not match — the verb gate catches that). Injected instantly, consumed on use.
3. **Inline draft** — cache miss, server up, prompt ≤ 240 chars: one short Gemma call
   classifies `easy|hard`. **Only easy drafts are injected** — a wrong draft costs input
   tokens instead of saving them, so hard prompts go to the big model untouched (~0.8 s
   overhead). Long prompts skip the call entirely (0 ms).
4. **Server down** — silent no-op (and a background auto-launch kicks off).

Every injection is wrapped in "verify, supersede if wrong" framing — drafts are
**advisory context**, never auto-applied.

### Idle time (`speculate.sh`, detached `setsid` worker — never blocks you)

When a turn ends, the worker:

1. **Judges the previous draft** (if one was injected): did the big model's answer reuse
   the draft's words (containment ≥ 60% → `accepted`) or do its own thing (`superseded`)?
   This is the `/spec-stats` acceptance rate — the predictor's real accuracy, not just
   its activity.
2. **Predicts your next request** from the last user/assistant exchange (one short line).
3. **Pre-drafts the answer** to that predicted request, written for an agent with file
   and shell access (never "I can't see your files").
4. **Pre-executes repo status** — exactly `git status --short --branch`, `git diff --stat`,
   `git log --oneline -8`. This command set is **hard-coded**: the model chooses nothing,
   no transcript text ever reaches a shell (an earlier model-picks-from-allowlist design
   was a prompt-injection vector and was removed — PRD R10).
5. **Pre-OCRs recent images** (≤30 min old, ≤8 MB, max 2) so a future Read is instant.

### Read interception (`describe.sh`)

- **Images**: Gemma OCRs/describes the file locally and the big model gets *text* — zero
  image tokens. Cached by path+mtime (editing the image re-OCRs). Prewarmed reads return
  in ~150 ms; cold ones take one synchronous OCR (10–60 s).
  *Escape hatch:* a Read with `offset`/`limit` passes the raw image through (the deny
  message says so), and `SPEC_IMAGE_OFFLOAD=0` disables interception entirely.
- **Large logs** (`*.log`/`*.out` ≥ 64 KB): instead of a raw dump (~250K tokens/MB), the
  big model gets exact first/last 30 lines + the first 40 `error|warn|fail...` grep
  lines + a Gemma pattern summary (computed in the background, appears on the next read). Binary
  `.out` files (NUL bytes) pass through untouched.
  *Escape hatch:* `offset`/`limit` Reads pass raw — exact bytes are always one call away.
  This is **never** applied to code or other text files: the big model needs exact source.

### Safety properties

- **Graceful degradation everywhere** — server down / not an image / OCR fails / jq error:
  hooks emit nothing and the normal turn proceeds. No path can error your prompt.
- **Speculation is read-only and discardable** — drafts are advisory, caches are
  consume-once with a 120-min TTL, nothing is ever auto-applied.
- **No model-chosen commands** — the only pre-executed commands are the three hard-coded
  git inspections above, run by direct exec (no shell), 5 s timeout, output capped.
- **Local only** — everything talks to `127.0.0.1:8080`; nothing leaves the machine.

---

## Configuration (all optional, env vars)

| Variable | Default | Meaning |
|---|---|---|
| `SPEC_AUTOSTART` | `1` | Auto-launch the optimal server when down (`0` = off) |
| `SPEC_IMAGE_OFFLOAD` | `1` | Intercept image Reads with local OCR (`0` = off) |
| `SPEC_LOG_OFFLOAD` | `1` | Digest big logs (`0` = off) · `SPEC_LOG_MINKB` (64) sets the threshold |
| `SPEC_REVIEW` | `0` | Gemma second opinion after Write/Edit (`1` = on) |
| `SPEC_MATCH_MIN` | `34` | Fuzzy-hit word-overlap threshold (%) |
| `SPEC_MATCH_VERB` | `1` | Require the lead verb to match on fuzzy hits (`0` = overlap only) |
| `SPEC_ACCEPT_MIN` | `60` | Draft-word containment (%) to count an outcome as `accepted` |
| `SPEC_INLINE_MAXCHARS` | `240` | Longest prompt that still gets an inline draft attempt |
| `SPEC_CACHE_TTL_MIN` | `120` | Minutes before speculative cache entries expire |
| `SPEC_HOST` / `SPEC_PORT` / `SPEC_MODEL` | `127.0.0.1` / `8080` / gemma-4-26b-a4b-qat | Where the draft tier lives |

Runtime state (`cache/`, `stats.jsonl`) is gitignored.

## Troubleshooting

- **No drafts appearing?** `curl -s localhost:8080/health` — if down, check
  `/tmp/gemma4-server.log` (the auto-launcher writes there) or start manually with
  `bash scripts/start.sh`. `SPEC_AUTOSTART_DRYRUN=1 .claude/spec/ensure-server.sh` prints
  what would be launched without launching.
- **Drafts feel wrong/noisy?** That's expected sometimes — they're advisory and the big
  model supersedes them. Check `/spec-stats`: a low acceptance rate on `miss_drafted`
  means the easy/hard classifier is over-claiming; raise the bar by lowering
  `SPEC_INLINE_MAXCHARS` or live with it (a superseded draft costs on the order of
  100–200 input tokens).
- **Want a one-off Gemma call?** `.claude/spec/gemma.sh --system "be terse" "your prompt"`
  or `--image pic.png` for OCR.

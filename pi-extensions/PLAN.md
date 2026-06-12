# pi Extensions Plan — empowering local Gemma4

**Target runtime:** pi (`@earendil-works/pi-coding-agent`) running Gemma4 20B QAT 4-bit
via custom locally-compiled llama.cpp. Max context ~120k. Fully standalone — no cloud
model available at runtime. Claude Code is used only to *author* these extensions.

**Motivation:** carbon/energy reduction. Prefill dominates laptop inference energy.

## Cross-cutting rules (apply to EVERY extension)

- **R1 — KV-cache discipline.** llama.cpp reuses KV cache only for the unchanged prompt
  *prefix*. System prompt, tool schemas, and any session-start injection must be
  byte-stable for the whole session (snapshot at `session_start`). Anything dynamic
  (recalled memories, plan state, nudges) is injected at the TAIL of the message list.
  Keep pi on a single llama.cpp slot.
- **R2 — Teaching errors.** A rejected tool call never returns "invalid input"; it
  returns what was wrong + one correct example. Small models retry exactly as well as
  the error message instructs.
- **R3 — Output caps.** Every tool result is hard-capped (~50 lines / ~2 KB) with an
  explicit continuation hint, e.g. "412 more lines — call again with offset=50".
- **R4 — Enforce > persuade.** If a behavior must happen, the tool does it on Gemma's
  behalf (redirects, auto-checks) rather than relying on a prompt rule.
- **R5 — Terse schemas.** Tool descriptions one line; minimal parameters; few tools.
  Schemas are resent every request — they are a standing prefill tax.
- **R6 — Templates over open prompts.** Anything Gemma must *write* (memories,
  snapshots) gets a fill-in template (Task:/Done:/Next:/Files:), never "summarize".

**Reference extension:** `web-search/` in this directory (working example of
`registerTool`, typebox params, error shape). API types:
`web-search/node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts`.
Useful events: `session_start`, `session_before_compact`, `session_shutdown`,
`context` (returns modified message list — injection point), `tool_execution_end`,
`thinking_level_select`, `registerCommand`.

---

## Work items, in build order

### 1. verified-edits
**Goal:** every file write auto-runs the cheapest checker and returns errors in the
same tool result, replacing Gemma's weak self-verification with ground truth.
**How:** hook `tool_execution_end` for pi's built-in edit/write tools; by file
extension run: `.py` → `python -m py_compile` (fall back `ruff check` if available),
`.ts/.js` → `npx tsc --noEmit --allowJs` (project tsconfig if present), `.rs` →
`cargo check --message-format short`, `.go` → `go vet`. 5 s timeout per check;
on timeout/missing checker, stay silent (never block the edit).
**Result append format:** `CHECK FAILED (py_compile): line 12: unexpected indent`
or nothing on pass.
**Accept:** editing a Python file with a syntax error surfaces the error in the same
turn; editing a clean file adds zero output; a missing checker doesn't error the turn.

### 2. symbols
**Goal:** Gemma never reads 800 lines to find one signature.
**Tools:** `get_symbols(path)` → function/class signatures + imports, one per line with
line numbers. `find_symbol(name)` → definition site(s) project-wide.
**How:** universal-ctags if installed (`ctags --output-format=json`), else bundle
tree-sitter (npm: `tree-sitter`, language packs for py/ts/js/rs/go/c). Cache per file
mtime. Project index for `find_symbol` rebuilt lazily.
**Enforcement (R4):** intercept reads of code files >200 lines → return outline +
note: "full content: pass full=true".
**Accept:** `get_symbols` on a 500-line .py returns <40 lines incl. line numbers;
`find_symbol` locates a symbol defined in an unopened file; big-file read redirects.

### 3. loop-breaker
**Goal:** stop Gemma repeating a failing action.
**How:** `tool_execution_end` keeps a rolling window of (tool, args-hash, isError).
On 3rd identical failing call, append to the result: "This exact call failed 3 times.
Change approach or ask the user." Reset counter on any different call.
**Accept:** simulated 3× identical failing bash call gets the nudge on the 3rd; an
interleaved different call resets the count. ~30 lines total.

### 4. plan
**Goal:** external task state so Gemma stops re-deciding what it's doing.
**Tools:** `plan_set(steps: string[])` (once per task), `plan_check(step_index)`,
`plan_show()`.
**How:** state in memory + mirrored to `<session-dir>/plan.json`. `context` handler
injects current state at TAIL each turn, one line per step: `[x]/[ ]`. Cap 10 steps,
80 chars each (R2 error if exceeded).
**Compaction tie-in:** `session_before_compact` writes a snapshot using template
(R6): Task / Done / Next / Files touched — persisted so item 6 can ingest it.
**Accept:** plan survives compaction; tail injection visible in request log; caps
enforced with teaching errors.

### 5. semantic-memory
**Goal:** cross-session memory with zero retrieval burden on the model.
**Storage:** `~/.pi/memory/<project-slug>/MEMORY.md` (curated, human-editable) +
`chunks.jsonl` ({id, text, source, date, vector b64 Float32Array}).
**Embeddings:** second tiny `llama-server --embeddings` (EmbeddingGemma GGUF, CPU ok),
endpoint in extension config. Graceful degradation: Ollama-down ⇒ queue text un-embedded,
embed lazily; recall falls back to substring search.
**Index:** brute-force cosine over Float32Array. NO ANN/quantization/Turbovec below
~100k vectors. Hide behind `search(vec, k)` so the index is swappable.
**Tools:** `remember(fact)` (writes MEMORY.md + index; description carries format:
one line, concrete, include paths), `recall(query, k=3)`, `forget(id_or_text)`.
**Auto-recall (the point):** `context` handler embeds the latest user message
(~10 ms), injects top-2 chunks ≥0.55 cosine at TAIL, ≤150 tokens each.
**Passive injection:** MEMORY.md (≤1 KB) injected once, byte-stable from
`session_start` (R1) — new facts become visible next session.
**Ingestion:** `session_before_compact`/`session_shutdown` snapshots from item 4 are
embedded as chunks automatically.
**Accept:** fact remembered in session A is auto-injected in session B when relevant;
MEMORY.md injection is byte-identical across turns of one session; kill the embedding
server and nothing errors.

### 6. operating-manual (self-awareness scaffold)
**Goal:** Gemma always uses the empowering tools.
**How:** ~10 if-then imperative lines appended to the system prompt (stable prefix,
R1): "Before reading any code file, call get_symbols first." "Never do arithmetic
yourself — use bash + python." "If a command fails twice the same way, change
approach." Rules are triggers→actions, never introspection. Plus just-in-time
one-line nudges at tail via `tool_execution_end` (e.g. grep with >100 matches →
"narrow the pattern"). Loop-breaker (item 3) is the sharpest instance; reuse its hook.
**Accept:** system prompt grows ≤600 bytes; nudges only fire on their trigger.

### 7. stats
**Goal:** every optimization proves its value in numbers (spec-stats style).
**How:** capture llama.cpp `timings` (prompt_n, predicted_n, prompt_ms, predicted_ms)
per request via `after_provider_response`; append to
`~/.pi/memory/<slug>/stats.jsonl`. `registerCommand("stats")` prints session totals:
prefill/decode tokens, est. cache-hit ratio (prompt_n vs context length), est. Wh
using calibrated J/token. One-time calibration script reads
`/sys/class/powercap/intel-rapl` while running a fixed prompt (separate
`calibrate.mjs`, run manually).
**Accept:** /stats shows non-zero prefill/decode split after one turn; Wh estimate
appears once calibration file exists.

### 8. fetch-page
**Goal:** close the search→read loop (Gemma's world knowledge is thin).
**How:** reuse web-search's stealth Playwright setup (UA from `browser.version()`
minus "Headless", `--disable-blink-features=AutomationControlled`, webdriver hidden —
see `web-search/index.ts`, fixed 2026-06-10). Extract readable text (strip
nav/script/style; Readability if cheap to vendor). Output capped per R3 with offset
continuation.
**Accept:** fetches a JS-light article to clean text ≤50 lines per call; paginates.

### 9. goal (autonomous-loop anchor)
**Goal:** a durable, machine-checkable north-star that keeps unattended Gemma working until
the objective is *provably* met, then stops cleanly — so a `/loop`-style run terminates on a
real done-condition instead of stopping early (premature "done") or never stopping (drift /
runaway). This is the **macro-loop manager**: `plan` tracks the steps *inside* one cycle;
`goal` decides when the whole job is finished and drives the next cycle. The energy lever
(the whole point): a verified stop kills the loop the moment the objective is met, and a
cycle budget caps worst-case wasted carbon.

**Distinct from neighbors** (the repo cares about non-overlap):
- `plan` = changing tactical steps (HOW), no gate, no self-continuation. `goal` = immutable
  objective + done-condition (WHAT DONE IS) that *drives* the loop. They compose — `goal_done`
  can additionally require the plan's steps complete.
- `loop-breaker` stops a *micro* loop (same failing call ×3). `goal` runs the *macro* loop
  (keep going until done) with a hard cycle cap; loop-breaker's repeated-failure signal is a
  natural "blocked" trigger (future).
- `advisor` rescues a *wrong* plan; `goal` keeps a *right* plan from quitting early. On
  BLOCKED, suggest an `advisor` call (future).

**Weakness covered (for TECHNICAL §15 table):** *No autonomous termination* — unattended
Gemma either declares "done" before the objective is met or never stops. The existing ten
don't cover the macro-loop's stop condition.

**State** (one goal per session; persisted so a resumed/looped session reloads it):
`{objective, criteria:{text,done}[], doneWhen|null, maxCycles, cycle, status:
active|done|blocked, blockedReason?}`. Live JSON at `<session-dir>/goal-<sessionId>.json`
(R1-safe resume) + a human-readable `goal-status.md` snapshot in the project memory dir
(`~/.pi/memory/<slug>/`, shared with plan/semantic-memory) on every cycle and on done/blocked
— durable state a human or the next session can read (mirrors the user's STATE.md
autonomous-loop convention: state on disk, not chat memory).

**Tools** (R5 — few, terse):
- `goal_set(objective, criteria?, done_when?, max_cycles?)` — set the north-star once.
  `objective` one line; `criteria` ≤8 short phrases; `done_when` a shell command (exit 0 ⇒
  objectively done); `max_cycles` default 20. Teaching errors (R2) on empty objective / too
  many criteria / over-length.
- `goal_check(n)` — tick acceptance criterion n (1-based), like `plan_check`.
- `goal_status()` — objective, criteria ticks, cycle/budget, last `done_when` exit + output tail.
- `goal_done()` — Gemma claims completion. **Validates** (pull): runs `done_when` via
  `pi.exec` and checks every criterion ticked. Unmet ⇒ teaching error naming exactly what's
  unmet (failing command + output tail, or the unchecked criteria); status unchanged. Met ⇒
  status=done, write DONE snapshot, stop self-continuation.

**Enforcement — pull + bounded push (R4):**
- *Pull:* `goal_done` is the validating gate above.
- *Push (the loop driver):* on `agent_end`, if status==active —
  1. Run `done_when` (if set). Exit 0 ⇒ mark done even if Gemma forgot to call `goal_done`
     (machine-checkable termination), snapshot, **do not** re-engage.
  2. Else if `cycle < maxCycles`: `cycle++`, then `pi.sendUserMessage(<north-star + unmet
     list + "continue">, {deliverAs:"followUp"})` to start the next cycle. Re-entrancy-guarded
     so it fires at most once per `agent_end`.
  3. Else (budget spent): status=blocked, write a durable BLOCKED snapshot with reason,
     **do not** re-engage — stop cleanly. "BLOCKED is durable, not silent"; never a silent
     runaway.
- `/goal` command (registerCommand) lets a human set/inspect the goal before launching an
  unattended run; `/goal clear` ends the loop manually.

**R-compliance:** R1 — the immutable `objective` is injected byte-stable into the system
prefix via `before_agent_start` (set once at session start; set-mid-session costs one cache
invalidation); all dynamic status (ticks, cycle, `done_when` tail) is tail-injected via
`context`, never the prefix. R3 — `done_when` output clipped (~50 lines / 2 KB) in any
injection; full output saved beside the snapshot with a pointer. R6 — `goal_set` takes
structured fields (never "describe your goal"); DONE/BLOCKED snapshots use a fixed
Objective/Status/Cycles/Unmet/Files template.

**Accept:**
- `goal_set` rejects empty objective / >8 criteria / over-length with teaching errors.
- `goal_done` with a failing `done_when` (or unticked criteria) returns a teaching error
  naming what's unmet; with all met ⇒ status=done.
- `done_when` exit 0 at `agent_end` auto-marks done and suppresses re-engagement.
- After `max_cycles` unmet continuations: status=blocked, durable snapshot written, no
  further `sendUserMessage`.
- objective appears byte-identical in the prefix across turns; dynamic status only at tail.
- goal JSON reloads on `session_start` (resume/loop survives a pi restart).
- `done_when` output is clipped per R3; full output saved with a pointer.
- re-engagement fires at most once per `agent_end` (no double-trigger).
- tests run with no live model and no real loop (simulate `agent_end` + stub `exec`), meeting
  the set's bar (~20+ checks).

**Open question for implementation:** `sendUserMessage` triggering a turn from *inside*
`agent_end` is the first time any extension drives the agent — verify in a real pi run that it
re-engages cleanly (no re-entrancy / double-turn) before relying on it; fall back to
`sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})` if needed.

---

## Engine-level energy levers

- **Thinking-level routing:** ✅ DONE as the `thinking-router` extension (it was
  mis-filed here — pi exposes `setThinkingLevel()`, so it's pi code, not manual).
  Routes off/low/medium per-turn by input difficulty; respects manual `/thinking`.
- **Speculative decoding:** manual, llama.cpp side — `--model-draft` with a small
  same-vocab Gemma4 QAT; ~1.5–2.5× decode, identical output, ~2 GB RAM.
- **Constrained tool calls:** manual — check whether the custom build exposes
  GBNF/JSON-schema enforcement for tool calls; if yes, wire pi to it.

## Rejected (do not revisit without new evidence)

- Turbovec / any ANN index — alpha-stage, pays off only ≫100k vectors (index is
  swappable behind `search()` if that day comes).
- Key-value store_fact/retrieve_fact — models can't query what they don't know they
  forgot; passive injection instead.
- Embedding the codebase — churns too fast; outlines (item 2) beat stale vectors.
- Model routing e2b/e4b — single 20B model; a second model wouldn't stay resident.

## Status

- [x] 1. verified-edits — done, 11 tests passing
- [x] 2. symbols — done, 30 tests passing (regex extractor; tree-sitter swap deferred)
- [x] 3. loop-breaker — done, 15 tests passing
- [x] 4. plan — done, 22 tests passing
- [x] 5. semantic-memory — done, 34 tests passing (needs llama-server --embeddings)
- [x] 6. operating-manual — done, 19 tests passing
- [x] 7. stats — done, 26 tests passing (RAPL energy_uj often root-only)
- [x] 8. fetch-page — done, 18 tests passing (live-verified google + wikipedia)
- [x] +. thinking-router — done, 14 tests passing (engine-level lever as pi code)
- [x] +. advisor — done, 45 tests passing (external reviewer agent via tui-driver; sees the whole session)
- [x] 9. goal — done, 48 tests passing (autonomous-loop anchor: machine-checkable north-star + bounded self-continuation)

All items complete. 282 tests passing across the set (`./run-tests.sh`). goal is the first
extension that *drives* the agent (`sendUserMessage` from `agent_end`) — validate that
re-engagement in a real pi run before relying on unattended loops (fallback:
`sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})`).
Deployed via `install.sh`; extension loading verified in a real pi run (the
model call itself OOM'd in CI's 1.7 GiB, but all extensions loaded cleanly).

Each item is one directory under `pi-extensions/`, symlinked into
`~/.pi/agent/extensions/<name>/` (plus a shared `node_modules` symlink there —
see README, required because pi loads with --preserve-symlinks).

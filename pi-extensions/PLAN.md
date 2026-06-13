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
- **R5 — Terse, model-optimal schemas.** A tool definition is resent every request — a
  standing prefill tax AND the text the model reads to choose a tool, so optimize
  *token-per-behavior*, not prose, and not for humans. One line, imperative, lead with the
  capability. Keep only tokens that route behavior: when to use it vs. an alternative, and the
  one gotcha that prevents misuse. Cut implementation/library names, internal mechanics, human
  rationale, and anything the tool name or parameter type already says. Parameter descriptions
  carry a format/default/example only when it prevents a malformed call. Few tools.
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
- `plan` = the tactical steps (HOW), no gate, no self-continuation. `goal` = the objective +
  done-condition (WHAT DONE IS) that *drives* the loop, with **no checklist of its own** (that
  would duplicate `plan`). They compose via the seam: `goal_done` reads `plan-<id>.json` and
  requires the steps complete.
- `loop-breaker` stops a *micro* loop (same failing call ×3). `goal` runs the *macro* loop
  (keep going until done) with a hard cycle cap; loop-breaker's repeated-failure signal is a
  natural "blocked" trigger (future).
- `advisor` rescues a *wrong* plan; `goal` keeps a *right* plan from quitting early. On
  BLOCKED, suggest an `advisor` call (future).

**Weakness covered (for TECHNICAL §15 table):** *No autonomous termination* — unattended
Gemma either declares "done" before the objective is met or never stops. The existing ten
don't cover the macro-loop's stop condition.

**State** (one goal per session; persisted so a resumed/looped session reloads it):
`{objective, doneWhen|null, maxCycles, cycle, status: active|done|blocked, blockedReason?}` —
no checklist (plan owns that). Live JSON at `<session-dir>/goal-<sessionId>.json`
(R1-safe resume) + a human-readable `goal-status.md` snapshot in the project memory dir
(`~/.pi/memory/<slug>/`, shared with plan/semantic-memory) on every cycle and on done/blocked
— durable state a human or the next session can read (mirrors the user's STATE.md
autonomous-loop convention: state on disk, not chat memory).

**Tools** (R5 — few, terse):
- `goal_set(objective, done_when?, max_cycles?)` — set the north-star once. `objective` one
  line; `done_when` a shell command (exit 0 ⇒ objectively done); `max_cycles` default 20. Use
  `plan_set` for the steps. Teaching errors (R2) on empty / over-length objective.
- `goal_status()` — objective, cycle/budget, last `done_when` exit + output tail.
- `goal_done()` — Gemma claims completion. **Validates** (pull): runs `done_when` via `pi.exec`
  AND reads `plan-<id>.json` to require every step done. Unmet ⇒ teaching error naming the
  failing command / output tail and any unfinished plan steps; status unchanged. Met ⇒
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
invalidation); all dynamic status (cycle, `done_when` tail) is tail-injected via `context`,
never the prefix (the steps are plan's tail injection). R3 — `done_when` output clipped
(~50 lines / 2 KB) in any injection; full output saved beside the snapshot with a pointer.
R6 — `goal_set` takes structured fields (never "describe your goal"); DONE/BLOCKED snapshots
use a fixed Objective/Status/Cycles/Done-when/Last-check template.

**Accept:**
- `goal_set` rejects empty / over-length objective with teaching errors.
- `goal_done` with a failing `done_when` (or unfinished plan steps) returns a teaching error
  naming what's unmet; with all met ⇒ status=done.
- `done_when` exit 0 at `agent_end` auto-marks done and suppresses re-engagement.
- After `max_cycles` unmet continuations: status=blocked, durable snapshot written, no
  further `sendUserMessage`.
- objective appears byte-identical in the prefix across turns; dynamic status only at tail.
- goal JSON reloads on `session_start` (resume/loop survives a pi restart).
- `done_when` output is clipped per R3; full output saved with a pointer.
- re-engagement fires at most once per `agent_end` (no double-trigger).
- tests run with no live model and no real loop (simulate `agent_end`, stub `exec`, simulate
  `plan-<id>.json`), meeting the set's bar (~20+ checks).

**Open question for implementation:** `sendUserMessage` triggering a turn from *inside*
`agent_end` is the first time any extension drives the agent — verify in a real pi run that it
re-engages cleanly (no re-entrancy / double-turn) before relying on it; fall back to
`sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})` if needed.

### 10. grounding (think-time engineering mindset)
**Goal:** make Gemma reason like an engineer, not from recollection — at *think time*, so no
tokens are wasted generating a hand-wavy answer only to review and regenerate it. The framing
is the *scientific method*, not just "look it up": a remembered thing is a hypothesis, not a
fact; every claim must be **established** by one of three means — **derive it** (a mental
experiment), **simulate it** (run a script/test/calculation), or **reference it** (read the
real source). Tools are only *how* you simulate/reference. Distinct from `web-search`/
`fetch-page` (which supply *missing* knowledge): this targets *trusting recollection over proof*.
**How (two injections that bracket the reasoning, like `plan` keeps state present):**
- *beginning* — `before_agent_start` appends a byte-stable `MINDSET` to the system prefix (rule
  R1): the standing principle, always on so it stays cache-stable.
- *end* — `context` appends a *different*, sharper `CHECK` at the TAIL, the last thing read
  before the reasoning starts: an act-now "for each claim — derived / simulated / referenced
  this turn? else prove it or label it unverified" pass.
There is no API to seed the reasoning stream directly; prefix + tail injection is the
highest-salience way to reach it. The tail check skips `off`/`minimal` thinking levels (trivial
turns); the prefix stays unconditional. Two hooks, no state, no tools.
**Prevention, not a gate (by design):** high-salience guidance the reasoning follows, not a
hard guarantee — a guarantee needs detect-and-regenerate, the exact tokens this saves, so
there is deliberately no backstop. Pairs with `thinking-router` (reads its level to skip
trivial turns).
**Accept:** MINDSET + CHECK both carry the three modes and differ from each other; the prefix
injection is byte-stable and unconditional; the tail injection leaves the prefix untouched and
is byte-identical across turns; trivial-turn skip fires on the tail only; degrades when no
thinking level is reported; CHECK survives the threaded context pipeline regardless of order.

### 11. pipe (orchestration / UX — not a weakness fix)
**Goal:** chain slash-commands, e.g. `/pipe /goal implement the results from /plan a python
script that says hello world`, so a multi-command intent runs as one ordered flow.
**Constraint (why it's a command, not real piping):** pi has no command piping — handlers
return `void`, there's no `executeCommand`, and the `input` event is read-only. So `/a | /b`
shell-style piping is impossible.
**How:** one `registerCommand("pipe")` parses the expression itself (`parsePipe` splits on
`/<known-command>` tokens — only known commands split, so `/tmp/foo` in an arg isn't a
command), reverses to innermost-first execution order (the outer command references the inner
one's result), expands each stage into a tool-using directive (`plan`→plan_set, `goal`→goal_set),
and drives the agent with the whole thing via `sendUserMessage` (the only lever). Extend the
`ACTIONS` map to teach it a new command.
**Accept:** the example expands to step 1 = plan, step 2 = goal referencing step 1; unknown/no
command → usage error, agent not driven; path slashes aren't parsed as commands; one
`sendUserMessage` per valid pipe.

### 12. toolsets (context economy — announce only the tools you need)
**Goal:** cut the per-request tool tax (and small-model wrong-tool confusion) by not
announcing situational tools a session doesn't need. The wording pass (R5) shrank each tool
entry; this shrinks the *set*.
**R1 is the whole constraint:** tool schemas live in the KV-cached prefix, so `setActiveTools`
re-prefills from the tools onward. Naive per-turn toggling trades schema tokens for re-prefill
tokens — usually a loss. So the set is chosen ONCE per session (`session_start`) and changes
only on an explicit `/tools` action (bounded, understood). Per-turn auto-gating is deliberately
deferred (future work).
**How:** groups of OUR situational tools — `web` (web_search, fetch_page), `memory` (remember,
recall, forget), `advisor`; everything else (built-ins, symbols, plan, goal) is always active.
Config (`~/.pi/agent/toolsets-config.json`, env `PI_TOOLSETS_DISABLED`) lists groups to disable
at start; `session_start` computes active = current − disabled-group tools via `setActiveTools`,
working from `getActiveTools()` so built-ins / unknown tools are never dropped. A `/tools`
command lists groups and toggles one live. No model-facing tool — just a command + the startup
gate, so it adds nothing to the announcement. Default disables nothing (opt-in). Hiding the
`recall` tool doesn't break semantic-memory's auto-recall (a context injection, not a tool call).
**Accept:** disabling `web` removes web_search/fetch_page and nothing else; `/tools on web`
restores them; built-ins never dropped; unknown groups error cleanly; nothing-disabled is a
no-op; config precedence env > file > defaults.

---

## Engine-level energy levers

- **Thinking-level routing:** ✅ DONE as the `thinking-router` extension (it was
  mis-filed here — pi exposes `setThinkingLevel()`, so it's pi code, not manual).
  Routes off/low/medium per-turn by input difficulty; respects manual `/thinking`.
- **Speculative decoding:** ✅ DONE — Gemma 4 MTP wired into `start.sh` (`MTP=1`,
  `--spec-type draft-mtp`) and measured. Lossless; the QAT draft head is only **0.25 GB**
  (ships in the model repo, fits at the same NCMOE). Gain is regime-dependent, not a flat
  ~1.5–2.5×: **+15–30 % at greedy/coding temp, within measurement noise at temp 1.0** on this
  `--cpu-moe` rig. `p_min` must stay 0 (degenerates output). EAGLE3 was also tried and does NOT
  work on this build. Full study: `docs/mtp-benchmark.md`.
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
- [x] 9. goal — done, 37 tests passing (autonomous-loop anchor: objective + done_when + loop; verifies plan's steps, no checklist of its own)
- [x] 10. grounding — done, 26 tests passing (think-time engineering mindset; MINDSET prefix + prove-it CHECK at tail)
- [x] 11. pipe — done, 23 tests passing (chain slash-commands via nested-command expansion; orchestration/UX)
- [x] 12. toolsets — done, 19 tests passing (context economy: gate situational tool groups; R1-safe, set once per session)

All items complete. 352 tests passing across the set (`./run-tests.sh`). goal is the first
extension that *drives* the agent (`sendUserMessage` from `agent_end`) — validate that
re-engagement in a real pi run before relying on unattended loops (fallback:
`sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})`).
Deployed via `install.sh`; extension loading verified in a real pi run (the
model call itself OOM'd in CI's 1.7 GiB, but all extensions loaded cleanly).

Each item is one directory under `pi-extensions/`, symlinked into
`~/.pi/agent/extensions/<name>/` (plus a shared `node_modules` symlink there —
see README, required because pi loads with --preserve-symlinks).

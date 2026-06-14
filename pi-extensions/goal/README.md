# goal

The macro-loop manager. `plan` owns the **steps** (the HOW); **`goal`** holds the durable
**objective** and a **machine-checkable done condition** (the WHAT/DONE) and decides when the
*whole job* is finished — so an unattended run stops on a real condition instead of declaring
victory early (premature "done") or never stopping (drift / runaway). The verified stop is the
energy lever: the loop dies the moment the objective is provably met, and a cycle budget caps
worst-case wasted carbon.

> **No checklist of its own.** `goal` deliberately does *not* track a list of steps — that would
> duplicate `plan`. Instead `goal_done` reads `plan`'s state and verifies the steps are complete.
> One checklist (in `plan`), one done-decision (in `goal`).

## Tools

- **`goal_set(objective, check?, done_when?, max_cycles?)`** — set the north-star. `check` is *what
  to verify before declaring done* (e.g. "re-render the SVG and confirm it reads as a pelican on a
  bike"); it defaults to a MINDSET-grounded *"verify, don't assume."* `done_when` is an optional
  shell command (exit 0 ⇒ met) for a machine stop; `max_cycles` bounds the loop (default 20).
- **`goal_status()`** — objective, cycle/budget, the current **phase**, the check, last `done_when` result.
- **`goal_done()`** — claim completion. Verifies **`done_when`** (if set) *and* that **plan's steps
  are complete** (it reads `plan-<id>.json`); else returns a teaching error, or marks the goal done.
- **`goal_conclude(outcome, summary)`** — the **model-owned stop affordance**: deliberately land the
  work as `partial` or `abandoned` with a one-line `summary` when the objective can't be fully met.
  *Not* the verified-done path (that's `goal_done`). **Gated** to the cold phase (commit/decide) so it
  can't be used to bail early — see [Annealed nudging](#annealed-nudging-the-teacher-schedule).

Plus a `/goal` command for a human: **`/goal <task>` sets the objective *and starts the loop*** (the
kickoff drives a turn with the full task text via `sendUserMessage`, the lever `pipe` uses); `/goal`
(show); `/goal clear`.

## It's a self-judged loop

`/goal <task>` starts the work **and keeps it going across turns until the goal is reached** — the
thing you actually wanted. After each turn an `agent_end` hook re-engages Gemma (restating the
objective + the completion check, and *why* the last `goal_done` was rejected if it was), and it
stops only when:

- Gemma **applies the check, sees it pass, and calls the `goal_done` tool** (self-judged — no
  external reviewer; Gemma checks its own work, and for visual goals it can re-render and *look*); or
- the **cycle cap** is hit → **blocked** (durable, never a silent runaway); or
- **you press `Esc`, or you type** — see below.

**No hijack — it yields the instant you take over.** The loop's own re-engagements go through
`sendUserMessage`, which pi tags with input `source: "extension"`; *your* typing is `source:
"interactive"`. Two things make it step aside, both checked on that turn's `agent_end`:

- **You type.** The `input` hook flags `source === "interactive"`, so the turn you started never
  re-engages.
- **You press `Esc`.** pi finalizes the aborted turn with `stopReason: "aborted"`; a `message_end`
  latch (with an `agent_end` message-scan backstop, the same signal [`interrupt-notice`](../interrupt-notice/)
  uses) makes the loop yield instead of re-engaging. *This is load-bearing:* without it the loop
  re-engaged straight through repeated `Esc` presses and the only escape was killing pi.

So it runs autonomously yet steps aside the moment you say *or* signal anything. (To end it for
good rather than just pause: `/goal clear`.)

### Two stop signals

- **Self-judged** (no `done_when` — the default). The loop runs until Gemma calls `goal_done` after
  the check passes, or the cap. The rigor lives in the prompt: the north-star and each re-engagement
  tell Gemma to *establish* the result (derive / simulate / read), not assume — then `goal_done`.
- **Autonomous** (`done_when` set). Adds a *machine* stop on top: after each turn the extension runs
  `done_when`; exit 0 → **done** even if Gemma forgot `goal_done`. Authoritative — a machine check
  beats a self-reported checkbox.

`goal_done` gates either way: `done_when` (if set) **and** plan steps complete before accepting.

## Annealed nudging (the teacher schedule)

A flat loop coaches cycle 1 and cycle 19 identically, and then cuts the work off at the budget. `goal`
instead **anneals** the nudge over its own counter (`cycle / max_cycles`), like a good teacher who
also respects the clock — generous and exploratory early, increasingly directive late, and at the end
*insisting you land your best verified answer* rather than yanking the work away. Full rationale:
[`docs/goal-annealing-prd.md`](../../docs/goal-annealing-prd.md).

**Four phases**, chosen by *reserved cycle counts* (not raw temperature), so the arc stays sane at any
budget — always ≥1 explore when the budget allows, the last cycle is **always** decide, and
`max_cycles=1` is pure decide:

| Phase | When (default) | The push cools… | …and so does the verification ask |
|---|---|---|---|
| **explore** | first ~half | range widely, question assumptions, don't lock in | *establish broadly* — derive/run/read, be skeptical of memory |
| **consolidate** | middle | converge on the most promising line, close threads | verify what *this direction* depends on |
| **commit** | last ~25% | commit to your best result, finish it | verify what the *outcome* hinges on; flag the rest unverified |
| **decide** | final cycle | **you cannot iterate further — decide now** | verify the decision-critical claim; mark the rest unverified |

The **honesty floor never melts**: every phase keeps "verified, or explicitly marked *unverified*" —
only the *emphasis and effort-triage* cool, never the bar. (grounding's `MINDSET` enforces that bar on
every call regardless; the cooling register rides here, in `goal`'s push, so grounding is untouched.)

**Graceful stop, not a guillotine.** The terminal *ramps*: the final re-engagement is the **decide**
phase — an explicit "this is your last cycle, land it" — and only if the model still neither finishes
nor concludes does the hard `blocked` backstop fire (durable, never silent). The model's own exit is
`goal_conclude`, unlocked in the cold phase: `partial` (some achieved, gaps flagged) or `abandoned`
(can't be done, with the reason) → a new **`concluded`** status, distinct from `blocked` ("ran out of
road") and `done` (verified). So an unattended run ends on a *stated decision*, not a silent cut.

**Two channels, one schedule:**

- **Channel A — prompt pressure** (default **on**, pure-prompt). The phases above, in `buildContinue`.
- **Channel B — sampling temperature** (default **off**, `PI_GOAL_TEMP_ANNEAL=1`). Cools the model's
  *actual* sampling temperature across cycles via `before_provider_request` — diverse/exploratory
  early, greedy/decisive late, on a **cosine** curve (holds heat through explore, drops late; classic
  geometric cooling is convex and would cool fastest at the *start*, backwards here). It cools the
  request's **own** temperature *downward* toward `PI_GOAL_TEMP_LO` — never raises it — so a base
  temperature you've configured is the hot-end ceiling, not something it clobbers. Fail-open and
  active-goal-gated. The mutation reaches the wire (in pi's openai-completions provider `onPayload`
  runs after the params are built and its return is sent as-is, `openai-completions.ts:146-157`);
  only "does llama.cpp honor the field" (it does) is left unspiked, so it ships off by default.

**Config (env):** `PI_GOAL_ANNEAL=0` disables Channel A (flat fallback, byte-for-byte the old push).
`PI_GOAL_TEMP_ANNEAL=1` enables Channel B. `PI_GOAL_COMMIT_FRACTION` (0.25), `PI_GOAL_EXPLORE_FRACTION`
(0.5), `PI_GOAL_ANNEAL_SHAPE` (`cosine`|`linear`), `PI_GOAL_TEMP_LO`/`PI_GOAL_TEMP_HI` (0.3 / 1.0) tune
the schedule. Malformed values fall back to defaults — a typo never breaks the loop. The active phase
and temperature show up in `goal_status` and the `goal-status.md` snapshot.

## The `plan` ↔ `goal` seam

`plan` persists its checklist to `<session-dir>/plan-<id>.json`; `goal_done` reads that file and
treats any unfinished step as "not done yet". No module coupling — just the file `plan` already
writes. If `plan` isn't in use, that check is simply empty. Conversely `plan`, when every step is
ticked, defers the *finish* to `goal_done` rather than declaring completion itself.

## How it works

- **R1 — stable prefix + dynamic tail.** The immutable `objective` is injected byte-stable into
  the system prompt via `before_agent_start` (the always-present anchor, paid once). Dynamic
  loop state (cycle count, last `done_when` output) rides the **tail** via `context`, folded into
  the trailing user turn as a `<reminder>` block (the shared fleet convention — see
  [`grounding`](../grounding/)'s `ANCHOR`); the steps are `plan`'s injection, not duplicated here.
  Once done/blocked, both fall silent.
- **The loop driver.** An `agent_end` hook is the push: it re-engages any *active* goal via
  `sendUserMessage(deliverAs:"followUp")`. Re-entrancy-guarded (once per `agent_end`) and it yields
  on two signals — an `input` hook flags input `source === "interactive"` (you typed), and a
  `message_end`/`agent_end` latch flags `stopReason === "aborted"` (you pressed `Esc`) — so a turn
  you started or stopped never re-engages.
- **Gradient on rejection.** When `goal_done` is refused (failing `done_when`, or unfinished plan
  steps), the reason is captured and folded into the *next* re-engagement ("Your last goal_done was
  rejected — …"), so the loop hands back something to act on instead of a bare "not yet" — and the
  re-engagement tells Gemma to **call the `goal_done` tool**, not narrate completion in prose.
- **R3 — output caps.** `done_when` output is clipped (~50 lines / 2 KB, keeping the tail) in any
  injection; the full output is saved to `~/.pi/memory/<project>/goal-last-check.log`.
- **R6 — templates.** `goal_set` takes structured fields; the durable
  `~/.pi/memory/<project>/goal-status.md` snapshot uses a fixed
  Objective/Status/Cycles/Done-when/Last-check template. Live state mirrors to
  `<session-dir>/goal-<id>.json` for resume.

`PI_GOAL_TIMEOUT_MS` overrides the `done_when` timeout (default 120 000 ms).

> **Re-engagement is verified (2026-06-13).** A stub-capture run (pi pointed at a fake
> OpenAI endpoint) confirmed `agent_end → sendUserMessage` genuinely re-drives turns in this pi
> build: a goal with `done_when="false"` produced requests 3/4/5 *on their own* (each carrying the
> `buildContinue` re-engagement), stopping at the cap. The `deliverAs:"followUp"` lever works; the
> documented fallback `sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})` was not needed.

## Test

```bash
node --experimental-strip-types goal/test.mjs
```

116 assertions. The original 63: helpers (clip/render/snapshot incl. the completion `check`, the
rejection-gradient in `buildContinue`, `lastAssistantStopReason`), `readPlanRemaining` (the plan↔goal
seam), the tools with validation, the `goal_done` pull gate (`done_when` **and** plan-steps-complete),
the **self-judged loop** (`agent_end` re-engages without `done_when`, budget → blocked, auto-done on a
passing `done_when`), the **yield-to-human** path (`input` `source:"interactive"` suppresses the next
re-engagement; `"extension"` does not), the **Esc/abort stop** (`message_end` latch *and* `agent_end`
message-scan backstop both yield; a normal finish still re-engages; one-shot), the **rejection
gradient** (a refused `goal_done` names its reason in the next push, once), R1 prefix/tail injection,
persistence reload, and `/goal`.

53 more cover **annealing**: the pure schedule (band boundaries for `max_cycles ∈ {1,2,3,20,…}`,
reserved-tail math, monotone/normalized `temperature`, cosine-vs-linear, env config incl. fail-safe
parsing), the **banded `buildContinue`** (phase markers, the honesty floor in every band,
`goal_conclude` offered only in the cold phases), `renderGoal`/snapshot phase + `concluded` lines, the
**`goal_conclude`** gate (refused in explore, accepted in decide, empty-summary guard, persistence,
**reload round-trip** of the `concluded` status + outcome/summary), the **terminal ramp** (final push
is the decide phase; conclude stops the loop; blocked only *after* a decide turn — FR7), **Channel B**
`applyTempAnneal` (fail-open guards: off-by-default, shape-guard, active-goal gate, copy-not-mutate,
cools across cycles, and **FR9 — never clobbers the request's own temperature upward**), and a
**subprocess** check that `PI_GOAL_ANNEAL=0` really gives the flat fallback. No tmux, no live model —
`exec`/`sendUserMessage` are stubbed and `plan-<id>.json` is simulated.

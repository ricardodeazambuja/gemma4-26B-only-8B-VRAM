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
- **`goal_status()`** — objective, cycle/budget, the check, last `done_when` result.
- **`goal_done()`** — claim completion. Verifies **`done_when`** (if set) *and* that **plan's steps
  are complete** (it reads `plan-<id>.json`); else returns a teaching error, or marks the goal done.

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

63 assertions: helpers (clip/render/snapshot incl. the completion `check`, the rejection-gradient in
`buildContinue`, `lastAssistantStopReason`), `readPlanRemaining` (the plan↔goal seam), the tools with
validation, the `goal_done` pull gate (`done_when` **and** plan-steps-complete), the **self-judged
loop** (`agent_end` re-engages without `done_when`, budget → blocked, auto-done on a passing
`done_when`), the **yield-to-human** path (`input` `source:"interactive"` suppresses the next
re-engagement; `"extension"` does not), the **Esc/abort stop** (`message_end` latch *and* `agent_end`
message-scan backstop both yield; a normal finish still re-engages; one-shot), the **rejection
gradient** (a refused `goal_done` names its reason in the next push, once), R1 prefix/tail injection,
persistence reload, and `/goal`. No tmux, no live model — `exec`/`sendUserMessage` are stubbed and
`plan-<id>.json` is simulated.

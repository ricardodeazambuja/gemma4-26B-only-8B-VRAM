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

- **`goal_set(objective, done_when?, max_cycles?)`** — set the north-star once. `done_when` is a
  shell command (exit 0 ⇒ met); `max_cycles` bounds the auto-continue loop (default 20). Break
  the work into steps with `plan_set`.
- **`goal_status()`** — objective, cycle/budget, last `done_when` result.
- **`goal_done()`** — claim completion. Verifies **`done_when`** *and* that **plan's steps are
  complete** (it reads `plan-<id>.json`); returns a teaching error naming exactly what's unmet,
  or marks the goal done.

Plus a `/goal` command for a human: `/goal` (show), `/goal clear`, `/goal <text>` (set an
advisory objective; `done_when` is set via the tool).

## Two modes (the safety line)

Enforcement is **pull + bounded push** (rule R4), and the push is gated on a *trustworthy* signal:

- **Autonomous** — a `done_when` is set. After every agent run, the extension runs `done_when`;
  while it fails *and* the cycle budget remains, it re-engages Gemma for another cycle
  (`sendUserMessage`, restating the objective for drift correction). It stops when `done_when`
  passes (→ **done**, even if Gemma forgot to call `goal_done`) or the budget is spent
  (→ **blocked**, durable, never a silent runaway). `done_when` is authoritative for the
  auto-stop — it's the machine signal, more trustworthy than self-reported checkboxes.
- **Advisory** — no `done_when`. The extension never auto-continues (auto-continuing without a
  machine signal would be unsafe and could hijack an interactive session). `goal_done` still
  gates: it checks plan's steps are complete before accepting.

## The `plan` ↔ `goal` seam

`plan` persists its checklist to `<session-dir>/plan-<id>.json`; `goal_done` reads that file and
treats any unfinished step as "not done yet". No module coupling — just the file `plan` already
writes. If `plan` isn't in use, that check is simply empty. Conversely `plan`, when every step is
ticked, defers the *finish* to `goal_done` rather than declaring completion itself.

## How it works

- **R1 — stable prefix + dynamic tail.** The immutable `objective` is injected byte-stable into
  the system prompt via `before_agent_start` (the always-present anchor, paid once). Dynamic
  loop state (cycle count, last `done_when` output) is injected at the **tail** via `context`;
  the steps are `plan`'s tail injection, not duplicated here. Once done/blocked, both fall silent.
- **The loop driver.** An `agent_end` hook is the push. It is re-entrancy-guarded (fires at most
  once per `agent_end`) and is the first extension in the set that *drives* the agent.
- **R3 — output caps.** `done_when` output is clipped (~50 lines / 2 KB, keeping the tail) in any
  injection; the full output is saved to `~/.pi/memory/<project>/goal-last-check.log`.
- **R6 — templates.** `goal_set` takes structured fields; the durable
  `~/.pi/memory/<project>/goal-status.md` snapshot uses a fixed
  Objective/Status/Cycles/Done-when/Last-check template. Live state mirrors to
  `<session-dir>/goal-<id>.json` for resume.

`PI_GOAL_TIMEOUT_MS` overrides the `done_when` timeout (default 120 000 ms).

> **Note (first agent-driving extension).** `goal` re-engages the agent with `sendUserMessage`
> from inside `agent_end`. Validate that re-engagement in a real pi run before relying on
> unattended loops; the fallback is `sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})`.

## Test

```bash
node --experimental-strip-types goal/test.mjs
```

37 assertions: helpers (clip/render/snapshot), `readPlanRemaining` (the plan↔goal seam), the
tools with validation, the `goal_done` pull gate (`done_when` **and** plan-steps-complete), the
`agent_end` push (re-engage, budget → blocked, auto-done on pass, advisory no-push), R1
prefix/tail injection, persistence reload, and `/goal`. No tmux, no live model — `exec`/
`sendUserMessage` are stubbed and `plan-<id>.json` is simulated.

# goal

The macro-loop manager. `plan` tracks the steps *inside* one cycle; **`goal`** holds a
durable, machine-checkable north-star and decides when the *whole job* is finished — so an
unattended run stops on a real done-condition instead of declaring victory early (premature
"done") or never stopping (drift / runaway). The verified stop is the energy lever: the loop
dies the moment the objective is provably met, and a cycle budget caps worst-case wasted
carbon. PLAN.md item 9.

## Tools

- **`goal_set(objective, criteria?, done_when?, max_cycles?)`** — set the north-star once.
  `done_when` is a shell command (exit 0 ⇒ met); `criteria` are optional acceptance phrases;
  `max_cycles` bounds the auto-continue loop (default 20).
- **`goal_check(n)`** — tick acceptance criterion `n` (1-based).
- **`goal_status()`** — objective, criteria, cycle/budget, last `done_when` result.
- **`goal_done()`** — claim completion. Runs `done_when` and checks every criterion; returns a
  teaching error naming exactly what's unmet, or marks the goal done.

Plus a `/goal` command for a human: `/goal` (show), `/goal clear`, `/goal <text>` (set an
advisory objective; `done_when` is set via the tool).

## Two modes (the safety line)

Enforcement is **pull + bounded push** (rule R4), but the push is gated on a *trustworthy*
signal:

- **Autonomous** — a `done_when` is set. After every agent run, the extension runs `done_when`;
  while it fails *and* the cycle budget remains, it re-engages Gemma for another cycle
  (`sendUserMessage`, restating the objective for drift correction). It stops when `done_when`
  passes (→ **done**, even if Gemma forgot to call `goal_done`) or the budget is spent
  (→ **blocked**, durable, never a silent runaway).
- **Advisory** — no `done_when`. The extension never auto-continues, because auto-continuing on
  the model's *self-reported* criteria ticks is exactly the unreliable signal this repo
  distrusts. `goal_done` still gates on the ticked criteria (pull only).

This split also keeps `goal` from hijacking ordinary interactive sessions: no `done_when`, no
push.

## How it works

- **R1 — stable prefix + dynamic tail.** The immutable `objective` is injected byte-stable into
  the system prompt via `before_agent_start` (the always-present anchor, paid once). All
  dynamic progress (criteria ticks, cycle count, last `done_when` output) is injected at the
  **tail** via `context`, so the KV-cache prefix is never disturbed. Once the goal is
  done/blocked, both injections fall silent.
- **The loop driver.** An `agent_end` hook is the push. It is re-entrancy-guarded (fires at
  most once per `agent_end`) and is the first extension in the set that *drives* the agent.
- **R3 — output caps.** `done_when` output is clipped (~50 lines / 2 KB, keeping the tail) in
  any injection; the full output is saved to `~/.pi/memory/<project>/goal-last-check.log`.
- **R6 — templates.** `goal_set` takes structured fields; the durable
  `~/.pi/memory/<project>/goal-status.md` snapshot uses a fixed
  Objective/Status/Cycles/Done-when/Unmet/Criteria template (the STATE.md convention for
  unattended loops). Live state also mirrors to `<session-dir>/goal-<id>.json` for resume.

`PI_GOAL_TIMEOUT_MS` overrides the `done_when` timeout (default 120 000 ms).

> **Note (first agent-driving extension).** `goal` re-engages the agent with
> `sendUserMessage` from inside `agent_end`. Validate that re-engagement in a real pi run
> before relying on unattended loops; the fallback is
> `sendMessage(…, {deliverAs:"nextTurn", triggerTurn:true})`.

## Test

```bash
node --experimental-strip-types goal/test.mjs
```

48 assertions: helpers (clip/render/snapshot), all four tools with validation, the `goal_done`
pull gate, the `agent_end` push (re-engage, budget exhaustion → blocked, auto-done on pass,
advisory no-push), R1 prefix/tail injection, persistence reload, and the `/goal` command. No
tmux, no live model — `exec`/`sendUserMessage` are stubbed.

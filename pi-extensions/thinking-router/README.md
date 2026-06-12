# thinking-router

Spend decode tokens — the expensive, controllable part of laptop inference — in
proportion to how hard each turn is. Easy turns (a greeting, a one-line factual
question) get a low or zero thinking budget; anything that looks like real work
(code, "why/how/design", long prose) gets the full budget.

This is the **pi-code half of the plan's engine-level energy levers**. With a single
resident model you can't usefully route *between* models (a second one won't stay in
RAM), but you can route the *thinking level* on one model — same quality where it
matters, less wasted generation on trivia.

## How it works

- An `input` hook classifies the user's message with `routeLevel` (a pure, tested
  heuristic) and calls `pi.setThinkingLevel(...)` before the turn runs.
- Levels: `off` (trivial), `low` (short factual), `medium` (code / reasoning /
  longer asks). It never forces `high`/`xhigh` — those stay a deliberate user choice.
- **Respects you:** once you set a level manually with `/thinking`, auto-routing
  backs off for the session (detected via `thinking_level_select` with source `set`).
- Only routes genuine human input (`source === "user"`).

## Caveat

This is a coarse keyword heuristic, not a difficulty model — it will occasionally
under-think a tersely-worded hard question. That's why it tops out at `medium` and
never suppresses a level you set yourself.

## Test

```bash
node --experimental-strip-types thinking-router/test.mjs
```

14 assertions: the routing heuristic across trivial/factual/code/reasoning/long
inputs, and the hooks including the manual-pin backoff and the non-user-input guard.

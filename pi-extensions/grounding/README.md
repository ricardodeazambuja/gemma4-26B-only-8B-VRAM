# grounding

Stops Gemma hand-waving — asserting things from training memory instead of checking —
**at think time, not after.** Each turn it appends a terse *reasoning protocol* at the
**tail** of the context: the last thing the model reads before it thinks. The
"did this come from a tool, or from memory?" check then happens *inside* the
chain-of-thought, so the only answer that ever gets decoded is already grounded.
PLAN.md item 10 (think-time energy lever).

## Why think-time, and why no gate

The wasteful design is a post-hoc gate: let the model generate a hand-wavy answer, spend
tokens reviewing it, then spend more tokens regenerating. That burns exactly the energy a
local model can't spare. `grounding` prevents instead of corrects — the directive lands
where the reasoning starts, so the guess is never formed in the first place.

There is **no API to seed Gemma's reasoning stream directly** (the thinking is the model's
own output). Tail injection via the `context` hook is the highest-salience way to reach it:
the protocol sits after the whole conversation, so it frames the immediately-following
reasoning pass while leaving the byte-stable prefix / KV cache untouched (rule R1) — only a
short block re-prefills.

**Honest scope.** This is high-salience guidance the reasoning follows, not a hard
guarantee. A guarantee would require detecting a bad answer and regenerating — the exact
tokens this exists to save — so **by design there is no backstop.** If you ever want
certainty over economy, that's a separate, deliberately heavier mechanism.

## How it works

- **`context` hook, tail injection.** Appends the `PROTOCOL` as the last message each turn.
- **Trivial-turn skip.** When the thinking level is `off`/`minimal` (greetings, "thanks",
  "continue" — routed there by `thinking-router`), there's no reasoning to steer and no
  hand-wave risk, so the injection is skipped to save the prefill. Degrades gracefully: if
  no thinking level is reported, it injects anyway.
- That's the whole extension — one hook, no state, no tools.

The protocol text (tune it freely; it's a single exported constant):

> Before asserting any fact — about this codebase, a file's contents, an API, a command's
> result, or the outside world — check in your reasoning: did this come from a tool, or from
> memory? If it is from memory, verify it now (read / grep / get_symbols / find_symbol /
> web_search / fetch_page / bash) before stating it. If you cannot verify it, say "I haven't
> verified this" rather than presenting a guess as fact. Ground the answer in tool output, not recall.

## Test

```bash
node --experimental-strip-types grounding/test.mjs
```

16 assertions: the protocol's content (reasoning-directed, tool-vs-memory, names tools,
requires an unverified flag, terse), tail injection (appended, correct role, prefix
untouched, byte-identical across turns), the trivial-turn skip, and graceful degradation
when no thinking level is available.

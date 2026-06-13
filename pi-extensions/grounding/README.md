# grounding

Makes Gemma **reason like an engineer instead of from recollection — at think time.** A thing
you "remember" is a hypothesis, not a fact; every claim must be *established* before you rely on
it, by one of three means:

- **derive it** — work it out step by step (a mental experiment you could defend),
- **simulate it** — run it and read the real result (a script, a test, a calculation),
- **reference it** — read the actual source (the file, the docs), not your memory of it.

This is the scientific method, not just "look things up": tools are only *how* you simulate (run
it) or reference (read it).

## Two injections that bracket the reasoning

Like `plan` keeps state present, `grounding` brackets every reasoning pass with **two different
injections**, so the model can't derail from start to finish:

- **Beginning — `MINDSET`** in the byte-stable system prefix (rule R1): the standing principle
  (the three modes above), plus a **"Work economically"** rule — spend tokens only where they buy
  correctness (above all in reasoning: think dense, not human prose), keep it simple, don't
  over-engineer. Always on, so it stays cache-stable; paid once, then cached.
- **End — `CHECK`** appended at the **tail**, the last thing read before the reasoning starts:
  a sharper *act-now* pass — "for each claim you're about to make, have you derived / simulated /
  read it THIS turn? If it rests on memory, do that now or label it 'unverified'." Different text
  from the prefix on purpose: principle up front, checklist at the moment of answering.

There is no API to seed the reasoning stream directly, so prefix + tail injection is the
highest-salience way to reach it. No generate-then-review-then-regenerate: zero wasted
answer/review tokens (the repo's energy thesis applied to itself).

## Why think-time, and why no gate

The wasteful design is a post-hoc gate: let the model produce an ungrounded answer, spend tokens
reviewing it, then regenerate. `grounding` prevents instead of corrects — the guidance lands
where the reasoning starts, so the guess is never formed.

**Honest scope.** This is high-salience guidance the reasoning follows, not a hard guarantee. A
guarantee would require detect-and-regenerate — the exact tokens this exists to save — so **by
design there is no backstop.**

## How it works

- **`before_agent_start`** → appends `MINDSET` to the system prompt (unconditional, byte-stable).
- **`context`** → appends `CHECK` at the tail each turn; the prefix / KV cache is untouched and
  only one short block re-prefills.
- **Trivial-turn skip** — the tail check is skipped when the thinking level is `off`/`minimal`
  (greetings, "continue" — routed there by `thinking-router`): no reasoning to steer. The prefix
  stays unconditional so it remains cache-stable. Degrades gracefully if no level is reported.
- Two hooks, no state, no tools. Both texts are exported constants — tune them freely.

## Test

```bash
node --experimental-strip-types grounding/test.mjs
```

26 assertions: the scientific-method content of both `MINDSET` and `CHECK` (three modes, memory
as hypothesis, unverified label, terse, and that they differ), the byte-stable unconditional
prefix injection, the tail injection (role, prefix untouched, byte-identical across turns), the
trivial-turn skip, and order-independent composition with other tail-injectors.

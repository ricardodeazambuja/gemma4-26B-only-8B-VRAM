# grounding

Makes Gemma **reason like an engineer instead of from recollection — at think time.** A thing
you "remember" is a hypothesis, not a fact; every claim must be *established* before you rely on
it, by one of three means:

- **derive it** — work it out step by step (a mental experiment you could defend),
- **simulate it** — run it and read the real result (a script, a test, a calculation),
- **reference it** — read the actual source (the file, the docs), not your memory of it.

This is the scientific method, not just "look things up": tools are only *how* you simulate (run
it) or reference (read it).

## Three injections, all in or folded into the prompt

- **`MINDSET`** — the standing principle (the three modes above) plus a **"Work economically"**
  rule, appended to the **byte-stable system prefix**. Always on, paid once, then cached.
- **`ANCHOR`** — also in the system prefix: tells the model that blocks wrapped in
  `<reminder>…</reminder>` are *automated context injected by the harness, not a new instruction
  and not the user speaking*, and that its task each turn is the user's **most recent real
  request** (the unwrapped text). This is what keeps the model on the user's instruction even as
  several extensions inject reminders.
- **`CHECK`** — the sharper *act-now* pass: "for each claim you're about to make, have you derived
  / simulated / read it THIS turn? If it rests on memory, do that now or label it 'unverified'."

## Why CHECK is *folded into* the user turn, not appended

CHECK used to be appended as its own tail `role: "user"` message. Two problems, both observed:

1. **It read as a new instruction.** A bare user-role message is, to the model, the user speaking
   — so it would answer the *check* instead of treating it as a self-directed reminder.
2. **It wasn't actually last.** Context hooks run in `readdirSync` order, and `grounding` runs
   *first* among them; `plan` / `goal` / `semantic-memory` / the notices then appended their own
   user turns *after* it (whenever their state was active). So the "last thing read" was a plan
   checklist or memory dump — and the real request was buried under a stack of fake user turns.

So CHECK is now **folded into the trailing user turn** as a `<reminder>` block (the user's real
text stays first), and `ANCHOR` teaches the model what that marker means. No fresh user turn,
nothing masquerading as the user.

**The whole fleet uses this marker now.** `plan`, `goal`, `semantic-memory`, and the
compaction / interrupt notices share a byte-identical `foldReminder` helper: each wraps its
injection in `<reminder>…</reminder>` and folds it into the trailing user turn (appending one only
when the tail is a tool result, where the first to inject creates the user turn the rest fold
into). The net effect is that **every reminder collapses into a single user turn** — the real
request as the unwrapped `content[0]`, every injection a wrapped block underneath — so the one
`ANCHOR` note covers all of them and none impersonate the user. A top-level `marker.test.sh`
asserts the delimiter bytes are identical across all six.

## Why think-time, and why no gate

The wasteful design is a post-hoc gate: let the model produce an ungrounded answer, spend tokens
reviewing it, then regenerate. `grounding` prevents instead of corrects — the guidance lands
where the reasoning starts, so the guess is never formed.

**Honest scope.** This is high-salience guidance the reasoning follows, not a hard guarantee. A
guarantee would require detect-and-regenerate — the exact tokens this exists to save — so **by
design there is no backstop.**

## How it works

- **`before_agent_start`** → appends `MINDSET` + `ANCHOR` to the system prompt (unconditional,
  byte-stable).
- **`context`** → folds `CHECK` into the trailing user turn as a wrapped `<reminder>` block; the
  prefix / KV cache is untouched and only that turn re-prefills.
- **Turn-start only (no treadmill).** `CHECK` fires *once per user request* — only when the
  conversation tail (ignoring reminder-only turns other injectors appended) is the user's genuine
  message. **Mid tool-loop the tail is a tool result, so `CHECK` is skipped.** This is deliberate:
  re-stamping an act-now "prove it" imperative on *every* tool step is exactly what could let a
  small model read it as a fresh instruction to answer and loop on it (read check → "verify" with a
  tool → check re-injected → …). The standing `MINDSET` prefix still grounds those in-between steps;
  only the sharp per-turn pass is gated. So across turns the model gets one nudge per request, never
  a self-sustaining review loop.
- **Trivial-turn skip** — the check is also skipped when the thinking level is `off`/`minimal`
  (greetings, "continue" — routed there by `thinking-router`): no reasoning to steer. The prefix
  stays unconditional so it remains cache-stable. Degrades gracefully if no level is reported.
- **Pure transform.** The `context` edit is per-request and does not mutate the stored
  conversation, so reminders never accumulate across turns. Two hooks, no state, no tools. All
  texts are exported constants — tune them freely.

## Test

```bash
node --experimental-strip-types grounding/test.mjs
```

45 assertions: the scientific-method content of `MINDSET` and `CHECK`, the `ANCHOR` framing, the
byte-stable unconditional prefix injection, the fold-into-the-user-turn behaviour (no new message,
real request stays first, wrapped marker, original not mutated, byte-identical across turns), the
empty-history fallback, the trivial-turn skip, the **turn-start throttle** (`isTurnStart` /
skips mid tool-loop / fires on a genuine user turn), and order-independent composition with other
tail-injectors.

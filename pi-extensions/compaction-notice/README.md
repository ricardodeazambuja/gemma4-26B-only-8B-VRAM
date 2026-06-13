# compaction-notice — tell the model its context was just compacted

When the conversation outgrows the window, pi **compacts** it: older turns are replaced by a short
summary to claw back room. But pi injects **nothing** telling the model this happened. A small model
then keeps trusting its "memory" of detail that was actually summarized away — exact file contents,
values, decisions, what was already tried — and acts on a stale recollection ("I already fixed that
in `foo.ts`" when the real edit is gone from context). This extension closes that gap: it detects
the compaction and injects **one tail note** on the next turn, so the model knows the detail is gone
and re-reads / re-derives instead of guessing.

## How it works

A one-flag state machine, no tools:

- **Detect** (`session_compact`): pi fires this **after** a compaction completes (so the summary is
  already in the message list). It carries `compactionEntry.tokensBefore` — the context size that
  was summarized — which the note surfaces as a concrete "~Nk summarized" anchor. A pending flag is
  latched; it is idempotent, so several compactions before the next call still arm it once.
- **Inject** (`context`): on the next LLM call, if the flag is set, append the notice at the
  **tail** of the message list (after the compaction summary) and clear the flag.

The injected line (the `~Nk` clause is omitted when the token count is unknown):

```
[The conversation was just compacted (~48k tokens of earlier turns were summarized).] Earlier turns
have been replaced by a short summary, so specific detail — exact file contents, values, decisions,
what was already tried — may be lost or imprecise. Do not trust your memory of it: re-read the
files, re-run the checks, or re-derive what you need before acting, and rely on the summary and the
user's latest message for intent.
```

Design-rule fit: **R1** — compaction **rewrites the prompt prefix by definition**, so that turn
already pays a full re-prefill; the tail note's marginal cost is ~zero (strictly cheaper than a note
added to an otherwise byte-stable prefix). **R5** — one terse line with a concrete anchor. **R2** —
it tells the model what to *do* (re-read, re-derive), not just what happened. It is **self-limiting**:
it fires exactly once per compaction (the flag clears on inject), so a normal turn never carries it.

It deliberately ignores `fromExtension`/`fromHook`: even when an extension triggers the compaction
(e.g. [`plan`](../plan/) snapshots its state in `session_before_compact`), pi's underlying
compaction is still lossy, so the warning holds.

## Verify it on your build

The test drives the handlers with event shapes *it* constructs, so it proves the extension's logic
but **not** pi's real behavior. Three load-bearing assumptions only a live run can confirm:

1. pi fires `session_compact` (with `compactionEntry.tokensBefore`) when it compacts.
2. a `context` event fires *after* that, before the next response, so the note lands.
3. the compaction **summary is rendered into that context's message list** — the note tells the
   model to "rely on the summary," which is only true if the summary is actually present.

Confirm it live: drive a real session to a `/compact` (or let it auto-compact) and check the note
appears on the next turn. Turn on debug logging to see exactly what fired:

```bash
PI_COMPACTION_DEBUG=1 pi                       # logs to $TMPDIR/compaction-notice-debug.log
PI_COMPACTION_DEBUG=/tmp/compact.log pi        # or a path you choose (must contain a "/")
```

It appends one line when a compaction arms the flag (with the `tokensBefore` it saw) and one when
the notice is injected. Logging goes to a file, never the console, so it can't corrupt pi's TUI.
Unset or `=0`/`=false` disables it.

## Relationship to interrupt-notice

This is the compaction sibling of [`interrupt-notice`](../interrupt-notice/): same latch → tail-inject
→ fire-once shape, different trigger. Both warn the model about a change to the conversation it can't
otherwise see (you stopped it / its context was summarized). They compose cleanly — each appends to
`event.messages` rather than replacing it, so in the `context` pipeline every notice survives.

## Test

```bash
node --experimental-strip-types compaction-notice/test.mjs
```

Covers the tracker (a compaction arms it; `observe` returns whether it armed; carries the token
count; fires once), the notice text (mentions compaction, says what to do, the `~Nk` anchor rounds /
is omitted when unknown), and the real `session_compact`/`context` handlers (note appended at the
tail, original messages preserved, self-limiting, injects even without a token count). A
child-process case sets `PI_COMPACTION_DEBUG` and checks the log is written. No tmux, no live model
— see the caveat in **Verify it on your build**.

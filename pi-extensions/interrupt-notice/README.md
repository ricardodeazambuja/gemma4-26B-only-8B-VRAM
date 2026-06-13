# interrupt-notice — tell the model when you stopped it

When you interrupt the agent mid-turn (Esc / abort), pi finalizes the half-written
assistant message with `stopReason: "aborted"` and goes idle — but it injects **nothing**
into the conversation. So on your next turn Gemma can't tell its previous answer was cut
off, and a small model will often barrel on as if the interrupted work landed ("done!")
instead of noticing you stopped it. This extension closes that gap: it detects the aborted
turn and injects **one tail note** on the next turn, so the model knows it was interrupted
and waits for your steer.

## How it works

A one-flag state machine, no tools. Detection rides **two** events for robustness, and injection
rides a third:

- **Detect** (`message_end`, primary): every finalized message is inspected; an **assistant**
  message whose `stopReason` is **`"aborted"`** latches a pending flag. (`StopReason` is defined in
  `@earendil-works/pi-ai` as `stop | length | toolUse | error | aborted`; the user-abort path
  — `session.abort()` — is what produces `"aborted"`.)
- **Detect** (`agent_end`, backstop): when the run ends, the most recent assistant message in the
  handed-back conversation is checked the same way. This catches the abort even if a given pi build
  does not emit `message_end` for the aborted partial. The pending flag is idempotent, so both
  events firing on one abort still arms it exactly once.
- **Inject** (`context`): on the next LLM call, if the flag is set, fold the notice into the
  **trailing user turn** as a `<reminder>` block — the shared fleet convention (see
  [`grounding`](../grounding/)'s `ANCHOR`) so it doesn't read as a fresh user instruction — and
  clear the flag.

The injected line:

```
[Your previous response was interrupted by the user before it finished.] Do not assume that
work completed or that the task is done. Re-read the user's latest message and act on it; if
they didn't add anything new, ask what they want next.
```

Design-rule fit: **R1** — the note is a *tail* injection, so the byte-stable prompt prefix (and
the llama.cpp KV cache) is untouched; only one short block re-prefills. **R5** — one terse line.
**R2** — it tells the model what to do, not just what happened. It is **self-limiting**: it
fires exactly once per interrupt (the flag clears on inject), so a normal turn never carries it.

## Verify it on your build

This keys on `stopReason: "aborted"` reaching `message_end` and/or `agent_end` on the user-abort
path. That was read from pi's type surface (v0.79.1), **not** observed live — and the test, which
drives the handlers with event shapes *it* constructs, proves the extension's logic but **not**
that your pi build actually fires those events on a real abort. So confirm it live: interrupt a
real session with **Esc** and check the note appears on the next turn.

To see exactly what fired, turn on debug logging:

```bash
PI_INTERRUPT_DEBUG=1 pi                       # logs to $TMPDIR/interrupt-notice-debug.log
PI_INTERRUPT_DEBUG=/tmp/intnotice.log pi      # or a path you choose (must contain a "/")
```

It appends one line when an abort arms the flag (`armed via message_end` or `armed via agent_end`,
so you learn which event your build emits) and one when the notice is injected. Logging goes to a
file, never the console, so it can't corrupt pi's TUI. Unset or `=0`/`=false` disables it. If a
future pi stops firing **both** events on abort, the debug log will show nothing armed — the place
to look is `turn_end` (it also carries the messages).

## Alternative: implement it directly in pi's source

This extension reacts *after* the fact, on the next LLM call. If you'd rather the notice be
**first-class** — recorded in the session the instant you hit Esc, visible in the transcript and
persisted with the conversation even before any further model call — patch pi core instead:

- Find the abort path in `agent-session.ts` (the `abort()` method / where the assistant
  message's `stopReason` is set to `"aborted"`).
- There, push a synthetic message into the session (a short user/system note like the one above)
  so it becomes part of the durable history rather than an injection re-added each turn.

Trade-offs:

- **Pro:** the marker is part of the real conversation record — it survives export/resume the
  way any message does — and it appears immediately, not only on the next turn. That's closer to
  how cloud harnesses surface a "[Request interrupted by user]" line.
- **Con:** it's a fork of pi. You carry the patch across pi upgrades, and it lives outside the
  symlink-and-go extension flow that keeps everything else in this repo upgrade-safe. The
  extension here needs no fork and rides pi's public event API.

Prefer the extension unless you specifically need the marker persisted in-session; reach for the
core patch only if the event-API approach proves insufficient.

## Test

```bash
node --experimental-strip-types interrupt-notice/test.mjs
```

Covers the tracker (only an aborted assistant turn arms it; `observe` returns whether it armed;
fires once), the notice text, and the real `message_end`/`agent_end`/`context` handlers (note
appended at the tail, original messages preserved, self-limiting, both detect events arm exactly
once together, normal turns untouched). A child-process case sets `PI_INTERRUPT_DEBUG` and checks
the log is written. No tmux, no live model — see the caveat in **Verify it on your build**.

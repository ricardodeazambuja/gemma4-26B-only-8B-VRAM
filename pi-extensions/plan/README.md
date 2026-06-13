# plan

An external task checklist so Gemma stops re-deciding what it's doing every turn.
Small models lose the thread on multi-step work; this gives them a persistent
"where am I" that lives outside their reasoning and is refreshed at the tail of the
context on every turn.

## Tools

- **`plan_set(steps)`** — set an ordered list of short steps (once per task).
- **`plan_check(step)`** — mark a step done by its 1-based number.
- **`plan_show()`** — show the checklist and what remains.

Caps (rule R2 teaching errors): ≤10 steps, ≤80 chars each; violations return a
specific message with a correct example, not "invalid input".

## How it works

- **Tail injection (rule R1).** A `context` hook folds the rendered checklist into the
  **trailing user turn** every turn, wrapped in a `<reminder>…</reminder>` block (the shared
  fleet convention — see [`grounding`](../grounding/)'s `ANCHOR`) so it reads as injected context,
  not a fresh user instruction. Because it rides the tail, the byte-stable prompt prefix — and
  therefore the llama.cpp KV cache — is untouched; only the reminder re-prefills.
- **Survives compaction.** State is module-level (persists across turns in the same
  process) and mirrored to `<session-dir>/plan-<id>.json` for resume.
- **Pre-compaction snapshot.** `session_before_compact` writes a
  `Task / Done / Next / Files touched` snapshot to
  `~/.pi/memory/<project>/snapshots/` — the exact moment detail is about to be
  destroyed. semantic-memory ingests these. "Files touched" is gathered
  passively from a `tool_result` hook on edits/writes.

## Test

```bash
node --experimental-strip-types plan/test.mjs
```

22 assertions: renderers, snapshot template, all three tools with validation,
tail injection (and proof the prefix is unchanged), touched-file tracking, and
the compaction snapshot.

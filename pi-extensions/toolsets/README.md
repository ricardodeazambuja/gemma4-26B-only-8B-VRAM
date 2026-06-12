# toolsets

Context economy: **don't announce situational tools a session doesn't need.** Tool
definitions are resent to the model every request (the standing prefill tax, R5) — and a
smaller, focused set also means fewer wrong-tool calls from a small model. This shrinks the
*set*; the wording pass shrank each entry. PLAN.md item 12.

## R1 is the whole constraint

Tool schemas live in the **KV-cached prefix**, so calling `setActiveTools` **re-prefills** from
the tools onward. Naive per-turn toggling would trade schema tokens for re-prefill tokens —
usually a net loss, and it fights R1. So `toolsets` chooses the set **once per session**
(`session_start`) and changes it only on an explicit `/tools` action (a bounded, understood
cost). **Per-turn auto-gating is deliberately not done here** — it'd cost more cache than it
saves. (A high-precision auto-reveal is possible future work.)

## Groups

Only OUR situational tools are grouped; everything else — pi's built-ins, `symbols`, `plan`,
`goal` — is **always active and never touched**.

| Group | Tools |
|---|---|
| `web` | `web_search`, `fetch_page` |
| `memory` | `remember`, `recall`, `forget` |
| `advisor` | `advisor` |

Hiding the `recall` *tool* does **not** break semantic-memory's auto-recall — that's a context
injection, not a tool call.

## Use

Default disables **nothing** (opt-in, zero surprise). To trim the tax for a session, disable
the groups you won't need — `~/.pi/agent/toolsets-config.json`:

```json
{ "disabled": ["web", "advisor"] }
```

or per-run: `PI_TOOLSETS_DISABLED=web,advisor`. You can also redefine/extend `groups` in the
file. Precedence: **env > file > defaults** (same pattern as advisor/semantic-memory).

Live control (each change re-prefills once):

```
/tools              # list groups and which are active
/tools off web      # hide a group now
/tools on web       # reveal it again
```

## How it works

- **Safety by construction.** It only ever *removes known group tools by name* from the
  current active set (`getActiveTools()`), so built-ins and unrecognised tools can never be
  dropped. Computed via `applyDisabled` (a pure function).
- **`session_start`** applies the configured `disabled` groups once (no-op if none).
- **`/tools` is a command, not a tool** — it adds nothing to the model's announcement.
- Degrades cleanly if a pi build doesn't expose `getActiveTools`/`setActiveTools`.

## Test

```bash
node --experimental-strip-types toolsets/test.mjs
```

19 assertions: config precedence (env > file > defaults), the pure gating helpers (removes
only a group's tools, never built-ins/core), the `session_start` gate (acts only when
something is disabled), and the `/tools` command (on/off, unknown group, bare list).

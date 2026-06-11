# advisor — consult a stronger external agent mid-session

Gives Gemma an `advisor` tool, modeled on Claude Code's advisor: a reviewer
that sees the **whole session** and answers "is this approach sound, what was
missed, what's next?". The reviewer is any TUI agent you choose (agy, claude,
aichat, another pi…), driven programmatically through
[tui-driver](https://github.com/ricardodeazambuja/tui-driver) (tmux).

**Problem this solves.** Every other extension in this set assumes Gemma's plan
is recoverable — verify it, nudge it, remind it. But a wrong plan executed
carefully is still wrong, and a model cannot reliably review its own reasoning.
This is the escalation path: a genuinely different (usually stronger) model
judges the whole session arc. See `docs/TECHNICAL.md` §15 for the full design
rationale.

## How it works

On each `advisor` call the extension:

1. Serializes the current session branch (`ctx.sessionManager.getBranch()`)
   into a readable transcript — user/assistant text, thinking (clipped),
   tool calls (args clipped to ~400 chars), tool results (clipped to
   `maxToolResultChars`), `!` bash runs, extension injections, compaction
   summaries — and writes it to `<private tmp dir>/<call>.md`. The dir is a
   per-process `mkdtemp` (0o700) and files are 0o600: transcripts can contain
   anything the session saw, so they never go world-readable into /tmp.
2. Brings up the configured TUI via `tui-driver <command> start` (only if
   `status` says stopped; the cmd+dir lock makes later calls reuse it). The
   TUI starts in pi's working directory, so an agentic advisor can also read
   project files itself.
3. `tui-driver <command> send "<prompt>"` — blocking; tui-driver waits for the
   screen to stabilise and prints the extracted reply.
4. Returns the reply as the tool result (capped at `maxReplyChars`, full text
   saved next to the transcript). With `keepSession: false` the
   TUI is stopped after each call; otherwise tui-driver's idle watchdog
   (default 10 min, `TUI_WATCHDOG`) reaps it eventually.

The tool takes one optional parameter, `focus`, so Gemma can point the review
("focus: is the locking correct?").

## Configuration (required before first use)

There is deliberately **no default agent** — consulting an external agent can
cost money, so it must be an explicit choice. Until configured, the tool
returns a teaching error showing this exact setup. Precedence:
**env > `~/.pi/agent/advisor-config.json` > defaults** (same pattern as
semantic-memory's embed-config). Quickest start:

```bash
cp advisor/advisor-config.example.json ~/.pi/agent/advisor-config.json
```

```json
{
  "command": "agy",
  "timeoutSec": 600,
  "keepSession": true,
  "inlineTranscript": false
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `command` | — (required) | TUI command tui-driver drives. Becomes part of tmux session/lock names — keep it a single word (use a wrapper script or shell alias for flags). |
| `tuiDriver` | `~/.local/bin/tui-driver` | Path to the tui-driver executable (where `./tui-driver.sh install` puts it). Checked before every call; a missing file returns install instructions. |
| `timeoutSec` | `600` | Max wait for the reply (`TUI_TIMEOUT`). |
| `keepSession` | `true` | Reuse the TUI across calls (faster; watchdog reaps it when idle). |
| `inlineTranscript` | `false` | Paste transcript text into the prompt instead of a file path — for chat TUIs that cannot read files. Capped at `maxInlineChars` (60k). |
| `promptTemplate` | built-in | `{transcript}` → path (or text when inline), `{focus}` → optional focus. File-only (multiline). |
| `maxToolResultChars` | `1500` | Per-message clip in the transcript. |
| `maxReplyChars` | `12000` | Reply cap; full reply is always saved to /tmp. |

Env overrides: `PI_ADVISOR_CMD`, `PI_ADVISOR_TUI_DRIVER`,
`PI_ADVISOR_TIMEOUT_SEC`, `PI_ADVISOR_KEEP_SESSION`, `PI_ADVISOR_INLINE`,
`PI_ADVISOR_CONFIG` (config file location).

## Design-rule compliance

- **Teaching errors** (#2): unconfigured/failed-start/empty-reply errors say
  what to fix and show the example config.
- **Output caps** (#3): transcript entries and the reply are clipped, with a
  pointer to the full saved file.
- **Terse schema** (#5): one-line description, one optional string param.
- **KV-cache** (#1): the schema is byte-stable; everything dynamic lives in
  the tool *result* (tail).

## Why drive a TUI instead of an API?

Alternatives considered and rejected:

- **Direct cloud-API calls** — per-provider keys, billing wiring, and request
  formats inside the extension. Driving a TUI reuses agents the user already
  installed, authenticated, and paid for; supporting a new agent costs one
  config string.
- **A second Gemma/pi as reviewer** — no capability lift; self-review by the
  same weights is the failure mode this escapes. (Still configurable if you
  want a fresh-context second opinion.)
- **MCP/RPC integration per agent** — more machinery, fewer supported agents.
  tui-driver already handles throttling, approval prompts, orphan reaping,
  and works with any TUI unmodified.
- **A default agent** — rejected on cost grounds; the unconfigured tool
  teaches the setup instead of silently picking a vendor.

## Caveats

- **Prerequisites:** `tmux` and tui-driver. tui-driver is expected at
  `~/.local/bin/tui-driver` (run `./tui-driver.sh install` from its checkout);
  the tool verifies this before every call and returns install instructions
  if it's missing.
- The advised TUI must not require interactive approval for reading the
  transcript file (e.g. run claude with permissions that allow reads in /tmp,
  or use `inlineTranscript: true`).
- Reply extraction is screen-scraping: tui-driver returns what appeared after
  the prompt on screen, so TUI chrome can leak into the reply.
- The call is synchronous: Gemma blocks for up to `timeoutSec` while the
  advisor thinks.

## Future work

- **Async mode** — tui-driver has `send-async`/`poll`; return immediately and
  inject the verdict at tail (`deliverAs: "steer"`) when it lands.
- **Auto-escalation** — after loop-breaker's nudge fires repeatedly with no
  behavior change, suggest an `advisor` call (suggest, not force — it costs).
- **Structured verdicts** — a fill-in template (Sound?/Missed:/Next:) per
  design rule R6, so verdicts become parseable and plan-injectable.
- **Reply cleaning** — per-TUI post-filters to strip spinners/box-drawing.

## Test

```bash
node --experimental-strip-types advisor/test.mjs   # 45 checks, no tmux needed
```

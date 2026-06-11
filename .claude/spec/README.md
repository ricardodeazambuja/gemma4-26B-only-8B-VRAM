# The Gemma draft tier — `/gemma-draft`

An explicit slash command that puts the **local Gemma 4 server in front of Claude Code**
as a cheap draft tier: you ask for a draft on purpose, Gemma produces it locally, and the
big model **verifies** instead of generating from scratch. When the draft lands, an
expensive generate turns into a cheap accept.

> **History:** v1 of this system was an always-on hook pipeline (automatic draft injection
> on every prompt, branch prediction of your next request, image-Read interception). It was
> retired in favour of this explicit command — drafts now happen **only when you ask**.
> The full design history and decision log live in
> [docs/PRD-speculative-agent.md](../../docs/PRD-speculative-agent.md).

Everything lives in this directory, ships with the repo, and is **inert until you invoke
the command**. No install step, no hooks.

---

## Quick start

1. **Prerequisites** — you've run the repo setup once (`bash scripts/setup.sh`, see the
   [main README](../../README.md)), so the model is downloaded and `jq`, `curl`, and the
   llama.cpp environment exist.
2. **Open Claude Code in this repo** and type:

   ```
   /gemma-draft summarize what scripts/run-server.sh does
   /gemma-draft docs/screenshot.png what error does this show?
   ```

   If the Gemma server is down, the first invocation auto-launches it in the background
   with your machine's tuned-optimal config (healthy in ~40 s) and tells you to retry.
3. **Watch usage** — run `/spec-stats` after a few drafts.

---

## What runs when

| Surface | Script | What it does |
|---|---|---|
| `/gemma-draft <task>` | `draft.sh` | Text task → one local Gemma draft, injected as advisory context for the big model to verify. Image path in the task → Gemma OCRs/describes it locally (**0 big-model image tokens**, up to 3 images) |
| `/spec-stats` | `stats.sh` | Usage report: drafts, local image reads, failures, estimated savings |
| (internal) | `gemma.sh` | Minimal OpenAI-API client to llama-server on :8080 (`--image` for vision) |
| (internal) | `ensure-server.sh` | Auto-launch the tuned-optimal server when down (single-flight, non-blocking) |

Every draft is advisory: the command's instructions tell the big model to **verify, keep
what's right, fix what's wrong** — nothing is ever applied blindly.

## Safety properties

- **Graceful degradation everywhere** — server down / image unreadable / jq error: the
  command reports it and the turn proceeds normally. No path can error your prompt.
- **Nothing automatic** — no hooks, no background speculation, no Read interception.
  The draft tier runs only when you type `/gemma-draft`.
- **Local only** — everything talks to `127.0.0.1:8080`; nothing leaves the machine.

---

## Configuration (all optional, env vars)

| Variable | Default | Meaning |
|---|---|---|
| `SPEC_AUTOSTART` | `1` | Auto-launch the optimal server when down (`0` = off) |
| `SPEC_DRAFT_MAX` | `512` text / `768` image | Max draft tokens |
| `SPEC_DRAFT_TIMEOUT` | `120` | Seconds to wait for a manual draft |
| `SPEC_HOST` / `SPEC_PORT` / `SPEC_MODEL` | `127.0.0.1` / `8080` / gemma-4-26b-a4b-qat | Where the draft tier lives |

Runtime state (`cache/`, `stats.jsonl`) is gitignored.

## Troubleshooting

- **"Server is down" every time?** `curl -s localhost:8080/health` — if down, check
  `/tmp/gemma4-server.log` (the auto-launcher writes there) or start manually with
  `bash scripts/start.sh`. `SPEC_AUTOSTART_DRYRUN=1 .claude/spec/ensure-server.sh` prints
  what would be launched without launching.
- **Drafts feel wrong?** That's expected sometimes — they're advisory and the big model
  supersedes them. A bad draft costs on the order of 100–200 input tokens.
- **Want a raw one-off Gemma call?** `.claude/spec/gemma.sh --system "be terse" "your prompt"`
  or `--image pic.png` for OCR.

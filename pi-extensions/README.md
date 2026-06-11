# pi-extensions — empowering local Gemma4

A set of [pi](https://pi.dev) extensions whose single purpose is to get the most
out of **Gemma4 20B QAT 4-bit running locally on a custom llama.cpp build**, with
a ~120k context window, fully offline. The motivation is energy/carbon: a small
local model on hardware that's already powered on, doing real work, if the harness
covers its weaknesses.

> These run inside pi with Gemma — there is no cloud model in the loop at runtime.
> The extensions are deterministic code; Gemma is the only intelligence. Anything
> the model must *write* (memories, summaries) is shaped by a fixed template.

## Design rules (every extension obeys these)

1. **KV-cache discipline.** llama.cpp reuses the KV cache only for the unchanged
   prompt *prefix*. System prompt, tool schemas, and session-start injections stay
   byte-stable for the whole session; anything dynamic is injected at the **tail**.
   Run pi on a single llama.cpp slot.
2. **Teaching errors.** A rejected tool call returns *what was wrong + a correct
   example*, never just "invalid input".
3. **Output caps.** Every tool result is capped (~50 lines) with an explicit
   continuation hint.
4. **Enforce > persuade.** Where a behavior must happen, the tool does it on
   Gemma's behalf rather than relying on a prompt rule.
5. **Terse schemas.** Tool descriptions are one line; few tools. Schemas are
   resent every request — they are a standing prefill tax.
6. **Templates over open prompts.** Anything Gemma writes gets a fill-in template.

The full spec, build order, and rationale live in
[`~/.pi/agent/extensions/PLAN.md`](../) (authored separately).

## Extensions

| # | Name | Status | What it does |
|---|------|--------|--------------|
| — | [`web-search`](web-search/) | ✅ done | Google search via stealth Playwright (bot-detection bypass). |
| 1 | [`verified-edits`](verified-edits/) | ✅ done | Auto-runs the cheapest checker after every edit; appends errors in-band. |
| 2 | [`symbols`](symbols/) | ✅ done | `get_symbols`/`find_symbol` outlines instead of whole-file reads. |
| 3 | [`loop-breaker`](loop-breaker/) | ✅ done | Nudge after 3 identical failing tool calls. |
| 4 | `plan` | ⬜ planned | External task-state checklist, re-injected at tail. |
| 5 | `semantic-memory` | ⬜ planned | Cross-session memory with automatic recall. |
| 6 | `operating-manual` | ⬜ planned | If-then rules in the system prefix + JIT nudges. |
| 7 | `stats` | ⬜ planned | Per-session token/energy accounting. |
| 8 | `fetch-page` | ⬜ planned | Readable-text page fetcher (closes the search→read loop). |

## Layout & dependencies

Each extension is a directory with `index.ts`, `test.mjs`, and `README.md`. A
single shared `package.json`/`node_modules` at this level serves all of them —
pi resolves `node_modules` up the directory tree.

```bash
cd pi-extensions && npm install
```

## Install into pi

```bash
./install.sh          # symlinks each extension into ~/.pi/agent/extensions/
```

Symlinks mean edits here are live in pi immediately (after a pi restart/reload).

## Testing

Extensions are plain TypeScript; Node 22 strips types natively, so tests run the
real extension code with no build step:

```bash
node --experimental-strip-types <ext>/test.mjs
./run-tests.sh        # runs every extension's test
```

# pi-extensions — empowering local Gemma4

A set of [pi](https://github.com/badlogic/pi-mono) extensions whose single purpose is
to get the most out of **Gemma 4 26B-A4B QAT 4-bit running locally on a custom
llama.cpp build**, with a ~120k context window, fully offline. The motivation is
energy/carbon: a small local model on hardware that's already powered on, doing real
work — if the harness covers its weaknesses. (Claude Code is used only to *author*
these extensions; there is no cloud model in the loop at runtime.)

> These run inside pi with Gemma — Gemma is the only intelligence. The extensions are
> deterministic code; anything the model must *write* (memories, summaries) is shaped
> by a fixed template, never an open-ended "summarize this."

## Design rules (every extension obeys these)

These six rules (R1–R6) are why the set works on a small model and stays cheap on a
laptop. Each extension's README notes which it leans on.

- **R1 — KV-cache discipline.** llama.cpp reuses the KV cache only for the unchanged
  prompt *prefix*. So everything static (system prompt, tool schemas, session-start
  injections) is byte-stable for the whole session; everything dynamic (recalled
  memories, plan state, nudges) is injected at the **tail** of the message list. That
  is the difference between paying prefill once and paying it every turn. Run pi on a
  single llama.cpp slot.
- **R2 — Teaching errors.** A rejected tool call never returns "invalid input" — it
  returns *what was wrong + one correct example*. A small model retries exactly as
  well as the error message instructs it to.
- **R3 — Output caps.** Every tool result is hard-capped (~50 lines / ~2 KB) with an
  explicit continuation hint (e.g. "412 more lines — call again with offset=50").
- **R4 — Enforce > persuade.** Where a behavior must happen, the tool *does it* on
  Gemma's behalf (auto-checks, redirects) rather than relying on a prompt rule the
  model's limited attention will drop.
- **R5 — Terse, model-optimal schemas.** A tool definition is resent every request — a
  standing prefill tax *and* the text the model reads to choose a tool. Optimize
  tokens-per-behavior, not prose: one imperative line, lead with the capability, keep
  only what routes behavior (when to use it vs. an alternative, the one gotcha). Few
  tools.
- **R6 — Templates over open prompts.** Anything Gemma must *write* (memories,
  snapshots) gets a fill-in template (Task:/Done:/Next:/Files:), never "summarize."

Per-extension design notes — the *why/how/results* for each — live in each
extension's own `README.md` (linked below). The engineering rationale for the set as
a whole is in [`docs/TECHNICAL.md` §15](../docs/TECHNICAL.md#15-the-harness-layer-pi-extensions).

## Extensions

Each covers one observed weakness of a 4-bit 26B-A4B MoE coding agent. Click through
for the full write-up; test counts are the standalone `test.mjs` assertions.

| Extension | What it does | Tests |
|-----------|--------------|------:|
| [`web-search`](web-search/) | Google search via stealth Playwright (bot-detection bypass) | live |
| [`fetch-page`](fetch-page/) | Readable-text page fetcher — closes the search→read loop | 18 |
| [`verified-edits`](verified-edits/) | Auto-runs the cheapest checker after every edit; appends errors in-band | 11 |
| [`symbols`](symbols/) | `get_symbols`/`find_symbol` outlines instead of whole-file reads | 30 |
| [`loop-breaker`](loop-breaker/) | Nudge after 3 identical failing tool calls | 15 |
| [`interrupt-notice`](interrupt-notice/) | Tell the model it was interrupted when you stop it mid-turn | 28 |
| [`compaction-notice`](compaction-notice/) | Tell the model its context was just compacted, so it re-reads instead of trusting stale memory | 26 |
| [`plan`](plan/) | External task-state checklist, re-injected at the tail | 25 |
| [`semantic-memory`](semantic-memory/) | Cross-session memory with automatic recall | 39 |
| [`operating-manual`](operating-manual/) | If-then rules in the system prefix + JIT nudges | 19 |
| [`stats`](stats/) | Per-session token/energy accounting | 26 |
| [`goal`](goal/) | Machine-checkable north-star that drives an unattended loop until `done_when` passes; the nudge anneals to a forced decision | 116 |
| [`grounding`](grounding/) | Tail-injects a reasoning protocol so Gemma verifies-or-flags at think time | 46 |
| [`pipe`](pipe/) | Chain slash-commands: `/pipe /goal … /plan …` expands into one ordered directive | 23 |
| [`toolsets`](toolsets/) | Context economy: gate situational tool groups to shrink the per-request tool tax | 19 |
| [`thinking-router`](thinking-router/) | Routes the per-turn thinking budget by input difficulty (engine-level energy lever) | 14 |
| [`advisor`](advisor/) | On-demand review by a stronger external agent that sees the whole session | 45 |

All complete and green via **`./run-tests.sh` — 519 checks** (the per-extension counts above
sum to 500; the rest are install-flow checks, and `web-search` uses a live test). `goal` is the only
one that *drives* the agent (`sendUserMessage` from `agent_end`) — validate that
re-engagement in a real pi run before relying on unattended loops.

## Engine-level energy levers

Two levers act on the inference engine itself rather than the prompt, so they live
partly outside the extensions:

- **Thinking-level routing** — ✅ done as the [`thinking-router`](thinking-router/)
  extension (pi exposes `setThinkingLevel()`, so it's pi code). Routes off/low/medium
  per turn by input difficulty; respects a manual `/thinking`.
- **Speculative decoding (MTP)** — ✅ done and wired into `start.sh` (`MTP=1`). Gemma 4's
  0.25 GB QAT draft head, lossless. Gain is regime-dependent: **+15–30 % at greedy/coding
  temperature, within measurement noise at temp 1.0**. `p_min` must stay 0 (positive values
  degenerate the output); EAGLE3 does **not** work on this build. Full study:
  [`docs/mtp-benchmark.md`](../docs/mtp-benchmark.md).
- **Constrained tool calls** — open: check whether the custom llama.cpp build exposes
  GBNF / JSON-schema enforcement for tool calls, and if so wire pi to it.

## Rejected (do not revisit without new evidence)

- **Turbovec / any ANN index** — alpha-stage, pays off only ≫100k vectors. The index is
  swappable behind `search()` if that day comes.
- **Key-value `store_fact`/`retrieve_fact`** — models can't query what they don't know
  they forgot; `semantic-memory` uses passive injection instead.
- **Embedding the codebase** — churns too fast; `symbols` outlines beat stale vectors.
- **Model routing (e2b/e4b)** — a single resident model; a second wouldn't stay in RAM.

## Developing an extension

The smallest working example is [`web-search/`](web-search/) — the reference for
`registerTool`, typebox params, and the tool-result/error shape. The pi extension API
types are in
`node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts`; the
events the set leans on are `session_start`, `before_agent_start`, `context` (returns a
modified message list — the tail-injection point), `tool_result`, `session_before_compact`,
`session_shutdown`, `thinking_level_select`, plus `registerTool`/`registerCommand`.

Each extension is a directory with `index.ts`, a `test.mjs`, and a `README.md`. A single
shared `package.json` / `node_modules` at this level serves all of them.

```bash
cd pi-extensions && npm install
```

## Install into pi

```bash
./setup.sh            # one-shot: npm install → link into pi → (optional) embeddings
```

`setup.sh` is the single entry point — everything is installable from this repo. It
installs deps, runs `install.sh`, and offers to pull + enable the embedding backend
(prompts before each side-effect; `--yes` runs non-interactively). Or do the steps
individually:

```bash
./install.sh                 # symlink each extension into ~/.pi/agent/extensions/
./install.sh --force         # also relink existing links AND stale real-dir copies (sync to repo)
./install.sh --prune         # drop our orphaned symlinks (extensions no longer in the repo)
```

`install.sh` symlinks each extension dir into `~/.pi/agent/extensions/`, **and** symlinks a
shared `node_modules` there too. By default it **adds what's missing and skips what already
exists** (symlinks track the repo live, so edits need no reinstall). When something is stale —
a real-directory copy left by an older install, or a moved repo — `--force`/`--relink`
replaces it with a fresh symlink; `--prune` removes *our* orphaned/broken links and never
touches extensions installed from elsewhere. Both flags also pass through `setup.sh`
(e.g. `./setup.sh --force`).

> **Why the shared node_modules matters.** pi loads extensions with
> `--preserve-symlinks`, so Node resolves each extension's `import`s from the
> symlink's location (`~/.pi/agent/extensions/<name>/`), **not** the real repo path.
> Without a `node_modules` at `~/.pi/agent/extensions/`, every import of
> `playwright`/`typebox`/`@earendil-works/*` fails with "Cannot find module." The
> shared symlink fixes all of them at once. (A real pi smoke test caught this — the
> plain-`node` import test missed it because plain Node follows symlinks to realpath.)

Symlinks mean edits here are live in pi immediately (after a pi restart/`/reload`).
Extensions auto-discover **globally** (all projects); for Gemma-only scope, put them
in a per-project `.pi/extensions/` instead.

## Testing

Extensions are plain TypeScript; Node 22 strips types natively, so tests run the
real extension code with no build step:

```bash
node --experimental-strip-types <ext>/test.mjs
./run-tests.sh        # runs every extension's test
```

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
| 4 | [`plan`](plan/) | ✅ done | External task-state checklist, re-injected at tail. |
| 5 | [`semantic-memory`](semantic-memory/) | ✅ done | Cross-session memory with automatic recall. |
| 6 | [`operating-manual`](operating-manual/) | ✅ done | If-then rules in the system prefix + JIT nudges. |
| 7 | [`stats`](stats/) | ✅ done | Per-session token/energy accounting. |
| 8 | [`fetch-page`](fetch-page/) | ✅ done | Readable-text page fetcher (closes the search→read loop). |
| 9 | [`goal`](goal/) | ✅ done | Machine-checkable north-star that drives an unattended loop until `done_when` passes. |
| 10 | [`grounding`](grounding/) | ✅ done | Tail-injects a reasoning protocol so Gemma verifies-or-flags at think time instead of hand-waving. |
| 11 | [`pipe`](pipe/) | ✅ done | Chain slash-commands: `/pipe /goal … /plan …` expands nested commands into one ordered directive. |
| + | [`thinking-router`](thinking-router/) | ✅ done | Routes the thinking budget per turn (engine-level energy lever, as pi code). |
| + | [`advisor`](advisor/) | ✅ done | On-demand review by an external agent (via tui-driver) that sees the whole session. |

## Layout & dependencies

Each extension is a directory with `index.ts`, `test.mjs`, and `README.md`. A
single shared `package.json`/`node_modules` at this level serves all of them.

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
> `playwright`/`typebox`/`@earendil-works/*` fails with "Cannot find module". The
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

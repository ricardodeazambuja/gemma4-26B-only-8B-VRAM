# pipe

Chain slash-commands with **nested-command expansion**:

```
/pipe /goal implement the results from /plan a python script that says hello world
```

runs as:

> 1. create a plan (`plan_set`) for: *a python script that says hello world*
> 2. set the goal (`goal_set`): *implement the results from* — use the result of step 1
> Then carry out the work so the final goal is achieved.

Unlike the others, `pipe` is an orchestration/UX convenience — not a model-weakness fix.

## Why a command, and not real piping

pi has **no command piping**: command handlers return `void` (no output to capture),
there's no `executeCommand` to invoke one command from another, and the `input` event is
read-only (can't rewrite what you typed). So `/a | /b` shell-style piping is impossible.

`/pipe` is the achievable substitute: a single command that parses the whole expression
itself and drives the agent with one ordered directive via `sendUserMessage` — the only
lever available. The model then calls the underlying tools (`plan_set`, `goal_set`, …) in
sequence. It's not literal output-capture, but it accomplishes the chaining.

## How it works

- **Parse** (`parsePipe`) — splits the expression on `/<known-command>` tokens into stages in
  textual order. Only known commands split it, so a stray `/tmp/foo` or `and/or` in an
  argument is never mistaken for a command.
- **Order** — execution is innermost-first: the outer command (`/goal`) references the inner
  one's result (`/plan`), so the inner runs as step 1. `buildDirective` reverses the textual
  order and numbers from 1, back-referencing each previous step.
- **Drive** — the handler `sendUserMessage`s the directive (always triggers a turn) and
  `notify`s a one-line `pipe → /plan ▸ /goal` summary. An expression with no known command
  returns a usage error and does **not** drive the agent.

## Teaching it new commands

`/pipe` chains the commands in `ACTIONS` (currently `plan`, `goal`). Add an entry mapping a
command name to the tool-using directive line, and it becomes pipeable — and a valid stage
boundary — automatically. (There's no runtime command registry in pi to discover them, so the
set is explicit by design.)

## Test

```bash
node --experimental-strip-types pipe/test.mjs
```

23 assertions: the parser (ordering, known-only splitting, path false-positives, degenerate
single-command), `describeStage` mapping, `buildDirective` (innermost-first, back-references,
carry-out line), and the command handler (drives the agent once on a valid pipe, stays silent
on an invalid one, survives a ui-less ctx).

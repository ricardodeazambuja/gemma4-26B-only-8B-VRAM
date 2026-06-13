# operating-manual

Help Gemma actually *use* the empowering tools and steer around its own foot-guns.
Telling a small model "you are weak at X" is self-knowledge it can't act on; what
works is concrete **if-then rules** (trigger → action).

## Two layers

1. **The manual in the stable system prefix** (`before_agent_start`). ≤600 bytes of
   imperative rules — "Before reading a whole file, call get_symbols first", "For a
   verifiable finish: goal_set(objective, done_when=…); goal_done verifies", "If a
   command fails twice the same way, change approach". It's appended to the system
   prompt every turn with identical
   text, so it costs prefill once per session and the KV cache holds (rule R1). It
   chains with `semantic-memory`'s system-prompt injection.

2. **Just-in-time nudges at the tail** (`tool_result`). A one-line tip appended to a
   result *only* when that result shows a known problem — currently a `grep` (or
   `find`/`ls`) that returns more than 80 lines gets "narrow the pattern or add a
   path filter". Fires at the moment of relevance, where a prefix rule would have
   faded after many turns. `loop-breaker` is the sharpest standalone case
   of this same idea.

The manual references the tools the other extensions provide (`get_symbols`,
`find_symbol`, `plan_set`, `goal_set`/`goal_done`, `remember`), so it's the glue that makes the set
cohere — but it degrades gracefully if some of those aren't installed (the rules
are just advice).

## Test

```bash
node --experimental-strip-types operating-manual/test.mjs
```

19 assertions: the manual's size/content/tone, the JIT nudge predicate (targeted at
grep/find only), and both hooks including byte-stability of the prefix injection.

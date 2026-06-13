# loop-breaker

Small models repeat a failing action verbatim — same command, same error — until
the context fills with identical failures and the session stalls. This extension
watches tool results and, after **3 identical failing calls in a row**, appends one
line telling Gemma to change approach. ~40 lines, no cost, and it rescues exactly
the sessions that would otherwise burn an hour of laptop GPU going in circles.

## How it works

- Hooks pi's `tool_result` event.
- Keys each call by `toolName + sha1(input)`. A failing call that matches the
  previous one increments a streak counter; **any different call or any success
  resets it**.
- When the streak reaches 3, appends `⟳ This exact … call has now failed N times …
  change your approach` to the result content.

The state machine (`makeTracker`) is exported and unit-tested independently of pi.

## Relationship to other extensions

This is the sharpest instance of the "just-in-time nudge" idea that
`operating-manual` generalizes. It stays a separate, always-on extension
because runaway retry loops are the single most expensive failure mode and deserve
their own guaranteed guard.

## Test

```bash
node --experimental-strip-types loop-breaker/test.mjs
```

15 assertions: streak counting, reset-on-different-call, reset-on-success,
args sensitivity, and the real handler appending the nudge at exactly the 3rd
failure.

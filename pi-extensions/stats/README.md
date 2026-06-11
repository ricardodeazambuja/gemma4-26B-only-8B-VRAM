# stats

Per-session token — and, with calibration, **energy** — accounting, so every other
extension can prove its value in numbers instead of vibes. PLAN.md item 7.

## What it records

Hooks `message_end` and pulls token usage off each assistant message into
`~/.pi/memory/<project>/stats.jsonl` (tagged with the session id). `readUsage` is
deliberately tolerant: it accepts `input`/`inputTokens`/`promptTokens`/`prompt_n`
(and the output/cache equivalents), so it works whether pi's build reports
OpenAI-style, camelCase, or llama.cpp-native field names.

## `/stats`

Prints the current session's:
- **prefill (input)** vs **decode (output)** token split — prefill is the energy hot
  spot, so watching it tells you whether the KV-cache discipline is paying off;
- **cache read** tokens and an estimated cache-hit % (`cacheRead / (cacheRead + input)`);
- **estimated Wh**, once a calibration exists.

## Energy calibration (Linux + Intel RAPL)

```bash
# while Gemma is generating a long output (note its tokens/sec from llama.cpp):
node --experimental-strip-types stats/calibrate.mjs --project "$PWD" --tps 18
```

It samples average package power from `/sys/class/powercap/intel-rapl:*/energy_uj`
over ~8 s and writes `J/token = watts / tokens_per_sec` to `calibration.json`.

This is a deliberate simplification — one rate for prefill and decode together — so
treat Wh as a **relative** measure for comparing sessions, not a power-meter
reading.

**Note:** `energy_uj` is frequently root-only (a side-channel mitigation). If
`calibrate.mjs` reports RAPL isn't readable, either grant read access
(`sudo chmod -R a+r /sys/class/powercap/intel-rapl:*/energy_uj`, or a udev rule) or
just write `{"jPerToken": <value>}` to `calibration.json` by hand. The token stats
work regardless; only the Wh estimate needs RAPL.

## Test

```bash
node --experimental-strip-types stats/test.mjs
```

26 assertions: tolerant usage extraction, aggregation + report formatting, RAPL
reading against a fake sysfs (incl. ignoring sub-domains), power sampling math,
calibration I/O, and the live `message_end` → `/stats` path.

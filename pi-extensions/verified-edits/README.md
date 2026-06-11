# verified-edits

After Gemma edits or writes a source file, automatically run the cheapest
available checker for that language and append any error to the **same** tool
result. Small models are bad at self-verification but excellent at reacting to a
concrete error — so we give them ground truth in-band instead of asking them to
re-read their own work.

## Why

The single worst habit of a small coding model is declaring broken code done.
This extension makes that almost impossible: a syntax error shows up as part of
the write's result, in the same turn, phrased as an instruction to fix it.

## How it works

- Hooks pi's `tool_result` event for the built-in `edit` and `write` tools.
- Resolves the file path against `ctx.cwd`, picks a checker by extension, runs it
  with a 5 s timeout, and on failure appends a `⚠ CHECK FAILED (label) …` note
  (capped at 8 lines) to the result content.
- **Never blocks or errors the turn**: if the checker is missing, can't spawn, or
  times out, it stays completely silent. A clean file adds zero output.

## Checkers (cheapest-first, first installed one wins)

| Ext | Checker |
|-----|---------|
| `.py` | `ruff check` → `python3 -m py_compile` |
| `.ts` `.tsx` | `tsc --noEmit --allowJs --skipLibCheck` (or `npx tsc`) |
| `.js` `.mjs` `.cjs` | `node --check` |
| `.rs` | `rustc --emit metadata` |
| `.go` | `gofmt -e` |
| `.json` | `JSON.parse` via node |
| `.sh` | `bash -n` |

Unknown extensions are ignored. Adding a language = one line in `CHECKERS`.

## Test

```bash
node --experimental-strip-types verified-edits/test.mjs
```

Drives the real extension through a fake pi against temp files with and without
syntax errors (Python, JSON, the skip paths). 11 assertions.

# symbols

Give Gemma a **code outline** instead of making it read an entire file to find one
signature. This is the biggest recurring prefill saving in the set: ~30 lines
where a full read would be hundreds, every time the model needs to locate
something. PLAN.md item 2.

## Tools

- **`get_symbols(path)`** — function/class/type signatures + imports, with line
  numbers. One line per symbol, capped at 60.
- **`find_symbol(name)`** — where a symbol is defined, project-wide, as
  `file:line` rows. Crawls the tree (skipping `node_modules`, `.git`, build dirs),
  pre-filters with a substring check before regex.

## Enforcement (rule R4: enforce > persuade)

A `tool_call` hook intercepts **full reads of large code files** (> ~8 KB) and
returns the outline instead, with a note on how to get the real content. Gemma
doesn't have to *remember* to call `get_symbols` — the harness substitutes it.
Escape hatches that pass through untouched:

- a ranged read (`offset`/`limit` set) — the model explicitly wants specific lines;
- small files, non-code files, or files with no extractable symbols.

## Extraction

Dependency-free **line-regex** extractor (`extract.ts`) covering Python, JS/TS,
Rust, Go, and C/C++. ctags isn't always installed and tree-sitter needs native or
WASM builds; regex is "good enough to navigate" and has zero build risk. `index.ts`
only imports `extractSymbols`/`langForExt`, so swapping in tree-sitter later is a
one-file change.

Trade-off: regex can miss exotic declarations (multi-line signatures, macros) and
isn't type-aware. It will not produce false *navigation* — every reported line
really contains that declaration — but it may omit some. Good enough for an outline.

## Known limitation

`get_symbols`/`find_symbol` resolve paths against `process.cwd()` because pi's tool
`execute(id, params, signal)` signature doesn't pass `ctx.cwd`. pi runs tools from
the session's working directory, so this matches in normal use. The read-redirect
hook *does* get `ctx.cwd` and uses it correctly.

## Test

```bash
node --experimental-strip-types symbols/test.mjs
```

30 assertions: extraction across 5 languages, both tools, the redirect hook, and
all four pass-through escape hatches.

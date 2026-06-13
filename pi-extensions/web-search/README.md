# web-search — give Gemma live Google results

Registers a `web_search` tool so the local model can pull **current/factual info**
it doesn't carry. A 26B model has a thin long tail; without search it confidently
answers from stale recollection. Drives a headless Chromium (Playwright) through a
real Google search and returns the top 5 organic results as title / link / snippet.
Pairs with [`fetch-page`](../fetch-page/) to close the **search → read** loop.

This is also the **reference extension** for the set — the smallest working example
of `registerTool`, typebox params, and the tool-result/error shape the others copy.

## Tool

`web_search(query)` — one string parameter. Returns up to 5 results, each as three
lines (title, URL, snippet) joined by `---`; `details.count` carries the total found
before the slice.

## How it works

Every call launches a headless Chromium, runs one Google query, and closes the
browser (no persistent session). Two problems dominate, and the code targets each:

- **Not getting flagged as a bot.** Google blocks obvious automation. So: launch with
  `--disable-blink-features=AutomationControlled`, hide `navigator.webdriver` via an
  init script, set an `en-US` locale and a 1366×768 viewport, and build the User-Agent
  from the bundled Chromium's own `browser.version()` **minus the "Headless" marker**
  (a stale hardcoded UA is itself a tell). If Google still serves its `/sorry/`
  interstitial, the tool returns a clear error ("blocked … try again later", `isError`)
  instead of garbage — a teaching error (rule R2), not a crash.
- **Surviving Google's churning HTML.** Class names change constantly, so extraction
  anchors on the **stable structure**: every organic result is an `<h3>` tied to a
  link. For each `<h3>` it finds the associated `http` anchor (skipping `google.com`
  links, de-duping by href), then walks up to 8 parents looking for a clamped-text
  snippet `div` — stopping before it spans into the next result.

## Caveats

- **Requires Playwright + a Chromium download** (`npx playwright install chromium`),
  installed once by the set's [`setup.sh`](../setup.sh) / `npm install`.
- **Scrapes live Google HTML.** A big layout change can degrade snippets; the
  structural `<h3>` anchoring is what keeps it from breaking outright. Bot-detection
  can still trip on unusual traffic — surfaced as the explicit error above.
- **Top 5, no pagination.** It's a "what's out there" probe; hand a promising URL to
  `fetch-page` for the full readable text.

## Test

```bash
node --experimental-strip-types web-search/test-final.mjs "your query"
```

`test-final.mjs` mirrors `execute()` exactly for standalone verification: it prints
the result `count` and the formatted results, and exits non-zero printing `BLOCKED`
if Google serves the bot page. Live-verified against Google and Wikipedia.

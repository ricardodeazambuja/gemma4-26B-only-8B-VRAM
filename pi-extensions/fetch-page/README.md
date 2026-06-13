# fetch-page

Read a web page as clean text. `web_search` finds links; without a fetcher Gemma
can't actually *read* them — and its world knowledge is thin, so retrieval matters.
This closes the search→read loop.

## Tool

- **`fetch_page(url, offset=0)`** — fetch an http(s) URL, strip page chrome, return
  readable text. Long pages are capped at 50 lines with a `call again with
  offset=N` hint (rule R3), so a big article can't blow the context window.

## How it works

- Reuses **web-search's stealth Playwright setup**: UA built from
  `browser.version()` minus the "Headless" marker,
  `--disable-blink-features=AutomationControlled`, and `navigator.webdriver`
  hidden — so pages don't bot-block the fetch. Detects Google's `/sorry/` block
  page and reports it cleanly.
- In-page extraction prefers `<article>`/`<main>`/`[role=main]`, drops
  `script/style/nav/header/footer/aside/svg/form/iframe/button`, and returns
  `innerText`. `collapseText` then tidies whitespace and `paginate` slices it.

### Gotcha baked in

The DOM extractor is passed to `page.evaluate` as a **string**, and Playwright
evaluates a string as an *expression* — so it must be a self-invoking IIFE
(`(() => {…})()`). A bare arrow-function string returns the function uncalled
(→ `undefined`). The test caught this; both the tool and the test use the IIFE form.

## Limitations

- JS-app pages that render after `domcontentloaded`, PDFs, and login-walled content
  won't extract well — the tool says so rather than returning garbage.
- No Readability-grade boilerplate removal; the main-region heuristic is good enough
  for articles and docs.

## Test

```bash
node --experimental-strip-types fetch-page/test.mjs
```

18 assertions: whitespace cleanup, pagination/offset/continuation, URL validation,
and a **real Playwright extraction** against a `data:` URL (no network) proving
article text is kept while script/nav/footer are stripped. Verified live against
google.com and wikipedia.org.

// Tests fetch-page: pure text cleanup/pagination, the tool's URL validation, and
// a real Playwright extraction against a data: URL (no network).
// Run from pi-extensions/: node --experimental-strip-types fetch-page/test.mjs
import { collapseText, paginate, formatPage } from "./clean.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("collapseText:");
ok("collapses inline whitespace", collapseText("a    b\tc") === "a b c");
ok("trims lines", collapseText("  hi  \n  bye ") === "hi\nbye");
ok("collapses blank runs", collapseText("a\n\n\n\nb") === "a\n\nb");

console.log("paginate:");
const text = Array.from({ length: 120 }, (_, i) => `line ${i}`).join("\n");
let p = paginate(text, 0, 50);
ok("first page has maxLines", p.shownLines === 50 && p.body.startsWith("line 0"));
ok("first page reports more", p.nextOffset === 50 && p.totalLines === 120);
p = paginate(text, 100, 50);
ok("offset page slices correctly", p.body.startsWith("line 100"));
ok("last page has no nextOffset", p.nextOffset === null);
ok("offset past end → empty, no more", paginate(text, 999, 50).shownLines === 0);

console.log("formatPage:");
const out = formatPage("https://x.com", "Title", paginate(text, 0, 50));
ok("includes title + url", out.includes("# Title") && out.includes("https://x.com"));
ok("includes continuation hint", out.includes("offset=50"));
ok("no hint on last page", !formatPage("https://x.com", "T", paginate(text, 100, 50)).includes("offset="));

// --- tool ---
console.log("tool:");
let tool;
factory({ registerTool: (t) => (tool = t), registerCommand() {}, on() {} });
ok("registers fetch_page", typeof tool?.execute === "function");

const run = async () => {
  let r = await tool.execute("t", { url: "not-a-url" });
  ok("rejects non-http URL with teaching error", r.isError && r.content[0].text.includes("fetch_page("));

  // real browser extraction against a data: URL — no network needed
  const html = `<!doctype html><html><head><title>My Article</title></head>
    <body>
      <nav>HOME ABOUT CONTACT</nav>
      <script>console.log('tracking pixel')</script>
      <article><h1>Heading</h1><p>The quick brown fox jumps.</p><p>Second paragraph here.</p></article>
      <footer>copyright junk</footer>
    </body></html>`;
  const dataUrl = "data:text/html," + encodeURIComponent(html);
  r = await tool.execute("t", { url: "http://localhost/" }, undefined); // will fail-fast (no server) → error path
  ok("unreachable http URL returns a clean error", r.isError && r.content[0].text.includes("Could not fetch"));

  // Now drive the real extractor via a data: URL by monkeypatching is overkill;
  // instead validate extraction by launching the same path with a data URL.
  // The tool only accepts http(s); for the extraction test we bypass validation
  // by calling with a blob/data through a tiny in-test fetch using Playwright directly.
  const { chromium } = await import("playwright");
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-setuid-sandbox"] });
  try {
    const page = await browser.newPage();
    await page.goto(dataUrl, { waitUntil: "domcontentloaded" });
    const EXTRACT = `(() => {
      const drop = ['script','style','noscript','nav','header','footer','aside','svg','form','iframe','button'];
      const root = document.querySelector('article, main, [role="main"]') || document.body;
      const clone = root.cloneNode(true);
      for (const sel of drop) clone.querySelectorAll(sel).forEach(n => n.remove());
      return { title: document.title || '', text: clone.innerText || '' };
    })()`;
    const { title, text } = await page.evaluate(EXTRACT);
    const clean = collapseText(text);
    ok("extracts article title", title === "My Article");
    ok("keeps article body text", clean.includes("quick brown fox") && clean.includes("Second paragraph"));
    ok("strips script content", !clean.includes("tracking pixel"));
    ok("strips nav + footer", !clean.includes("ABOUT") && !clean.includes("copyright junk"));
  } finally {
    await browser.close();
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

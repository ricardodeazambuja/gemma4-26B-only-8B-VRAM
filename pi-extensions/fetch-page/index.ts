import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { chromium } from "playwright";
import { collapseText, paginate, formatPage } from "./clean.ts";

// fetch-page — read a web page as clean text. Closes the search→read loop: Gemma's
// world knowledge is thin, so it needs to actually read the links web_search finds.
// Reuses web-search's stealth Playwright setup (UA from browser.version() minus the
// "Headless" marker, automation flags hidden) so pages don't bot-block it.
// PLAN.md item 8.

const MAX_LINES = 50;        // R3 output cap
const NAV_TIMEOUT = 30000;

// The DOM extractor runs inside the page. Strips chrome, prefers the main content
// region, returns its innerText. An IIFE string: Playwright evaluates a string as
// an expression, so it must self-invoke (a bare arrow would return uncalled).
const EXTRACT = `(() => {
  const drop = ['script','style','noscript','nav','header','footer','aside','svg','form','iframe','button'];
  const root = document.querySelector('article, main, [role="main"]') || document.body;
  if (!root) return { title: document.title || '', text: '' };
  const clone = root.cloneNode(true);
  for (const sel of drop) clone.querySelectorAll(sel).forEach(n => n.remove());
  return { title: document.title || '', text: clone.innerText || '' };
})()`;

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "fetch_page",
    label: "Fetch Page",
    description: "Fetch a URL and return its readable text (chrome stripped). Use offset to page through long pages.",
    parameters: Type.Object({
      url: Type.String({ description: "The URL to fetch (http/https)" }),
      offset: Type.Optional(Type.Number({ description: "Line offset for continuation (default 0)" })),
    }),
    async execute(_id, params, signal) {
      const url = (params.url || "").trim();
      if (!/^https?:\/\//i.test(url)) {
        return err(`fetch_page needs an http(s) URL, e.g. fetch_page(url="https://example.com/article").`);
      }
      const offset = Math.max(0, params.offset || 0);

      const browser = await chromium.launch({
        headless: true,
        args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-blink-features=AutomationControlled"],
      });
      try {
        const version = browser.version();
        const userAgent = `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`;
        const context = await browser.newContext({ userAgent, locale: "en-US", viewport: { width: 1366, height: 768 } });
        const page = await context.newPage();
        await page.addInitScript(() => {
          Object.defineProperty(navigator, "webdriver", { get: () => undefined });
        });
        signal?.addEventListener("abort", () => { browser.close().catch(() => {}); }, { once: true });

        await page.goto(url, { waitUntil: "domcontentloaded", timeout: NAV_TIMEOUT });
        if (page.url().includes("/sorry/")) {
          return err("The site returned a bot-detection page (unusual traffic). Try a different source.");
        }
        const { title, text } = (await page.evaluate(EXTRACT)) as { title: string; text: string };
        const clean = collapseText(text);
        if (!clean) {
          return { content: [{ type: "text", text: `${url}\n\n(No readable text extracted — the page may be a JS app, a PDF, or empty.)` }] };
        }
        const p = paginate(clean, offset, MAX_LINES);
        return { content: [{ type: "text", text: formatPage(url, title, p) }], details: { totalLines: p.totalLines, nextOffset: p.nextOffset } };
      } catch (e) {
        return err(`Could not fetch ${url}: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        await browser.close();
      }
    },
  });
}

function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}

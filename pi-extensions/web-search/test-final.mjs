// Mirrors index.ts execute() exactly, for standalone verification.
import { chromium } from "playwright";

const query = process.argv[2] || "playwright browser automation";

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

  await page.goto(`https://www.google.com/search?q=${encodeURIComponent(query)}&hl=en`, {
    waitUntil: "domcontentloaded",
    timeout: 30000,
  });

  if (page.url().includes("/sorry/")) {
    console.log("BLOCKED by Google bot detection");
    process.exit(1);
  }

  const results = await page.evaluate(() => {
    const seen = new Set();
    const out = [];
    for (const h3 of Array.from(document.querySelectorAll("h3"))) {
      const a = h3.closest("a") || h3.querySelector("a") || h3.parentElement?.closest("a");
      if (!a || !a.href || !a.href.startsWith("http")) continue;
      if (a.href.includes("google.com")) continue;
      if (seen.has(a.href)) continue;
      seen.add(a.href);

      let snippet = "";
      let el = a.parentElement;
      for (let i = 0; i < 8 && el; i++, el = el.parentElement) {
        if (el.querySelectorAll("h3").length > 1) break;
        const clamp = el.querySelector('div[style*="-webkit-line-clamp"], div[data-sncf]');
        if (clamp) {
          snippet = clamp.textContent || "";
          break;
        }
      }

      out.push({ title: h3.textContent || "", link: a.href, snippet });
    }
    return out;
  });

  const formattedResults = results.slice(0, 5).map((r) => `${r.title}\n${r.link}\n${r.snippet}\n`).join("\n---\n");
  console.log("count:", results.length);
  console.log(formattedResults || "No results found.");
} finally {
  await browser.close();
}

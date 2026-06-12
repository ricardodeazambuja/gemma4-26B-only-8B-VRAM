import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { chromium } from "playwright";

export default async function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "web_search",
    label: "Web Search",
    description: "Search the web for current or factual info.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
    }),
    async execute(toolCallId, params, signal) {
      const browser = await chromium.launch({
        headless: true,
        args: [
          "--no-sandbox",
          "--disable-setuid-sandbox",
          // Hides navigator.webdriver and other automation signals Google checks
          "--disable-blink-features=AutomationControlled",
        ],
      });

      try {
        // A stale hardcoded UA gets flagged as a bot; build one matching the
        // bundled Chromium version, minus the "Headless" marker.
        const version = browser.version();
        const userAgent = `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`;

        const context = await browser.newContext({
          userAgent,
          locale: "en-US",
          viewport: { width: 1366, height: 768 },
        });
        const page = await context.newPage();
        await page.addInitScript(() => {
          Object.defineProperty(navigator, "webdriver", { get: () => undefined });
        });

        await page.goto(
          `https://www.google.com/search?q=${encodeURIComponent(params.query)}&hl=en`,
          { waitUntil: "domcontentloaded", timeout: 30000 },
        );

        if (page.url().includes("/sorry/")) {
          return {
            content: [{
              type: "text",
              text: "Google blocked the request with its bot-detection page (unusual traffic). Try again later.",
            }],
            isError: true,
          };
        }

        const results = await page.evaluate(() => {
          // Google's classnames churn constantly; anchor on the stable
          // structure instead: every organic result is an <h3> tied to a link.
          const seen = new Set();
          const out = [];
          for (const h3 of Array.from(document.querySelectorAll("h3"))) {
            const a =
              h3.closest("a") ||
              h3.querySelector("a") ||
              h3.parentElement?.closest("a");
            if (!a || !a.href || !a.href.startsWith("http")) continue;
            if (a.href.includes("google.com")) continue;
            if (seen.has(a.href)) continue;
            seen.add(a.href);

            // Snippet depth varies by layout: walk up from the anchor until a
            // clamped-text div appears, but stop before spanning other results.
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

        const formattedResults = results
          .slice(0, 5)
          .map((r) => `${r.title}\n${r.link}\n${r.snippet}\n`)
          .join("\n---\n");

        return {
          content: [{
            type: "text",
            text: formattedResults || "No results found.",
          }],
          details: { count: results.length },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Error during search: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      } finally {
        await browser.close();
      }
    },
  });
}

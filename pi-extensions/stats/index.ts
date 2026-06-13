import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { memoryDir, statsPath, readUsage, appendStat, loadStats, aggregate, formatReport, type StatRecord } from "./core.ts";
import { loadCalibration } from "./energy.ts";

// stats — per-session token (and, with calibration, energy) accounting so every
// other optimization can prove its value in numbers. Records each assistant
// message's usage; `/stats` prints the session's prefill/decode split, cache-hit
// estimate, and estimated Wh.

export default function (pi: ExtensionAPI) {
  let dir = "";
  let session = "session";

  pi.on("session_start", async (_event, ctx) => {
    dir = memoryDir(ctx.cwd);
    session = ctx.sessionManager.getSessionId() || "session";
  });

  // Record usage as each assistant message completes.
  pi.on("message_end", async (event, ctx) => {
    if (!dir) return;
    const usage = readUsage(event.message);
    if (!usage) return; // user/tool messages, or a build that doesn't expose usage
    const ctxUsage = ctx.getContextUsage?.();
    const rec: StatRecord = {
      ts: new Date().toISOString(),
      session,
      input: usage.input,
      output: usage.output,
      cacheRead: usage.cacheRead,
      contextTokens: ctxUsage?.tokens ?? 0,
    };
    try { appendStat(dir, rec); } catch {}
  });

  pi.registerCommand("stats", {
    description: "Show this session's token usage and estimated energy",
    handler: async (_args, ctx) => {
      const d = dir || memoryDir(ctx.cwd);
      const records = loadStats(d, session);
      const report = formatReport(aggregate(records), loadCalibration(d));
      ctx.ui.notify(report, "info");
    },
  });
}

export { statsPath };

// Pure stats logic: extract token usage from a pi assistant message (tolerant to
// the exact field names), persist per-message records, aggregate per session, and
// format the /stats report.

import { homedir } from "node:os";
import { join } from "node:path";
import { existsSync, readFileSync, appendFileSync, mkdirSync } from "node:fs";
import { type Calibration, joulesForTokens, joulesToWh } from "./energy.ts";

export function memoryDir(cwd: string): string {
  const slug = cwd.replace(/^\/+/, "").replace(/[^a-zA-Z0-9]+/g, "-").replace(/-+$/g, "") || "root";
  return join(homedir(), ".pi", "memory", slug);
}
export function statsPath(dir: string): string { return join(dir, "stats.jsonl"); }

export interface UsageTokens { input: number; output: number; cacheRead: number; }
export interface StatRecord extends UsageTokens { ts: string; session: string; contextTokens: number; }

/** Pull token counts out of an assistant message regardless of which field-name
 * convention the provider/pi build uses. Returns null if nothing token-like found. */
export function readUsage(message: unknown): UsageTokens | null {
  if (!message || typeof message !== "object") return null;
  const m = message as Record<string, any>;
  const u = m.usage ?? m.tokenUsage ?? m;
  if (!u || typeof u !== "object") return null;
  const pick = (...names: string[]) => {
    for (const n of names) { const v = u[n]; if (typeof v === "number" && Number.isFinite(v)) return v; }
    return 0;
  };
  const input = pick("input", "inputTokens", "promptTokens", "prompt_tokens", "prompt_n");
  const output = pick("output", "outputTokens", "completionTokens", "completion_tokens", "predicted_n");
  const cacheRead = pick("cacheRead", "cache_read", "cachedTokens", "cached_tokens", "cacheReadTokens");
  if (input === 0 && output === 0) return null; // nothing useful
  return { input, output, cacheRead };
}

export function appendStat(dir: string, rec: StatRecord): void {
  mkdirSync(dir, { recursive: true });
  appendFileSync(statsPath(dir), JSON.stringify(rec) + "\n");
}

export function loadStats(dir: string, session?: string): StatRecord[] {
  const p = statsPath(dir);
  if (!existsSync(p)) return [];
  const out: StatRecord[] = [];
  for (const line of readFileSync(p, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try { const r = JSON.parse(line); if (!session || r.session === session) out.push(r); } catch {}
  }
  return out;
}

export interface Aggregate {
  messages: number;
  prefill: number;   // input tokens (what gets re-processed; the energy hot spot)
  decode: number;    // output tokens
  cacheRead: number;
  cacheHitPct: number | null; // cacheRead / (cacheRead + input)
}

export function aggregate(records: StatRecord[]): Aggregate {
  let prefill = 0, decode = 0, cacheRead = 0;
  for (const r of records) { prefill += r.input; decode += r.output; cacheRead += r.cacheRead; }
  const denom = cacheRead + prefill;
  return {
    messages: records.length,
    prefill, decode, cacheRead,
    cacheHitPct: denom > 0 ? (cacheRead / denom) * 100 : null,
  };
}

function fmt(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

export function formatReport(agg: Aggregate, cal: Calibration | null): string {
  if (agg.messages === 0) return "stats: no model responses recorded this session yet.";
  const lines = [
    `📊 Session stats (${agg.messages} responses)`,
    `  prefill (input):  ${fmt(agg.prefill)} tokens`,
    `  decode (output):  ${fmt(agg.decode)} tokens`,
    `  cache read:       ${fmt(agg.cacheRead)} tokens` + (agg.cacheHitPct !== null ? `  (~${agg.cacheHitPct.toFixed(0)}% of input served from cache)` : ""),
  ];
  if (cal) {
    const totalTok = agg.prefill + agg.decode;
    const wh = joulesToWh(joulesForTokens(totalTok, cal.jPerToken));
    lines.push(`  est. energy:      ${wh.toFixed(3)} Wh  (@ ${cal.jPerToken.toFixed(3)} J/token)`);
  } else {
    lines.push(`  est. energy:      (run calibrate.mjs to enable Wh estimates)`);
  }
  return lines.join("\n");
}

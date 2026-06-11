// Tests stats: usage extraction, aggregation, report formatting, RAPL reading
// (with a fake sysfs), power sampling, calibration I/O, and the live hooks.
// Run: node --experimental-strip-types stats/test.mjs
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readUsage, aggregate, formatReport, appendStat, loadStats, memoryDir } from "./core.ts";
import { readRaplEnergyUj, sampleAveragePowerW, loadCalibration, saveCalibration, joulesToWh, joulesForTokens } from "./energy.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("readUsage (tolerant to field names):");
ok("OpenAI-ish names", JSON.stringify(readUsage({ usage: { promptTokens: 100, completionTokens: 20 } })) === JSON.stringify({ input: 100, output: 20, cacheRead: 0 }));
ok("llama.cpp names", readUsage({ usage: { prompt_n: 50, predicted_n: 10 } }).input === 50);
ok("camelCase + cache", readUsage({ usage: { inputTokens: 5, outputTokens: 5, cacheRead: 3 } }).cacheRead === 3);
ok("usage at top level", readUsage({ input: 7, output: 2 }).input === 7);
ok("no usage → null", readUsage({ role: "user", content: "hi" }) === null);
ok("non-object → null", readUsage(null) === null);

console.log("aggregate + report:");
const recs = [
  { ts: "", session: "s", input: 1000, output: 200, cacheRead: 800, contextTokens: 0 },
  { ts: "", session: "s", input: 500, output: 100, cacheRead: 400, contextTokens: 0 },
];
const agg = aggregate(recs);
ok("sums prefill/decode/cache", agg.prefill === 1500 && agg.decode === 300 && agg.cacheRead === 1200);
ok("cache-hit pct computed", Math.abs(agg.cacheHitPct - (1200 / 2700) * 100) < 1e-6);
ok("empty aggregate → null pct", aggregate([]).cacheHitPct === null);
let report = formatReport(agg, null);
ok("report shows prefill/decode", report.includes("prefill") && report.includes("decode"));
ok("no calibration → Wh hint", report.includes("calibrate.mjs"));
report = formatReport(agg, { jPerToken: 2 });
ok("with calibration → Wh shown", report.includes("Wh") && report.includes("J/token"));
ok("empty session report", formatReport(aggregate([]), null).includes("no model responses"));

console.log("energy math:");
ok("joulesForTokens", joulesForTokens(100, 2) === 200);
ok("joulesToWh", Math.abs(joulesToWh(3600) - 1) < 1e-9);

console.log("RAPL (fake sysfs):");
const base = mkdtempSync(join(tmpdir(), "rapl-"));
mkdirSync(join(base, "intel-rapl:0"));
writeFileSync(join(base, "intel-rapl:0", "energy_uj"), "1000000");
mkdirSync(join(base, "intel-rapl:1"));
writeFileSync(join(base, "intel-rapl:1", "energy_uj"), "500000");
mkdirSync(join(base, "intel-rapl:0:0")); // sub-domain, must be ignored
writeFileSync(join(base, "intel-rapl:0:0", "energy_uj"), "999999999");
ok("sums package domains, ignores sub-domains", readRaplEnergyUj(base) === 1500000);
ok("missing base → null", readRaplEnergyUj(join(base, "nope")) === null);

let calls = 0;
const fakeSampler = () => (calls++ === 0 ? 1_000_000 : 3_000_000); // +2 J over window
const watts = await sampleAveragePowerW(1000, fakeSampler, async () => {});
ok("sampleAveragePowerW computes W", Math.abs(watts - 2) < 1e-6); // 2 J / 1 s = 2 W
ok("power null when RAPL absent", (await sampleAveragePowerW(10, () => null, async () => {})) === null);

console.log("calibration I/O:");
const cdir = mkdtempSync(join(tmpdir(), "cal-"));
ok("no calibration → null", loadCalibration(cdir) === null);
saveCalibration(cdir, { jPerToken: 1.5 });
ok("save+load roundtrips", loadCalibration(cdir).jPerToken === 1.5);

console.log("hooks + command:");
const hooks = {}; let cmd;
factory({ on: (e, h) => (hooks[e] = h), registerTool() {}, registerCommand: (n, o) => { if (n === "stats") cmd = o; } });
ok("registers message_end + session_start", !!hooks.message_end && !!hooks.session_start);
ok("registers /stats command", typeof cmd?.handler === "function");

const pdir = mkdtempSync(join(tmpdir(), "statsx-"));
const ctx = {
  cwd: pdir,
  sessionManager: { getSessionId: () => "sess1", getSessionDir: () => pdir },
  getContextUsage: () => ({ tokens: 1234, contextWindow: 120000, percent: 1 }),
  ui: { notify(msg) { ctx._last = msg; } },
};

const run = async () => {
  await hooks.session_start({}, ctx);
  await hooks.message_end({ message: { role: "assistant", usage: { input: 900, output: 120, cacheRead: 700 } } }, ctx);
  await hooks.message_end({ message: { role: "user", content: "hi" } }, ctx); // no usage → skipped
  const saved = loadStats(memoryDir(pdir), "sess1");
  ok("message_end records usage", saved.length === 1 && saved[0].input === 900);
  ok("records context tokens", saved[0].contextTokens === 1234);

  await cmd.handler("", ctx);
  ok("/stats notifies a report with the split", ctx._last.includes("prefill") && ctx._last.includes("900"));

  rmSync(base, { recursive: true, force: true });
  rmSync(cdir, { recursive: true, force: true });
  rmSync(pdir, { recursive: true, force: true });
  rmSync(memoryDir(pdir), { recursive: true, force: true });
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

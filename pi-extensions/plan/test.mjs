// Tests the plan extension: pure renderers + tools + context/compaction hooks.
// Run: node --experimental-strip-types plan/test.mjs
import { mkdtempSync, rmSync, existsSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";
import factory, { renderChecklist, buildSnapshot, memoryDir } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("renderers:");
ok("empty plan renders nothing", renderChecklist([]) === "");
const steps = [{ text: "read config", done: true }, { text: "fix bug", done: false }];
const cl = renderChecklist(steps);
ok("checklist marks done/undone", cl.includes("[x] 1. read config") && cl.includes("[ ] 2. fix bug"));
ok("checklist names current step", cl.includes("Current step: fix bug"));
ok("all-done defers completion to goal_done", renderChecklist([{ text: "x", done: true }]).includes("All steps complete") && renderChecklist([{ text: "x", done: true }]).includes("goal_done"));

console.log("snapshot:");
const snap = buildSnapshot(steps, ["a.py", "b.ts"]);
ok("snapshot has Task/Done/Next/Files", ["Task:", "Done:", "Next:", "Files touched:"].every((k) => snap.includes(k)));
ok("snapshot lists touched files", snap.includes("a.py, b.ts"));

console.log("memoryDir:");
ok("slug is filesystem-safe", /^[A-Za-z0-9/_.-]+$/.test(memoryDir("/home/x/My Proj!")));
ok("under ~/.pi/memory", memoryDir("/home/x/proj").startsWith(join(homedir(), ".pi", "memory")));

// --- tools + hooks ---
const tools = {};
const hooks = {};
const fakePi = {
  registerTool: (t) => { tools[t.name] = t; },
  registerCommand() {},
  on: (ev, h) => { hooks[ev] = h; },
};
factory(fakePi);

console.log("registration:");
ok("registers plan_set/check/show", !!tools.plan_set && !!tools.plan_check && !!tools.plan_show);
ok("registers context + compact hooks", !!hooks.context && !!hooks.session_before_compact && !!hooks.session_start);

const dir = mkdtempSync(join(tmpdir(), "plan-"));
const fakeCtx = {
  cwd: dir,
  sessionManager: { getSessionDir: () => dir, getSessionId: () => "testsess" },
};

const run = async () => {
  await hooks.session_start({}, fakeCtx);

  console.log("plan_set validation:");
  let r = await tools.plan_set.execute("t", { steps: [] });
  ok("empty steps → teaching error", r.isError && r.content[0].text.includes("plan_set("));
  r = await tools.plan_set.execute("t", { steps: Array(11).fill("x") });
  ok("too many steps → error", r.isError && r.content[0].text.includes("Too many"));
  r = await tools.plan_set.execute("t", { steps: ["a".repeat(90)] });
  ok("too-long step → error", r.isError && r.content[0].text.includes("under 80"));

  console.log("plan flow:");
  r = await tools.plan_set.execute("t", { steps: ["read config", "fix bug", "run tests"] });
  ok("valid plan_set succeeds", !r.isError && r.content[0].text.includes("Plan set (3 steps)"));
  ok("plan persisted to session dir", existsSync(join(dir, "plan-testsess.json")));

  r = await tools.plan_check.execute("t", { step: 1 });
  ok("plan_check marks step", r.content[0].text.includes("[x] 1. read config"));
  r = await tools.plan_check.execute("t", { step: 9 });
  ok("out-of-range step → error", r.isError && r.content[0].text.includes("out of range"));

  console.log("context injection:");
  const base = [{ role: "user", content: [{ type: "text", text: "hi" }] }];
  const res = await hooks.context({ messages: base });
  ok("appends a tail message", res.messages.length === base.length + 1);
  ok("tail is last (prefix unchanged)", res.messages[0] === base[0]);
  ok("tail carries the checklist", res.messages.at(-1).content[0].text.includes("read config"));

  console.log("touched-files tracking:");
  await hooks.tool_result({ toolName: "write", isError: false, input: { path: "src/x.py" }, content: [], type: "tool_result" });
  // snapshot before compaction
  await hooks.session_before_compact({});
  const snapDir = join(memoryDir(dir), "snapshots");
  const wrote = existsSync(snapDir) && readdirSync(snapDir).length > 0;
  ok("compaction writes a snapshot", wrote);
  if (wrote) {
    const f = readdirSync(snapDir)[0];
    ok("snapshot includes touched file", readFileSync(join(snapDir, f), "utf8").includes("src/x.py"));
    rmSync(memoryDir(dir), { recursive: true, force: true });
  } else { ok("snapshot includes touched file", false); }

  rmSync(dir, { recursive: true, force: true });
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

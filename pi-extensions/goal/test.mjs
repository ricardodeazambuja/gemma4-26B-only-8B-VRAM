// Tests the goal extension: pure renderers/helpers + tools + the agent_end loop
// driver (push), all with a stubbed exec/sendUserMessage — no live model, no real loop.
// Run: node --experimental-strip-types goal/test.mjs
import { mkdtempSync, rmSync, existsSync, readFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";
import factory, {
  freshGoal, isAutonomous, memoryDir, clip, unmetCriteria,
  renderGoal, buildContinue, buildSnapshot,
} from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

// ---------------------------------------------------------------- pure helpers
console.log("helpers:");
ok("freshGoal is active with default budget", freshGoal().status === "active" && freshGoal().maxCycles === 20);
ok("isAutonomous needs done_when", !isAutonomous(freshGoal()) && isAutonomous({ ...freshGoal(), doneWhen: "true" }));
ok("memoryDir is fs-safe under ~/.pi/memory", memoryDir("/home/x/My Proj!").startsWith(join(homedir(), ".pi", "memory")) && /^[A-Za-z0-9/_.-]+$/.test(memoryDir("/home/x/My Proj!")));

console.log("clip:");
ok("short text unchanged", clip("a\nb\nc") === "a\nb\nc");
const big = Array.from({ length: 120 }, (_, i) => `line${i}`).join("\n");
const clipped = clip(big);
ok("long text keeps the tail + flags clip", clipped.includes("…(clipped)") && clipped.includes("line119") && !clipped.includes("line0\n"));

console.log("renderers:");
const auto = { ...freshGoal(), objective: "all tests green", criteria: [{ text: "ci passes", done: false }], doneWhen: "pytest -q" };
const rg = renderGoal(auto);
ok("renderGoal shows objective + criteria", rg.includes("all tests green") && rg.includes("[ ] 1. ci passes"));
ok("autonomous render shows cycle budget", rg.includes("Cycle: 0/20"));
ok("advisory render hides cycle budget", !renderGoal({ ...freshGoal(), objective: "x" }).includes("Cycle:"));
ok("unmetCriteria lists unticked", unmetCriteria(auto).join() === "ci passes");
ok("buildContinue restates objective + asks for goal_done", buildContinue(auto).includes("all tests green") && buildContinue(auto).includes("goal_done"));
const snap = buildSnapshot({ ...auto, status: "blocked", blockedReason: "budget" });
ok("snapshot has the R6 template keys", ["Objective:", "Status:", "Cycles:", "Done-when:", "Unmet criteria:", "Criteria:"].every((k) => snap.includes(k)));
ok("snapshot records blocked reason", snap.includes("blocked — budget"));

// ---------------------------------------------------------------- harness
function makeHarness() {
  const tools = {}, hooks = {}, commands = {}, sent = [];
  const ctl = { execResult: { stdout: "", stderr: "", code: 0, killed: false }, execCalls: 0 };
  const fakePi = {
    registerTool: (t) => { tools[t.name] = t; },
    registerCommand: (n, o) => { commands[n] = o; },
    on: (ev, h) => { hooks[ev] = h; },
    exec: async () => { ctl.execCalls++; return ctl.execResult; },
    sendUserMessage: (content, opts) => { sent.push({ content, opts }); },
  };
  factory(fakePi);
  const dir = mkdtempSync(join(tmpdir(), "goal-"));
  const ctx = { cwd: dir, sessionManager: { getSessionDir: () => dir, getSessionId: () => "sess" } };
  return { tools, hooks, commands, sent, ctl, fakePi, dir, ctx };
}
const statusText = async (tools) => (await tools.goal_status.execute("t", {})).content[0].text;
const cleanup = (h) => { try { rmSync(memoryDir(h.dir), { recursive: true, force: true }); } catch {} rmSync(h.dir, { recursive: true, force: true }); };

const run = async () => {
  // ---- registration
  console.log("registration:");
  {
    const h = makeHarness();
    ok("registers all four tools", !!h.tools.goal_set && !!h.tools.goal_check && !!h.tools.goal_status && !!h.tools.goal_done);
    ok("registers /goal command", !!h.commands.goal);
    ok("registers prefix/context/agent_end/session_start hooks", !!h.hooks.before_agent_start && !!h.hooks.context && !!h.hooks.agent_end && !!h.hooks.session_start);
    cleanup(h);
  }

  // ---- goal_set validation (R2 teaching errors)
  console.log("goal_set validation:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    let r = await h.tools.goal_set.execute("t", { objective: "" });
    ok("empty objective → teaching error", r.isError && r.content[0].text.includes("goal_set("));
    r = await h.tools.goal_set.execute("t", { objective: "x".repeat(201) });
    ok("over-long objective → error", r.isError && r.content[0].text.includes("too long"));
    r = await h.tools.goal_set.execute("t", { objective: "ok", criteria: Array(9).fill("c") });
    ok("too many criteria → error", r.isError && r.content[0].text.includes("Too many criteria"));
    r = await h.tools.goal_set.execute("t", { objective: "ok", criteria: ["c".repeat(81)] });
    ok("over-long criterion → error", r.isError && r.content[0].text.includes("Criterion too long"));
    cleanup(h);
  }

  // ---- goal_set success + persistence + mode messaging
  console.log("goal_set success:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    let r = await h.tools.goal_set.execute("t", { objective: "all tests green", done_when: "pytest -q", max_cycles: 3 });
    ok("autonomous goal_set succeeds", !r.isError && r.content[0].text.includes("Autonomous"));
    ok("goal persisted to session dir", existsSync(join(h.dir, "goal-sess.json")));
    ok("durable status file written", existsSync(join(memoryDir(h.dir), "goal-status.md")));
    r = await h.tools.goal_set.execute("t", { objective: "ship it", criteria: ["docs updated"] });
    ok("criteria-only goal_set is advisory", !r.isError && r.content[0].text.includes("Advisory"));
    cleanup(h);
  }

  // ---- goal_check
  console.log("goal_check:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    let r = await h.tools.goal_check.execute("t", { n: 1 });
    ok("check before set → error", r.isError && r.content[0].text.includes("No goal set"));
    await h.tools.goal_set.execute("t", { objective: "x", done_when: "true" });
    r = await h.tools.goal_check.execute("t", { n: 1 });
    ok("check with no criteria → error", r.isError && r.content[0].text.includes("no criteria"));
    await h.tools.goal_set.execute("t", { objective: "x", criteria: ["a", "b"] });
    r = await h.tools.goal_check.execute("t", { n: 5 });
    ok("out-of-range criterion → error", r.isError && r.content[0].text.includes("out of range"));
    r = await h.tools.goal_check.execute("t", { n: 1 });
    ok("valid check ticks the criterion", !r.isError && r.content[0].text.includes("[x] 1. a"));
    cleanup(h);
  }

  // ---- goal_done (pull gate)
  console.log("goal_done pull gate:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "x", done_when: "pytest", criteria: ["a"] });
    h.ctl.execResult = { stdout: "", stderr: "1 failed", code: 1, killed: false };
    let r = await h.tools.goal_done.execute("t", {});
    ok("failing done_when → not done", r.isError && r.content[0].text.includes("done_when not satisfied"));
    h.ctl.execResult = { stdout: "", stderr: "", code: 0, killed: false };
    r = await h.tools.goal_done.execute("t", {});
    ok("done_when passes but criterion unticked → not done", r.isError && r.content[0].text.includes("unchecked criteria"));
    await h.tools.goal_check.execute("t", { n: 1 });
    r = await h.tools.goal_done.execute("t", {});
    ok("all met → goal complete", !r.isError && r.content[0].text.includes("Goal complete"));
    cleanup(h);
  }

  // ---- agent_end loop driver (push) — autonomous
  console.log("agent_end push (autonomous):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go", done_when: "check", max_cycles: 2 });
    h.ctl.execResult = { stdout: "", stderr: "not yet", code: 1, killed: false };
    await h.hooks.agent_end({});
    ok("failing check re-engages the agent", h.sent.length === 1 && h.sent[0].content.includes("go"));
    ok("cycle advanced to 1/2", (await statusText(h.tools)).includes("Cycle: 1/2"));
    await h.hooks.agent_end({});
    ok("max_cycles=2 allows a 2nd push", h.sent.length === 2 && (await statusText(h.tools)).includes("Cycle: 2/2"));
    await h.hooks.agent_end({});
    ok("budget exhausted → no further re-engage", h.sent.length === 2);
    const st = await statusText(h.tools);
    ok("status is blocked, durably", st.includes("blocked") && readFileSync(join(memoryDir(h.dir), "goal-status.md"), "utf8").includes("blocked"));
    cleanup(h);
  }

  // ---- agent_end auto-done when done_when passes
  console.log("agent_end auto-done:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go", done_when: "check", max_cycles: 5 });
    h.ctl.execResult = { stdout: "ok", stderr: "", code: 0, killed: false };
    await h.hooks.agent_end({});
    ok("passing check auto-marks done", (await statusText(h.tools)).includes("Status: done"));
    ok("done does not re-engage", h.sent.length === 0);
    cleanup(h);
  }

  // ---- criteria-only goal does NOT push at agent_end
  console.log("agent_end advisory (no push):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go", criteria: ["a"] });
    await h.hooks.agent_end({});
    ok("no done_when → no exec, no re-engage", h.ctl.execCalls === 0 && h.sent.length === 0);
    cleanup(h);
  }

  // ---- R1 injection: stable prefix + dynamic tail
  console.log("R1 injection:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "north star", done_when: "true" });
    const p1 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    const p2 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    ok("prefix injects the objective", p1 && p1.systemPrompt.includes("north star"));
    ok("prefix is byte-stable across turns", p1.systemPrompt === p2.systemPrompt);
    const base = [{ role: "user", content: [{ type: "text", text: "hi" }] }];
    const res = await h.hooks.context({ messages: base });
    ok("context appends a tail message", res.messages.length === base.length + 1);
    ok("prefix message untouched (tail-only)", res.messages[0] === base[0]);
    ok("tail carries the live status", res.messages.at(-1).content[0].text.includes("north star"));
    // once done, both injections fall silent
    h.ctl.execResult = { stdout: "", stderr: "", code: 0, killed: false };
    await h.hooks.agent_end({});
    const p3 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    ok("no prefix injection once not active", !p3);
    ok("no tail injection once not active", !(await h.hooks.context({ messages: base })));
    cleanup(h);
  }

  // ---- reload across a restart
  console.log("persistence reload:");
  {
    const h1 = makeHarness(); await h1.hooks.session_start({}, h1.ctx);
    await h1.tools.goal_set.execute("t", { objective: "survive restart", done_when: "true", max_cycles: 4 });
    // a fresh process/factory, same session dir + id
    const h2 = makeHarness();
    h2.ctx.cwd = h1.dir;
    h2.ctx.sessionManager = { getSessionDir: () => h1.dir, getSessionId: () => "sess" };
    await h2.hooks.session_start({}, h2.ctx);
    const st = await statusText(h2.tools);
    ok("goal reloads from disk on session_start", st.includes("survive restart") && st.includes("0/4"));
    rmSync(h2.dir, { recursive: true, force: true });
    cleanup(h1);
  }

  // ---- /goal command
  console.log("/goal command:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.commands.goal.handler("just do it", {});
    ok("/goal <text> sets an advisory objective", (await statusText(h.tools)).includes("just do it"));
    await h.commands.goal.handler("clear", {});
    ok("/goal clear clears the goal", (await statusText(h.tools)).includes("No goal set"));
    cleanup(h);
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

// Tests the goal extension (Option B: no checklist of its own — it verifies plan's steps).
// Pure helpers + tools + the agent_end loop driver, with stubbed exec/sendUserMessage and a
// simulated plan-<id>.json. No live model. Run: node --experimental-strip-types goal/test.mjs
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";
import factory, {
  freshGoal, isAutonomous, memoryDir, clip, readPlanRemaining,
  renderGoal, buildContinue, buildSnapshot,
} from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

// ---------------------------------------------------------------- pure helpers
console.log("helpers:");
ok("freshGoal is active, no criteria field", freshGoal().status === "active" && !("criteria" in freshGoal()));
ok("isAutonomous needs done_when", !isAutonomous(freshGoal()) && isAutonomous({ ...freshGoal(), doneWhen: "true" }));
ok("memoryDir is under ~/.pi/memory", memoryDir("/home/x/p").startsWith(join(homedir(), ".pi", "memory")));
ok("clip keeps the tail + flags truncation", clip(Array.from({ length: 120 }, (_, i) => `l${i}`).join("\n")).includes("l119"));

console.log("readPlanRemaining (the plan↔goal seam):");
{
  const d = mkdtempSync(join(tmpdir(), "goalplan-"));
  ok("no plan file → empty", readPlanRemaining(d, "s").length === 0);
  writeFileSync(join(d, "plan-s.json"), JSON.stringify({ steps: [{ text: "a", done: true }, { text: "b", done: false }] }));
  ok("returns only unfinished steps", readPlanRemaining(d, "s").join() === "b");
  writeFileSync(join(d, "plan-s.json"), JSON.stringify({ steps: [{ text: "a", done: true }] }));
  ok("all steps done → empty", readPlanRemaining(d, "s").length === 0);
  rmSync(d, { recursive: true, force: true });
}

console.log("renderers (no checklist — plan injects steps):");
const auto = { ...freshGoal(), objective: "all tests green", doneWhen: "pytest -q" };
const rg = renderGoal(auto);
ok("renderGoal shows objective + cycle budget", rg.includes("all tests green") && rg.includes("Cycle: 0/20"));
ok("renderGoal points at goal_done", rg.includes("goal_done"));
ok("active (self-judged) render shows the loop cycle too", renderGoal({ ...freshGoal(), objective: "x" }).includes("Cycle: 0/"));
ok("render shows a user-set completion check", renderGoal({ ...freshGoal(), objective: "x", check: "re-render and look at it" }).includes("re-render and look at it"));
ok("render falls back to the default (verify, don't assume) check", /verify/i.test(renderGoal({ ...freshGoal(), objective: "x" })));
ok("buildContinue restates objective + goal_done + the check", buildContinue(auto).includes("all tests green") && buildContinue(auto).includes("goal_done") && /verify/i.test(buildContinue(auto)));
const snap = buildSnapshot({ ...auto, status: "blocked", blockedReason: "budget" });
ok("snapshot has the R6 template keys", ["Objective:", "Status:", "Cycles:", "Done-when:", "Last check:"].every((k) => snap.includes(k)));
ok("snapshot has no criteria line", !/criteria/i.test(snap));

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
  return { tools, hooks, commands, sent, ctl, dir, ctx };
}
const statusText = async (tools) => (await tools.goal_status.execute("t", {})).content[0].text;
const writePlan = (dir, steps) => writeFileSync(join(dir, "plan-sess.json"), JSON.stringify({ steps }));
const cleanup = (h) => { try { rmSync(memoryDir(h.dir), { recursive: true, force: true }); } catch {} rmSync(h.dir, { recursive: true, force: true }); };

const run = async () => {
  console.log("registration:");
  {
    const h = makeHarness();
    ok("registers goal_set/status/done (NO goal_check)", !!h.tools.goal_set && !!h.tools.goal_status && !!h.tools.goal_done && !h.tools.goal_check);
    ok("registers /goal command + hooks", !!h.commands.goal && !!h.hooks.before_agent_start && !!h.hooks.context && !!h.hooks.agent_end && !!h.hooks.session_start);
    cleanup(h);
  }

  console.log("goal_set:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    let r = await h.tools.goal_set.execute("t", { objective: "" });
    ok("empty objective → teaching error", r.isError && r.content[0].text.includes("goal_set("));
    r = await h.tools.goal_set.execute("t", { objective: "x".repeat(201) });
    ok("over-long objective → error", r.isError && r.content[0].text.includes("too long"));
    r = await h.tools.goal_set.execute("t", { objective: "all tests green", done_when: "pytest -q", max_cycles: 3 });
    ok("autonomous goal_set succeeds + persists", !r.isError && r.content[0].text.includes("Autonomous") && existsSync(join(h.dir, "goal-sess.json")));
    r = await h.tools.goal_set.execute("t", { objective: "ship it" });
    ok("no done_when → self-judged loop", !r.isError && r.content[0].text.includes("Self-judged"));
    r = await h.tools.goal_set.execute("t", { objective: "fix the svg", check: "re-render and confirm it reads right" });
    ok("check is stored and surfaced", !r.isError && r.content[0].text.includes("re-render and confirm it reads right"));
    cleanup(h);
  }

  console.log("goal_done pull gate (done_when + plan steps):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "x", done_when: "pytest" });
    h.ctl.execResult = { stdout: "", stderr: "1 failed", code: 1, killed: false };
    let r = await h.tools.goal_done.execute("t", {});
    ok("failing done_when → not done", r.isError && r.content[0].text.includes("done_when not satisfied"));
    // done_when passes, but plan has an unfinished step
    h.ctl.execResult = { stdout: "", stderr: "", code: 0, killed: false };
    writePlan(h.dir, [{ text: "write code", done: false }]);
    r = await h.tools.goal_done.execute("t", {});
    ok("incomplete plan step blocks done", r.isError && r.content[0].text.includes("plan steps not done") && r.content[0].text.includes("write code"));
    // complete the plan
    writePlan(h.dir, [{ text: "write code", done: true }]);
    r = await h.tools.goal_done.execute("t", {});
    ok("done_when ok + plan complete → done", !r.isError && r.content[0].text.includes("Goal complete"));
    cleanup(h);
  }

  console.log("goal_done advisory (no done_when, no plan):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "x" });
    const r = await h.tools.goal_done.execute("t", {});
    ok("no machine check and no plan → accepts the claim", !r.isError && r.content[0].text.includes("Goal complete"));
    ok("no done_when → exec never called", h.ctl.execCalls === 0);
    cleanup(h);
  }

  console.log("agent_end push (autonomous):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go", done_when: "check", max_cycles: 2 });
    h.ctl.execResult = { stdout: "", stderr: "not yet", code: 1, killed: false };
    await h.hooks.agent_end({});
    ok("failing check re-engages the agent", h.sent.length === 1 && h.sent[0].content.includes("go"));
    ok("cycle advanced to 1/2", (await statusText(h.tools)).includes("Cycle: 1/2"));
    await h.hooks.agent_end({});
    ok("2nd push allowed (budget 2)", h.sent.length === 2);
    await h.hooks.agent_end({});
    ok("budget exhausted → no further re-engage", h.sent.length === 2);
    ok("status blocked, durably", (await statusText(h.tools)).includes("blocked") && readFileSync(join(memoryDir(h.dir), "goal-status.md"), "utf8").includes("blocked"));
    cleanup(h);
  }

  console.log("agent_end auto-done + self-judged loop:");
  {
    let h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go", done_when: "check" });
    h.ctl.execResult = { stdout: "ok", stderr: "", code: 0, killed: false };
    await h.hooks.agent_end({});
    ok("passing done_when auto-marks done, no re-engage", (await statusText(h.tools)).includes("Status: done") && h.sent.length === 0);
    cleanup(h);

    // Self-judged (no done_when) now LOOPS: it re-engages each turn until goal_done or the cap.
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.hooks.agent_end({});
    ok("self-judged goal re-engages without exec (loops)", h.ctl.execCalls === 0 && h.sent.length === 1 && h.sent[0].content.includes("go"));
    cleanup(h);
  }

  console.log("yield to the human (input source 'interactive'):");
  {
    let h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    ok("registers an input hook", !!h.hooks.input);
    await h.tools.goal_set.execute("t", { objective: "go" });
    // user types → input(interactive) → that turn must NOT trigger a re-engagement
    await h.hooks.input({ type: "input", source: "interactive", text: "actually do X instead" });
    await h.hooks.agent_end({});
    ok("a turn YOU started does not re-engage (no hijack)", h.sent.length === 0);
    // the flag is one-shot: it suppresses only the turn you drove, so a later loop-driven turn pushes
    await h.hooks.agent_end({});
    ok("interjection is one-shot — a later loop-driven turn still pushes", h.sent.length === 1);
    cleanup(h);

    // extension-sourced input (the loop's own re-engagement) must NOT count as a human interjection
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.hooks.input({ type: "input", source: "extension", text: "continue" });
    await h.hooks.agent_end({});
    ok("extension-sourced input still loops", h.sent.length === 1);
    cleanup(h);

    // setting a goal ARMS the loop even from a turn you typed (it's pursuit, not a redirect away)
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.hooks.input({ type: "input", source: "interactive", text: "set a goal to do X and work on it" });
    await h.tools.goal_set.execute("t", { objective: "do X" });
    await h.hooks.agent_end({});
    ok("goal_set during a typed turn still arms the loop", h.sent.length === 1 && h.sent[0].content.includes("do X"));
    cleanup(h);

    // a CLEARED goal must not re-engage (freshGoal status is 'active' but the objective is empty)
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.commands.goal.handler("clear", {});
    await h.hooks.agent_end({});
    ok("a cleared goal does not re-engage", h.sent.length === 0);
    cleanup(h);
  }

  console.log("R1 injection:");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "north star", done_when: "true" });
    const p1 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    const p2 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    ok("prefix injects the objective, byte-stable", p1.systemPrompt.includes("north star") && p1.systemPrompt === p2.systemPrompt);
    const baseMsgs = [
      { role: "user", content: [{ type: "text", text: "earlier" }] },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
      { role: "user", content: [{ type: "text", text: "hi" }] },
    ];
    const res = await h.hooks.context({ messages: baseMsgs });
    ok("folds status into the user turn, prefix untouched", res.messages.length === baseMsgs.length && res.messages[0] === baseMsgs[0] && res.messages.at(-1).content[0].text === "hi");
    const goalTail = res.messages.at(-1).content.at(-1).text;
    ok("status rides as a wrapped <reminder> block", goalTail.includes("north star") && goalTail.startsWith("<reminder>"));
    h.ctl.execResult = { stdout: "", stderr: "", code: 0, killed: false };
    await h.hooks.agent_end({});
    ok("no prefix/tail injection once not active", !(await h.hooks.before_agent_start({ systemPrompt: "BASE" })) && !(await h.hooks.context({ messages: baseMsgs })));
    cleanup(h);
  }

  console.log("persistence reload + /goal command:");
  {
    const h1 = makeHarness(); await h1.hooks.session_start({}, h1.ctx);
    await h1.tools.goal_set.execute("t", { objective: "survive restart", done_when: "true", max_cycles: 4 });
    const h2 = makeHarness();
    h2.ctx.cwd = h1.dir; h2.ctx.sessionManager = { getSessionDir: () => h1.dir, getSessionId: () => "sess" };
    await h2.hooks.session_start({}, h2.ctx);
    ok("goal reloads from disk", (await statusText(h2.tools)).includes("survive restart") && (await statusText(h2.tools)).includes("0/4"));
    rmSync(h2.dir, { recursive: true, force: true });

    const sentBefore = h1.sent.length;
    await h1.commands.goal.handler("just do it", {});
    ok("/goal <text> sets the objective", (await statusText(h1.tools)).includes("just do it"));
    ok("/goal <text> STARTS the work (drives a turn with the full text)", h1.sent.length === sentBefore + 1 && h1.sent.at(-1).content === "just do it");
    await h1.commands.goal.handler("clear", {});
    ok("/goal clear clears it", (await statusText(h1.tools)).includes("No goal set"));
    ok("/goal clear does not drive a turn", h1.sent.length === sentBefore + 1);
    cleanup(h1);
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

// Tests the goal extension (Option B: no checklist of its own — it verifies plan's steps).
// Pure helpers + tools + the agent_end loop driver, with stubbed exec/sendUserMessage and a
// simulated plan-<id>.json. No live model. Run: node --experimental-strip-types goal/test.mjs
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import factory, {
  freshGoal, isAutonomous, memoryDir, clip, readPlanRemaining,
  renderGoal, buildContinue, buildSnapshot, lastAssistantStopReason,
  bandGuidance, currentBand, applyTempAnneal,
} from "./index.ts";
import {
  bandFor, reservedCommit, temperature, isColdBand,
  annealConfigFromEnv, annealEnabled, tempAnnealEnabled, DEFAULT_ANNEAL,
} from "./anneal.ts";
const SRC = new URL(".", import.meta.url).pathname;

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
ok("buildContinue surfaces a goal_done rejection reason when given one", buildContinue(auto, null, "plan steps not done: write code").includes("rejected") && buildContinue(auto, null, "plan steps not done: write code").includes("write code"));
ok("buildContinue omits the rejection line when there is none", !buildContinue(auto).includes("rejected"));
ok("lastAssistantStopReason finds the latest assistant stopReason", lastAssistantStopReason([{ role: "assistant", stopReason: "stop" }, { role: "user" }, { role: "assistant", stopReason: "aborted" }]) === "aborted");
ok("lastAssistantStopReason is undefined for empty/non-array", lastAssistantStopReason([]) === undefined && lastAssistantStopReason(null) === undefined);
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

  console.log("Esc/abort stops the loop:");
  {
    // primary detect: message_end latches the aborted assistant turn → next agent_end yields
    let h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    ok("registers a message_end hook", !!h.hooks.message_end);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.hooks.message_end({ message: { role: "assistant", stopReason: "aborted" } });
    await h.hooks.agent_end({});
    ok("an aborted turn does NOT re-engage (Esc stops it)", h.sent.length === 0);
    // one-shot: the latch clears, so a later clean turn still drives the loop (abort doesn't brick it)
    await h.hooks.agent_end({});
    ok("abort is one-shot — a later clean turn still pushes", h.sent.length === 1);
    cleanup(h);

    // backstop detect: agent_end's own messages carry the aborted assistant (no message_end emitted)
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.hooks.agent_end({ messages: [{ role: "user", content: "go" }, { role: "assistant", stopReason: "aborted" }] });
    ok("abort detected from agent_end messages too (backstop)", h.sent.length === 0);
    cleanup(h);

    // a normal (non-aborted) finish still re-engages — the gate is abort-specific
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "go" });
    await h.hooks.message_end({ message: { role: "assistant", stopReason: "stop" } });
    await h.hooks.agent_end({ messages: [{ role: "assistant", stopReason: "stop" }] });
    ok("a normal finish still re-engages", h.sent.length === 1);
    cleanup(h);

    // STALENESS: abort latches regardless of goal status (message_end has no guard) and agent_end
    // only clears it past the active-status guard — so arming a NEW goal must clear it, or the new
    // goal's first turn gets swallowed and the loop looks dead.
    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.hooks.message_end({ message: { role: "assistant", stopReason: "aborted" } }); // latched while no goal
    await h.tools.goal_set.execute("t", { objective: "fresh" });
    await h.hooks.agent_end({});
    ok("goal_set clears a stale abort latch (new goal's first turn fires)", h.sent.length === 1);
    cleanup(h);

    h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.hooks.message_end({ message: { role: "assistant", stopReason: "aborted" } });
    const before = h.sent.length;
    await h.commands.goal.handler("do the thing", {}); // +1 kickoff
    await h.hooks.agent_end({});                        // +1 re-engage iff the stale latch was cleared
    ok("/goal kickoff clears a stale abort latch", h.sent.length === before + 2);
    cleanup(h);
  }

  console.log("a rejected goal_done feeds the next re-engagement (gradient):");
  {
    // self-judged goal, plan step left unchecked → goal_done is gated; the reason rides the next push
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "ship" });
    writePlan(h.dir, [{ text: "write code", done: false }]);
    const r = await h.tools.goal_done.execute("t", {});
    ok("goal_done gated on the unfinished plan step", r.isError && r.content[0].text.includes("write code"));
    await h.hooks.agent_end({});
    ok("the re-engagement names why goal_done was rejected", h.sent.length === 1 && h.sent[0].content.includes("rejected") && h.sent[0].content.includes("write code"));
    // and it's one-shot: a second push (still gated state, but no new goal_done) doesn't repeat it
    await h.hooks.agent_end({});
    ok("the rejection reason is surfaced once, then cleared", h.sent.length === 2 && !h.sent[1].content.includes("rejected"));
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

  // ============================================================ ANNEALING (PRD: docs/goal-annealing-prd.md)
  console.log("anneal schedule (pure):");
  {
    // reserved-tail math + band boundaries for tiny → normal budgets (FR4)
    ok("reservedCommit ≥1 always", reservedCommit(1) === 1 && reservedCommit(20) === 5);
    ok("maxCycles=1 → pure decide", bandFor(1, 1) === "decide");
    ok("maxCycles=2 → explore then decide", bandFor(1, 2) === "explore" && bandFor(2, 2) === "decide");
    ok("maxCycles=3 → explore, consolidate, decide", bandFor(1, 3) === "explore" && bandFor(2, 3) === "consolidate" && bandFor(3, 3) === "decide");
    const arc20 = Array.from({ length: 20 }, (_, i) => bandFor(i + 1, 20));
    ok("maxCycles=20 → explore[1-10] consolidate[11-15] commit[16-19] decide[20]",
      arc20[0] === "explore" && arc20[9] === "explore" && arc20[10] === "consolidate" &&
      arc20[14] === "consolidate" && arc20[15] === "commit" && arc20[18] === "commit" && arc20[19] === "decide");
    ok("last cycle is ALWAYS decide", [1, 2, 3, 7, 50, 200].every((m) => bandFor(m, m) === "decide"));
    ok("≥1 explore whenever budget > 1", [2, 3, 5, 20, 100].every((m) => Array.from({ length: m }, (_, i) => bandFor(i + 1, m)).includes("explore")));
    ok("isColdBand only commit/decide", isColdBand("commit") && isColdBand("decide") && !isColdBand("explore") && !isColdBand("consolidate"));
    // temperature: monotone, normalized endpoints
    let mono = true; for (let c = 1; c <= 20; c++) if (temperature(c, 20) > temperature(c - 1, 20) + 1e-9) mono = false;
    ok("temperature monotone non-increasing", mono);
    ok("temperature endpoints 1 → 0", temperature(0, 20) === 1 && temperature(20, 20) === 0);
    ok("cosine holds heat early (warmer than linear at p=0.25)", temperature(5, 20) > temperature(5, 20, { ...DEFAULT_ANNEAL, shape: "linear" }) && temperature(5, 20) > 0.8);
    // env config
    ok("annealEnabled default on, off via flag", annealEnabled({}) && !annealEnabled({ PI_GOAL_ANNEAL: "0" }) && !annealEnabled({ PI_GOAL_ANNEAL: "off" }));
    ok("tempAnnealEnabled default off, on via flag", !tempAnnealEnabled({}) && tempAnnealEnabled({ PI_GOAL_TEMP_ANNEAL: "1" }));
    const cfg = annealConfigFromEnv({ PI_GOAL_COMMIT_FRACTION: "0.4", PI_GOAL_TEMP_HI: "0.8", PI_GOAL_ANNEAL_SHAPE: "linear" });
    ok("annealConfigFromEnv reads overrides", cfg.commitFraction === 0.4 && cfg.tempHi === 0.8 && cfg.shape === "linear");
    ok("annealConfigFromEnv ignores garbage (fail-safe)", annealConfigFromEnv({ PI_GOAL_COMMIT_FRACTION: "xyz" }).commitFraction === DEFAULT_ANNEAL.commitFraction);
  }

  console.log("banded buildContinue + currentBand:");
  {
    const mk = (cycle, maxCycles) => ({ ...freshGoal(), objective: "render a pelican on a bike", maxCycles, cycle });
    ok("currentBand reflects cycle", currentBand(mk(1, 20)) === "explore" && currentBand(mk(20, 20)) === "decide");
    const explore = buildContinue(mk(1, 20));
    const commit = buildContinue(mk(17, 20));
    const decide = buildContinue(mk(20, 20));
    ok("explore push names the explore phase + invites breadth", explore.includes("phase: explore") && /explore/i.test(explore) && explore.includes("approach"));
    ok("explore does NOT offer goal_conclude (gated to cold)", !explore.includes("goal_conclude"));
    ok("commit push offers goal_conclude", commit.includes("phase: commit") && commit.includes("goal_conclude"));
    ok("decide push is decision-forcing + lists both exits", decide.includes("phase: decide") && decide.includes("goal_done") && decide.includes('goal_conclude(outcome="partial"') && decide.includes("abandoned"));
    ok("every band keeps the honesty floor (verify/unverified)", [explore, commit, decide].every((t) => /verif|unverified|establish/i.test(t)));
    ok("banded push still restates the objective", decide.includes("render a pelican on a bike"));
  }

  console.log("renderGoal phase line + concluded; snapshot:");
  {
    const active = { ...freshGoal(), objective: "x", maxCycles: 20, cycle: 3 };
    ok("active render shows Phase + temp", renderGoal(active).includes("Phase: explore") && /temp \d/.test(renderGoal(active)));
    const concluded = { ...freshGoal(), objective: "x", status: "concluded", outcome: "partial", summary: "80% there, lighting unverified" };
    ok("concluded render shows outcome + summary", renderGoal(concluded).includes("concluded — partial") && renderGoal(concluded).includes("80% there"));
    ok("snapshot records the conclusion distinctly", buildSnapshot(concluded).includes("concluded — partial") && buildSnapshot(concluded).includes("80% there"));
  }

  console.log("goal_conclude (model-owned stop affordance):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    ok("registers goal_conclude + before_provider_request hook", !!h.tools.goal_conclude && !!h.hooks.before_provider_request);
    let r = await h.tools.goal_conclude.execute("t", { outcome: "partial", summary: "x" });
    ok("conclude with no goal → error", r.isError && r.content[0].text.includes("No goal"));
    // early (explore) → gated
    await h.tools.goal_set.execute("t", { objective: "render a pelican", max_cycles: 20 });
    r = await h.tools.goal_conclude.execute("t", { outcome: "abandoned", summary: "nope" });
    ok("conclude in explore phase → gated (too early)", r.isError && r.content[0].text.includes("Too early") && r.content[0].text.includes("explore"));
    // tiny budget → cycle 0 is already the decide band → allowed
    await h.tools.goal_set.execute("t", { objective: "tiny", max_cycles: 1 });
    r = await h.tools.goal_conclude.execute("t", { outcome: "partial", summary: "" });
    ok("empty summary → error", r.isError && r.content[0].text.includes("summary"));
    r = await h.tools.goal_conclude.execute("t", { outcome: "partial", summary: "got 80% there; lighting unverified" });
    ok("conclude in decide band → concluded", !r.isError && r.content[0].text.includes("concluded"));
    ok("status concluded + outcome/summary surfaced", (await statusText(h.tools)).includes("concluded — partial") && (await statusText(h.tools)).includes("got 80%"));
    const md = readFileSync(join(memoryDir(h.dir), "goal-status.md"), "utf8");
    ok("snapshot persisted the conclusion", md.includes("concluded — partial") && md.includes("got 80%"));
    r = await h.tools.goal_conclude.execute("t", { outcome: "partial", summary: "again" });
    ok("conclude again → already concluded (idempotent-ish)", !r.isError && r.content[0].text.includes("already concluded"));
    // round-trip the concluded terminal status + outcome/summary through session_start reload
    const h2 = makeHarness();
    h2.ctx.cwd = h.dir; h2.ctx.sessionManager = { getSessionDir: () => h.dir, getSessionId: () => "sess" };
    await h2.hooks.session_start({}, h2.ctx);
    const reloaded = await statusText(h2.tools);
    ok("concluded status + outcome/summary survive reload", reloaded.includes("concluded — partial") && reloaded.includes("got 80%"));
    rmSync(h2.dir, { recursive: true, force: true });
    cleanup(h);
  }

  console.log("agent_end terminal ramp (decide before blocked; conclude stops the loop):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "land it", max_cycles: 2 });
    await h.hooks.agent_end({}); // cycle 0→1 → band(1,2)=explore
    ok("first push is the explore phase", h.sent.at(-1).content.includes("phase: explore"));
    await h.hooks.agent_end({}); // cycle 1→2 → band(2,2)=decide
    ok("final push is the DECIDE phase (forced decision)", h.sent.at(-1).content.includes("phase: decide") && h.sent.at(-1).content.includes("goal_conclude"));
    const r = await h.tools.goal_conclude.execute("t", { outcome: "partial", summary: "did what I could" });
    ok("conclude accepted in the decide turn", !r.isError);
    const sentAfter = h.sent.length;
    await h.hooks.agent_end({}); // status concluded → must be a no-op
    ok("a concluded goal neither re-engages nor blocks", h.sent.length === sentAfter && (await statusText(h.tools)).includes("concluded"));
    cleanup(h);
  }

  console.log("agent_end: budget exhausted AFTER a decide turn → blocked (FR7):");
  {
    const h = makeHarness(); await h.hooks.session_start({}, h.ctx);
    await h.tools.goal_set.execute("t", { objective: "land it", max_cycles: 2 });
    await h.hooks.agent_end({}); // →1 explore
    await h.hooks.agent_end({}); // →2 decide (the forced-decision turn)
    await h.hooks.agent_end({}); // budget out → blocked
    const st = await statusText(h.tools);
    ok("blocked only after a decide turn, with a decide-aware reason", st.includes("blocked") && st.includes("decide-phase"));
    cleanup(h);
  }

  console.log("Channel B: applyTempAnneal (pure guard logic, fail-open):");
  {
    const active = { ...freshGoal(), objective: "x", maxCycles: 20, cycle: 2 };
    const body = { model: "gemma", messages: [{ role: "user", content: "hi" }] };
    ok("tempOn=false → payload untouched (identity)", applyTempAnneal(body, active, { annealOn: true, tempOn: false }) === body);
    ok("annealOn=false → payload untouched", applyTempAnneal(body, active, { annealOn: false, tempOn: true }) === body);
    const out = applyTempAnneal(body, active, { annealOn: true, tempOn: true });
    ok("on+active+valid body → temperature added on a COPY", out !== body && typeof out.temperature === "number" && Array.isArray(out.messages));
    ok("input body is never mutated in place", !("temperature" in body));
    const tEarly = applyTempAnneal(body, { ...active, cycle: 1 }, { annealOn: true, tempOn: true }).temperature;
    const tLate = applyTempAnneal(body, { ...active, cycle: 19 }, { annealOn: true, tempOn: true }).temperature;
    ok("sampling temperature cools across cycles", tEarly > tLate);
    // FR9: never clobber the request's own temperature upward — cool from it, down toward the floor.
    const based = { model: "g", temperature: 0.6, messages: [{ role: "user", content: "hi" }] };
    const hot = applyTempAnneal(based, { ...active, cycle: 1 }, { annealOn: true, tempOn: true }).temperature;
    ok("hot end ≈ the request's own base temperature (no clobber)", Math.abs(hot - 0.6) < 0.02 && hot <= 0.6 + 1e-9);
    ok("never raises temperature above the base, any cycle", [1, 5, 10, 19, 20].every((c) => applyTempAnneal(based, { ...active, cycle: c }, { annealOn: true, tempOn: true }).temperature <= 0.6 + 1e-9));
    ok("a lower base yields a lower temp than a higher base", applyTempAnneal(based, { ...active, cycle: 5 }, { annealOn: true, tempOn: true }).temperature < applyTempAnneal(body, { ...active, cycle: 5 }, { annealOn: true, tempOn: true }).temperature);
    ok("no active goal → untouched", applyTempAnneal(body, { ...active, status: "done" }, { annealOn: true, tempOn: true }) === body);
    ok("empty objective → untouched", applyTempAnneal(body, { ...active, objective: "" }, { annealOn: true, tempOn: true }) === body);
    ok("non-object payload → untouched (fail-open)", applyTempAnneal("notabody", active, { annealOn: true, tempOn: true }) === "notabody");
    const noMsgs = { model: "x" };
    ok("body without messages[] → untouched (fail-open)", applyTempAnneal(noMsgs, active, { annealOn: true, tempOn: true }) === noMsgs);
  }

  console.log("env toggle: PI_GOAL_ANNEAL=0 → flat fallback (subprocess, real module flags):");
  {
    const probe = `import { buildContinue, freshGoal } from "./index.ts";` +
      `const s = { ...freshGoal(), objective: "x", maxCycles: 5, cycle: 1 };` +
      `process.stdout.write(buildContinue(s));`;
    const runProbe = (env) => execFileSync(process.execPath,
      ["--experimental-strip-types", "--input-type=module", "-e", probe],
      { cwd: SRC, env: { ...process.env, ...env } }).toString();
    const flat = runProbe({ PI_GOAL_ANNEAL: "0" });
    const banded = runProbe({ PI_GOAL_ANNEAL: "1" });
    ok("PI_GOAL_ANNEAL=0 → flat push, no bands", flat.includes("Keep working toward the goal") && !flat.includes("phase:"));
    ok("default → banded push (Phase —)", banded.includes("phase:") && banded.includes("Phase —"));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

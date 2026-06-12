// Tests the pipe extension: the parser, the directive builder, and the command handler
// (which drives the agent via sendUserMessage). No live model needed.
// Run: node --experimental-strip-types pipe/test.mjs
import factory, { parsePipe, describeStage, buildDirective } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

const EX = "/goal implement the results from /plan a python script that says hello world";

console.log("parsePipe:");
ok("empty → null", parsePipe("") === null && parsePipe("   ") === null);
ok("no command → null", parsePipe("just do the thing") === null);
const st = parsePipe(EX);
ok("splits into two stages in textual order", st.length === 2 && st[0].cmd === "goal" && st[1].cmd === "plan");
ok("outer keeps its connective text", st[0].text === "implement the results from");
ok("inner captures its argument", st[1].text === "a python script that says hello world");
ok("single command parses (degenerate pipe)", parsePipe("/plan write tests").length === 1);
ok("known commands parse in textual order", parsePipe("/plan a /goal b /plan c").map(s => s.cmd).join() === "plan,goal,plan");
ok("unknown commands → null (only known split)", parsePipe("/frobnicate x") === null);
ok("slashes inside args are not commands", (() => { const r = parsePipe("/plan touch /tmp/foo and and/or bar"); return r.length === 1 && r[0].text === "touch /tmp/foo and and/or bar"; })());

console.log("describeStage:");
ok("plan → plan_set", describeStage("plan", "X").includes("plan_set") && describeStage("plan", "X").includes("X"));
ok("goal → goal_set", describeStage("goal", "Y").includes("goal_set"));
ok("unknown → generic run", describeStage("frobnicate", "Z") === "run /frobnicate: Z");

console.log("buildDirective:");
const dir = buildDirective(st);
ok("executes innermost (plan) first as step 1", /1\. create a plan \(call plan_set\) for: a python script/.test(dir));
ok("outer (goal) is step 2", /2\. set the goal \(call goal_set\): implement the results from/.test(dir));
ok("step 2 references step 1", dir.includes("use the result of step 1"));
ok("step 1 has no back-reference", !/1\..*use the result of step/.test(dir.split("\n")[1]));
ok("ends with a carry-out line", /Then carry out the work/.test(dir));

console.log("command handler:");
{
  const commands = {}, sent = [];
  const fakePi = {
    registerCommand: (n, o) => { commands[n] = o; },
    sendUserMessage: (content, opts) => { sent.push({ content, opts }); },
  };
  factory(fakePi);
  ok("registers /pipe", !!commands.pipe);

  await commands.pipe.handler(EX, {});
  ok("valid pipe drives the agent once", sent.length === 1);
  ok("directive contains both expanded stages", sent[0].content.includes("plan_set") && sent[0].content.includes("goal_set"));
  ok("delivered as a turn (followUp)", sent[0].opts?.deliverAs === "followUp");

  await commands.pipe.handler("no commands here", {});
  ok("invalid pipe does NOT drive the agent", sent.length === 1);

  // ctx without a ui must not throw (notify is best-effort)
  let threw = false;
  try { await commands.pipe.handler("", {}); } catch { threw = true; }
  ok("missing ui/ctx is handled gracefully", !threw && sent.length === 1);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

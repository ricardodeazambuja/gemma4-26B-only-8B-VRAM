// Tests operating-manual: the manual text, JIT nudge logic, and both hooks.
// Run: node --experimental-strip-types operating-manual/test.mjs
import factory, { MANUAL, buildManual, nudgeForResult } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("manual:");
ok("manual is non-empty", buildManual().length > 0);
ok("manual stays under 600 bytes (prefix tax)", Buffer.byteLength(MANUAL, "utf8") <= 600);
ok("references get_symbols", MANUAL.includes("get_symbols"));
ok("references plan_set", MANUAL.includes("plan_set"));
ok("references remember", MANUAL.includes("remember"));
ok("rules are imperative triggers (no 'you are weak')", !/weak|bad at|struggle/i.test(MANUAL));

console.log("JIT nudge logic:");
const many = Array.from({ length: 100 }, (_, i) => `match ${i}`).join("\n");
ok("grep with many matches → nudge", nudgeForResult("grep", many)?.includes("Narrow"));
ok("grep with few matches → no nudge", nudgeForResult("grep", "one\ntwo") === null);
ok("find with many entries → nudge", nudgeForResult("find", many)?.includes("Scope"));
ok("bash large output → no nudge (not targeted)", nudgeForResult("bash", many) === null);
ok("read → no nudge", nudgeForResult("read", many) === null);

console.log("hooks:");
const hooks = {};
factory({ on: (e, h) => (hooks[e] = h), registerTool() {}, registerCommand() {} });
ok("registers before_agent_start", typeof hooks.before_agent_start === "function");
ok("registers tool_result", typeof hooks.tool_result === "function");

const run = async () => {
  let r = await hooks.before_agent_start({ systemPrompt: "BASE" });
  ok("appends manual to system prompt", r.systemPrompt.startsWith("BASE") && r.systemPrompt.includes("Operating rules"));
  const r2 = await hooks.before_agent_start({ systemPrompt: "BASE" });
  ok("manual injection is byte-stable", r2.systemPrompt === r.systemPrompt);

  const grepEv = { toolName: "grep", isError: false, content: [{ type: "text", text: many }] };
  r = await hooks.tool_result(grepEv);
  ok("noisy grep result gets a tail nudge", !!r && r.content.at(-1).text.includes("Narrow"));
  ok("nudge preserves original content", r.content[0].text === many);

  r = await hooks.tool_result({ toolName: "grep", isError: false, content: [{ type: "text", text: "a\nb" }] });
  ok("small grep result untouched", r === undefined);

  r = await hooks.tool_result({ toolName: "grep", isError: true, content: [{ type: "text", text: many }] });
  ok("errored result is skipped", r === undefined);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

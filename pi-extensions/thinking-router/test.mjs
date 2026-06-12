// Tests thinking-router: the routing heuristic + the input/pin hooks.
// Run: node --experimental-strip-types thinking-router/test.mjs
import factory, { routeLevel } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("routeLevel:");
ok("greeting → off", routeLevel("thanks!") === "off");
ok("ok/continue → off", routeLevel("ok continue") === "off");
ok("short factual → low", routeLevel("what port does it use") === "low");
ok("empty → low", routeLevel("") === "low");
ok("code request → medium", routeLevel("fix the bug in parser.ts") === "medium");
ok("code fence → medium", routeLevel("why does this fail\n```js\nx()\n```") === "medium");
ok("explain/why → medium", routeLevel("explain how the scheduler works") === "medium");
ok("long prose → medium", routeLevel("I was wondering if you could take a look at the whole pipeline and tell me what you think about it overall") === "medium");
ok("'hi there friend how are you doing today' (>6 words) not forced off", routeLevel("hi there friend how are you doing today") !== "off");

console.log("hooks:");
const hooks = {};
let lvl = "medium", pinnedCalls = 0;
const pi = {
  on: (e, h) => (hooks[e] = h),
  registerTool() {}, registerCommand() {},
  getThinkingLevel: () => lvl,
  setThinkingLevel: (l) => { lvl = l; pinnedCalls++; },
};
factory(pi);
ok("registers input + thinking_level_select", !!hooks.input && !!hooks.thinking_level_select);

const run = async () => {
  lvl = "medium";
  await hooks.input({ type: "input", text: "thanks!", source: "user" });
  ok("trivial input routes to off", lvl === "off");

  await hooks.input({ type: "input", text: "implement a retry loop in fetch.ts", source: "user" });
  ok("code input routes to medium", lvl === "medium");

  // non-user source is ignored
  lvl = "high";
  await hooks.input({ type: "input", text: "ok", source: "agent" });
  ok("non-user input is not routed", lvl === "high");

  // user pin via /thinking disables auto-routing
  await hooks.thinking_level_select({ source: "set", level: "xhigh" });
  lvl = "xhigh";
  await hooks.input({ type: "input", text: "thanks", source: "user" });
  ok("after a manual /thinking pin, auto-routing stops", lvl === "xhigh");

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

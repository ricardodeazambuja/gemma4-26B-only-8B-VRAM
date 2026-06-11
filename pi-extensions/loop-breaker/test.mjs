// Tests the loop-breaker state machine + the real tool_result handler.
// Run: node --experimental-strip-types loop-breaker/test.mjs
import { makeTracker, nudgeText } from "./index.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("tracker:");
let t = makeTracker();
ok("1st identical failure → count 1", t.record("bash", { command: "x" }, true) === 1);
ok("2nd identical failure → count 2", t.record("bash", { command: "x" }, true) === 2);
ok("3rd identical failure → count 3", t.record("bash", { command: "x" }, true) === 3);

console.log("reset behavior:");
t = makeTracker();
t.record("bash", { command: "x" }, true);
t.record("bash", { command: "x" }, true);
ok("a different call resets the streak", t.record("bash", { command: "y" }, true) === 1);

t = makeTracker();
t.record("bash", { command: "x" }, true);
t.record("bash", { command: "x" }, true);
ok("a success resets the streak", t.record("bash", { command: "x" }, false) === 0);
ok("after success, same call starts at 1 again", t.record("bash", { command: "x" }, true) === 1);

console.log("args sensitivity:");
t = makeTracker();
ok("same tool, same args → counts", (t.record("read", { path: "a" }, true), t.record("read", { path: "a" }, true)) === 2);
t = makeTracker();
t.record("read", { path: "a" }, true);
ok("same tool, different args → resets", t.record("read", { path: "b" }, true) === 1);

console.log("nudge text:");
ok("nudge names the tool and count", nudgeText("bash", 3).includes("bash") && nudgeText("bash", 3).includes("3 times"));

// --- real handler ---
console.log("handler:");
let handler;
factory({ on: (ev, h) => { if (ev === "tool_result") handler = h; }, registerTool() {}, registerCommand() {} });
ok("registers tool_result handler", typeof handler === "function");

const ev = (isError, input = { command: "bad" }) => ({
  type: "tool_result", toolName: "bash", toolCallId: "c", isError, input,
  content: [{ type: "text", text: "error: boom" }], details: undefined,
});

const run = async () => {
  let r = await handler(ev(true)); ok("1st failure: no nudge", r === undefined);
  r = await handler(ev(true)); ok("2nd failure: no nudge", r === undefined);
  r = await handler(ev(true));
  ok("3rd failure: nudge appended", !!r && r.content.length === 2 && r.content[1].text.includes("change your approach"));
  ok("3rd failure: original content preserved", r.content[0].text === "error: boom");

  // a success in between clears it
  await handler(ev(false));
  r = await handler(ev(true));
  ok("post-success failure: no nudge", r === undefined);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

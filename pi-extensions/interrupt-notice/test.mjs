// Tests the interrupt-notice tracker + the real message_end/context handlers.
// Run: node --experimental-strip-types interrupt-notice/test.mjs
import { makeInterruptTracker, NOTICE } from "./index.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("tracker:");
let t = makeInterruptTracker();
ok("nothing pending initially", t.consume() === false);

t = makeInterruptTracker();
t.observe("assistant", "aborted");
ok("aborted assistant turn → pending", t.consume() === true);
ok("consumed exactly once (self-limiting)", t.consume() === false);

t = makeInterruptTracker();
t.observe("assistant", "stop");
ok("normal stop → not pending", t.consume() === false);
t.observe("assistant", "error");
ok("error stop → not pending", t.consume() === false);
t.observe("user", undefined);
ok("user message → not pending", t.consume() === false);
t.observe("assistant", "toolUse");
ok("toolUse stop → not pending", t.consume() === false);

console.log("notice text:");
ok("mentions the interruption", NOTICE.toLowerCase().includes("interrupted"));
ok("tells the model what to do", NOTICE.includes("Re-read") || NOTICE.includes("ask"));

// --- real handlers ---
console.log("handlers:");
let onMessageEnd, onContext;
factory({
  on: (ev, h) => { if (ev === "message_end") onMessageEnd = h; if (ev === "context") onContext = h; },
  registerTool() {}, registerCommand() {},
});
ok("registers message_end handler", typeof onMessageEnd === "function");
ok("registers context handler", typeof onContext === "function");

const baseMsgs = () => [{ role: "user", content: [{ type: "text", text: "do the thing" }] }];
const ctxEvent = () => ({ type: "context", messages: baseMsgs() });
const msgEnd = (role, stopReason) => ({ type: "message_end", message: { role, stopReason, content: [] } });

const run = async () => {
  // no interrupt yet → context injects nothing
  let r = await onContext(ctxEvent());
  ok("no interrupt: context unchanged", r === undefined);

  // user aborts an assistant turn
  await onMessageEnd(msgEnd("assistant", "aborted"));
  r = await onContext(ctxEvent());
  ok("after abort: a note is appended", !!r && Array.isArray(r.messages) && r.messages.length === 2);
  ok("note is a tail user message with the exact NOTICE", r.messages[1].role === "user" && r.messages[1].content[0].text === NOTICE);
  ok("original messages preserved", r.messages[0].content[0].text === "do the thing");

  // self-limiting: next context (no new abort) injects nothing
  r = await onContext(ctxEvent());
  ok("note fires once, not every turn", r === undefined);

  // a normal turn end does not arm it
  await onMessageEnd(msgEnd("assistant", "stop"));
  r = await onContext(ctxEvent());
  ok("normal turn: no note", r === undefined);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

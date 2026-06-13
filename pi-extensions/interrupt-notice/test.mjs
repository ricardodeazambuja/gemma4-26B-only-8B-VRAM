// Tests the interrupt-notice tracker + the real message_end/agent_end/context handlers.
// Run: node --experimental-strip-types interrupt-notice/test.mjs
//
// Caveat these tests cannot remove: they drive the handlers with event shapes WE construct,
// so they prove the extension's internal logic, not that a real pi build fires message_end or
// agent_end with stopReason "aborted" on a user abort. That can only be confirmed live (Esc a
// real session; see the README "Verify it on your build").
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { makeInterruptTracker, NOTICE } from "./index.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("tracker:");
let t = makeInterruptTracker();
ok("nothing pending initially", t.consume() === false);

t = makeInterruptTracker();
ok("observe returns true when it arms", t.observe("assistant", "aborted") === true);
ok("observe returns false when already armed (idempotent)", t.observe("assistant", "aborted") === false);
ok("aborted assistant turn → pending", t.consume() === true);
ok("consumed exactly once (self-limiting)", t.consume() === false);

t = makeInterruptTracker();
ok("non-abort observe returns false", t.observe("assistant", "stop") === false);
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

// --- real handlers --- a fresh factory call per scenario keeps tracker state isolated.
console.log("handlers:");
const mk = () => {
  let onMessageEnd, onAgentEnd, onContext;
  factory({
    on: (ev, h) => {
      if (ev === "message_end") onMessageEnd = h;
      if (ev === "agent_end") onAgentEnd = h;
      if (ev === "context") onContext = h;
    },
    registerTool() {}, registerCommand() {},
  });
  return { onMessageEnd, onAgentEnd, onContext };
};

const baseMsgs = () => [{ role: "user", content: [{ type: "text", text: "do the thing" }] }];
const ctxEvent = () => ({ type: "context", messages: baseMsgs() });
const msgEnd = (role, stopReason) => ({ type: "message_end", message: { role, stopReason, content: [] } });
const agentEnd = (messages) => ({ type: "agent_end", messages });

{
  const { onMessageEnd, onAgentEnd, onContext } = mk();
  ok("registers message_end handler", typeof onMessageEnd === "function");
  ok("registers agent_end handler", typeof onAgentEnd === "function");
  ok("registers context handler", typeof onContext === "function");
}

const run = async () => {
  // ---- message_end (primary) ----
  {
    const { onMessageEnd, onContext } = mk();
    let r = await onContext(ctxEvent());
    ok("no interrupt: context unchanged", r === undefined);

    await onMessageEnd(msgEnd("assistant", "aborted"));
    r = await onContext(ctxEvent());
    ok("after abort: a note is appended", !!r && Array.isArray(r.messages) && r.messages.length === 2);
    ok("note is a tail user message with the exact NOTICE", r.messages[1].role === "user" && r.messages[1].content[0].text === NOTICE);
    ok("original messages preserved", r.messages[0].content[0].text === "do the thing");

    r = await onContext(ctxEvent());
    ok("note fires once, not every turn", r === undefined);

    await onMessageEnd(msgEnd("assistant", "stop"));
    r = await onContext(ctxEvent());
    ok("normal turn: no note", r === undefined);
  }

  // ---- agent_end (backstop) ----
  {
    const { onAgentEnd, onContext } = mk();
    // run ends with the aborted assistant as the last message
    await onAgentEnd(agentEnd([
      { role: "user", content: [{ type: "text", text: "do the thing" }] },
      { role: "assistant", stopReason: "aborted", content: [] },
    ]));
    let r = await onContext(ctxEvent());
    ok("agent_end backstop: trailing aborted assistant arms the note", !!r && r.messages.length === 2 && r.messages[1].content[0].text === NOTICE);
  }
  {
    const { onAgentEnd, onContext } = mk();
    // a trailing tool/user message must not hide the aborted assistant before it
    await onAgentEnd(agentEnd([
      { role: "assistant", stopReason: "aborted", content: [] },
      { role: "user", content: [{ type: "text", text: "tool result" }] },
    ]));
    let r = await onContext(ctxEvent());
    ok("agent_end: lastAssistant scans past a trailing non-assistant msg", !!r && r.messages.length === 2);
  }
  {
    const { onAgentEnd, onContext } = mk();
    await onAgentEnd(agentEnd([
      { role: "user", content: [{ type: "text", text: "do the thing" }] },
      { role: "assistant", stopReason: "stop", content: [] },
    ]));
    let r = await onContext(ctxEvent());
    ok("agent_end: a normally-ended run does not arm", r === undefined);
  }

  // ---- both events on the same abort still fire exactly once ----
  {
    const { onMessageEnd, onAgentEnd, onContext } = mk();
    await onMessageEnd(msgEnd("assistant", "aborted"));
    await onAgentEnd(agentEnd([{ role: "assistant", stopReason: "aborted", content: [] }]));
    let r = await onContext(ctxEvent());
    ok("message_end + agent_end on one abort → note appears", !!r && r.messages.length === 2);
    r = await onContext(ctxEvent());
    ok("…and only once (no double-fire across the two events)", r === undefined);
  }

  // ---- PI_INTERRUPT_DEBUG writes a log file ----
  // DEBUG_LOG is fixed at module-load from the env, so this needs a child process with the env
  // set before the import. Verifies the diagnostic plumbing the README documents — not pi's real
  // abort behavior (see the file header caveat).
  console.log("debug:");
  {
    const here = dirname(fileURLToPath(import.meta.url));
    const logPath = join(mkdtempSync(join(tmpdir(), "intnotice-")), "dbg.log");
    const driver = `
      const f = (await import(${JSON.stringify(join(here, "index.ts"))})).default;
      let onMessageEnd, onContext;
      f({ on:(e,h)=>{ if(e==='message_end')onMessageEnd=h; if(e==='context')onContext=h; },
          registerTool(){}, registerCommand(){} });
      await onMessageEnd({ type:'message_end', message:{ role:'assistant', stopReason:'aborted', content:[] } });
      await onContext({ type:'context', messages:[{ role:'user', content:[{ type:'text', text:'x' }] }] });
    `;
    const res = spawnSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", driver], {
      env: { ...process.env, PI_INTERRUPT_DEBUG: logPath },
      encoding: "utf8",
    });
    const log = res.status === 0 && existsSync(logPath) ? readFileSync(logPath, "utf8") : "";
    ok("PI_INTERRUPT_DEBUG=<path> logs the arm", log.includes("armed via message_end"));
    ok("PI_INTERRUPT_DEBUG=<path> logs the inject", log.includes("injected interrupt notice"));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

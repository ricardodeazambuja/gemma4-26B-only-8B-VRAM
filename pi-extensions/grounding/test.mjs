// Tests the grounding extension: MINDSET + ANCHOR in the byte-stable system prefix, and the CHECK
// folded into the latest user turn as a wrapped <reminder> block (not a bare user message), with
// the trivial-turn skip and the no-user-turn fallback. No live model needed.
// Run: node --experimental-strip-types grounding/test.mjs
import factory, { ANCHOR, CHECK, MINDSET, wrapReminder, isTurnStart } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("content — scientific-method framing:");
ok("MINDSET frames the engineering mindset", /engineer/i.test(MINDSET));
ok("MINDSET names all three modes (derive/simulate/reference)", /derive/i.test(MINDSET) && /simulate/i.test(MINDSET) && /reference/i.test(MINDSET));
ok("MINDSET treats memory as a hypothesis, not fact", /hypothesis/i.test(MINDSET) && /memory/i.test(MINDSET));
ok("MINDSET mentions concrete means (bash/read/web_search)", ["bash", "read", "web_search"].every((t) => MINDSET.includes(t)));
ok("CHECK is the act-now prove-it pass", /prove it/i.test(CHECK) && /before you answer/i.test(CHECK));
ok("CHECK invokes the same three modes", /derive/i.test(CHECK) && /simulate/i.test(CHECK) && /read it from a/i.test(CHECK));
ok("CHECK requires an explicit unverified label", CHECK.includes("unverified"));
ok("CHECK and MINDSET are different text", CHECK !== MINDSET && !MINDSET.includes(CHECK));
ok("both stay terse (MINDSET < 800, CHECK < 700 bytes)", Buffer.byteLength(MINDSET) < 800 && Buffer.byteLength(CHECK) < 700);

console.log("content — anchor (reminders are not the user):");
ok("ANCHOR explains the <reminder> marker", ANCHOR.includes("<reminder>"));
ok("ANCHOR says reminders are not a new instruction / not the user", /not a new instruction/i.test(ANCHOR) && /not the user/i.test(ANCHOR));
ok("ANCHOR re-anchors the task to the user's real request", /user's most recent real request/i.test(ANCHOR));
ok("ANCHOR stays terse (< 600 bytes)", Buffer.byteLength(ANCHOR) < 600);

// --- harness ---
function makeHarness(level) {
  const hooks = {};
  const fakePi = { on: (ev, h) => { hooks[ev] = h; } };
  if (level !== "NONE") fakePi.getThinkingLevel = () => level;
  factory(fakePi);
  return { hooks, fakePi };
}
// Realistic conversation at context time: ends with the user's latest request.
const base = () => [
  { role: "user", content: [{ type: "text", text: "first" }] },
  { role: "assistant", content: [{ type: "text", text: "ok" }] },
  { role: "user", content: [{ type: "text", text: "THE REAL REQUEST" }] },
];
const lastBlockText = (msg) => msg.content[msg.content.length - 1].text;
// True if any user message carries the wrapped CHECK as its last content block.
const hasCheck = (msgs) => msgs.some((m) => m.role === "user" && Array.isArray(m.content) && lastBlockText(m) === wrapReminder(CHECK));

const run = async () => {
  console.log("registration:");
  {
    const h = makeHarness("medium");
    ok("registers both hooks", !!h.hooks.before_agent_start && !!h.hooks.context && Object.keys(h.hooks).length === 2);
  }

  console.log("beginning — system-prefix injection (byte-stable):");
  {
    const h = makeHarness("medium");
    const p1 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    const p2 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    ok("appends MINDSET and ANCHOR to the system prompt", p1 && p1.systemPrompt.startsWith("BASE") && p1.systemPrompt.includes(MINDSET) && p1.systemPrompt.includes(ANCHOR));
    ok("is byte-stable across turns", p1.systemPrompt === p2.systemPrompt);
    const ho = makeHarness("off");
    ok("prefix injects even at trivial thinking level", (await ho.hooks.before_agent_start({ systemPrompt: "BASE" })).systemPrompt.includes(ANCHOR));
  }

  console.log("end — CHECK folded into the user's turn (not a new message):");
  {
    const h = makeHarness("medium");
    const msgs = base();
    const res = await h.hooks.context({ messages: msgs });
    ok("does NOT add a new message", res && res.messages.length === msgs.length);
    ok("folds into the last user turn", res.messages[2].role === "user" && res.messages[2].content.length === 2);
    ok("the user's real request stays first in the turn", res.messages[2].content[0].text === "THE REAL REQUEST");
    ok("the check is appended as a wrapped <reminder> block", lastBlockText(res.messages[2]) === wrapReminder(CHECK));
    ok("the wrapped block contains the CHECK and the marker", lastBlockText(res.messages[2]).includes(CHECK) && lastBlockText(res.messages[2]).includes("<reminder>"));
    ok("other messages are untouched (same refs)", res.messages[0] === msgs[0] && res.messages[1] === msgs[1]);
    ok("does not mutate the original user message", msgs[2].content.length === 1 && res.messages[2] !== msgs[2]);
    const res2 = await h.hooks.context({ messages: base() });
    ok("the folded block is byte-identical across turns", lastBlockText(res2.messages[2]) === lastBlockText(res.messages[2]));
  }

  console.log("empty history — appends a standalone wrapped reminder:");
  {
    const res = await makeHarness("medium").hooks.context({ messages: [] });
    ok("empty history is a fresh turn → standalone wrapped reminder", res && res.messages.length === 1 && res.messages[0].role === "user" && lastBlockText(res.messages[0]) === wrapReminder(CHECK));
  }

  console.log("trivial-turn skip:");
  {
    for (const lvl of ["off", "minimal"]) {
      ok(`skips the check at level "${lvl}"`, (await makeHarness(lvl).hooks.context({ messages: base() })) === undefined);
    }
    ok("folds the check at a real thinking level", (await makeHarness("high").hooks.context({ messages: base() })) !== undefined);
    ok("folds the check when thinking level is unavailable", (await makeHarness("NONE").hooks.context({ messages: base() })) !== undefined);
  }

  console.log("tool-loop throttle — CHECK only at the start of a turn:");
  {
    const h = makeHarness("medium");
    const loop = () => [
      { role: "user", content: [{ type: "text", text: "THE REAL REQUEST" }] },
      { role: "assistant", content: [{ type: "text", text: "calling a tool" }] },
      { role: "toolResult", content: [{ type: "text", text: "tool output" }] },
    ];
    ok("skips the check mid tool-loop (tail is a toolResult)", (await h.hooks.context({ messages: loop() })) === undefined);
    const loopThenReminder = [...loop(), { role: "user", content: [{ type: "text", text: wrapReminder("## Active plan") }] }];
    ok("skips when only reminder-only user turns sit after the toolResult", (await h.hooks.context({ messages: loopThenReminder })) === undefined);
    ok("fires at the start of a turn (tail is the user's genuine message)", (await h.hooks.context({ messages: base() })) !== undefined);
  }

  console.log("isTurnStart (pure):");
  {
    const userTurn = (t) => ({ role: "user", content: [{ type: "text", text: t }] });
    const tool = { role: "toolResult", content: [{ type: "text", text: "o" }] };
    ok("true when the tail is a genuine user turn", isTurnStart(base()));
    ok("false when the tail is a toolResult (mid loop)", !isTurnStart([userTurn("q"), tool]));
    ok("false when the tail is an assistant turn", !isTurnStart([userTurn("q"), { role: "assistant", content: [{ type: "text", text: "a" }] }]));
    ok("skips past reminder-only user turns to the real tail", !isTurnStart([userTurn("q"), tool, { role: "user", content: [{ type: "text", text: wrapReminder("CHECK") }] }]));
    ok("a folded real turn (real text + reminder) is still a turn start", isTurnStart([{ role: "user", content: [{ type: "text", text: "real" }, { type: "text", text: wrapReminder("CHECK") }] }]));
    ok("an image-only user turn counts as genuine", isTurnStart([{ role: "user", content: [{ type: "image", image: "…" }] }]));
    ok("empty history → treated as a fresh turn", isTurnStart([]));
  }

  console.log("integration (threaded context pipeline — order-independence):");
  {
    const thread = async (handlers) => {
      let cur = structuredClone(base());
      for (const hd of handlers) { const r = await hd({ type: "context", messages: cur }); if (r && r.messages) cur = r.messages; }
      return cur;
    };
    const g = makeHarness("medium").hooks.context;
    // Stubs mimic plan/memory appending their own (still bare, for now) user turns.
    const stub = (label) => async (e) => ({ messages: [...e.messages, { role: "user", content: [{ type: "text", text: label }] }] });
    const plan = stub("## Active plan"), recall = stub("## Recalled");
    for (const [name, order] of [["first", [g, plan, recall]], ["middle", [plan, g, recall]], ["last", [plan, recall, g]]]) {
      ok(`CHECK present when grounding runs ${name}`, hasCheck(await thread(order)));
    }
    // Production order (grounding first): folds into the REAL request, not a sibling's stub turn.
    const out = await thread([g, plan, recall]);
    ok("when grounding runs first, the check folds into the real request", out[2].role === "user" && out[2].content[0].text === "THE REAL REQUEST" && lastBlockText(out[2]) === wrapReminder(CHECK));
    ok("no fresh user turn added by grounding (count = base + plan + recall)", out.length === 5);
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

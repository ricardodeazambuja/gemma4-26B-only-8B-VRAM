// Tests the grounding extension: the protocol text + the tail-injection context hook,
// including the trivial-turn skip. No live model needed.
// Run: node --experimental-strip-types grounding/test.mjs
import factory, { PROTOCOL } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("protocol:");
ok("directs the reasoning, not the answer", /while reasoning/i.test(PROTOCOL));
ok("contrasts tool vs memory", /tool/i.test(PROTOCOL) && /memory/i.test(PROTOCOL));
ok("names investigative tools", ["read", "grep", "web_search", "fetch_page"].every((t) => PROTOCOL.includes(t)));
ok("requires an explicit unverified flag", PROTOCOL.includes("haven't") && /verify/i.test(PROTOCOL));
ok("forbids guessing as fact", /guess as fact/i.test(PROTOCOL));
ok("stays terse (< 600 bytes)", Buffer.byteLength(PROTOCOL) < 600);

// --- harness ---
function makeHarness(level) {
  const hooks = {};
  const fakePi = { on: (ev, h) => { hooks[ev] = h; } };
  if (level !== "NONE") fakePi.getThinkingLevel = () => level;
  factory(fakePi);
  return { hooks, fakePi };
}
const base = () => [
  { role: "user", content: [{ type: "text", text: "hi" }] },
  { role: "assistant", content: [{ type: "text", text: "ok" }] },
];

const run = async () => {
  console.log("registration:");
  {
    const h = makeHarness("medium");
    ok("registers exactly the context hook", !!h.hooks.context && Object.keys(h.hooks).length === 1);
  }

  console.log("tail injection (reasoning turn):");
  {
    const h = makeHarness("medium");
    const msgs = base();
    const res = await h.hooks.context({ messages: msgs });
    ok("appends one tail message", res && res.messages.length === msgs.length + 1);
    ok("tail is the protocol", res.messages.at(-1).content[0].text === PROTOCOL);
    ok("tail role is user", res.messages.at(-1).role === "user");
    ok("prefix messages untouched (same refs)", res.messages[0] === msgs[0] && res.messages[1] === msgs[1]);
    // identical text every turn (no cache churn from the injection itself)
    const res2 = await h.hooks.context({ messages: base() });
    ok("injected text is byte-identical across turns", res2.messages.at(-1).content[0].text === res.messages.at(-1).content[0].text);
  }

  console.log("trivial-turn skip:");
  {
    for (const lvl of ["off", "minimal"]) {
      const h = makeHarness(lvl);
      const r = await h.hooks.context({ messages: base() });
      ok(`skips injection at level "${lvl}"`, r === undefined);
    }
    const h = makeHarness("high");
    ok("injects at a real thinking level", (await h.hooks.context({ messages: base() })) !== undefined);
  }

  console.log("degrades without thinking-router:");
  {
    const h = makeHarness("NONE"); // no getThinkingLevel on the API
    const r = await h.hooks.context({ messages: base() });
    ok("injects when thinking level is unavailable", r !== undefined && r.messages.at(-1).content[0].text === PROTOCOL);
  }

  console.log("integration (threaded context pipeline — runner.js order-independence):");
  {
    // Faithfully mirror runner.js `emitContext`: thread currentMessages through each
    // context handler; a returned {messages} feeds the next. This proves grounding's
    // protocol survives no matter where it sits relative to the other tail-injectors
    // (goal/plan/recall), since load order (readdirSync) is not guaranteed.
    const thread = async (handlers) => {
      let cur = structuredClone(base());
      for (const h of handlers) {
        const r = await h({ type: "context", messages: cur });
        if (r && r.messages) cur = r.messages;
      }
      return cur;
    };
    const g = makeHarness("medium").hooks.context;
    const stub = (label) => async (e) => ({ messages: [...e.messages, { role: "user", content: [{ type: "text", text: label }] }] });
    const plan = stub("## Active plan"), recall = stub("## Recalled");
    const hasProto = (msgs) => msgs.some((m) => m.content?.[0]?.text === PROTOCOL);
    for (const [name, order] of [["first", [g, plan, recall]], ["middle", [plan, g, recall]], ["last", [plan, recall, g]]]) {
      ok(`protocol present when grounding runs ${name}`, hasProto(await thread(order)));
    }
    const out = await thread([g, plan, recall]);
    ok("composes with neighbors (all three injections land)",
      hasProto(out) && out.some((m) => m.content?.[0]?.text === "## Active plan") && out.some((m) => m.content?.[0]?.text === "## Recalled"));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

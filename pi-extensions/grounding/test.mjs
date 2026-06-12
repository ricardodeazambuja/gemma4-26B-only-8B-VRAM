// Tests the grounding extension: the two bracketing injections (MINDSET in the system
// prefix, CHECK at the tail), their content, the trivial-turn skip, and order-independence.
// No live model needed. Run: node --experimental-strip-types grounding/test.mjs
import factory, { MINDSET, CHECK } from "./index.ts";

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
ok("both stay terse (each < 700 bytes)", Buffer.byteLength(MINDSET) < 700 && Buffer.byteLength(CHECK) < 700);

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
    ok("registers both bracketing hooks", !!h.hooks.before_agent_start && !!h.hooks.context && Object.keys(h.hooks).length === 2);
  }

  console.log("beginning — system-prefix injection (byte-stable):");
  {
    const h = makeHarness("medium");
    const p1 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    const p2 = await h.hooks.before_agent_start({ systemPrompt: "BASE" });
    ok("appends MINDSET to the system prompt", p1 && p1.systemPrompt.startsWith("BASE") && p1.systemPrompt.includes(MINDSET));
    ok("is byte-stable across turns", p1.systemPrompt === p2.systemPrompt);
    // unconditional: the prefix must stay even on a trivial turn (otherwise the cache churns)
    const ho = makeHarness("off");
    ok("prefix injects even at trivial thinking level", (await ho.hooks.before_agent_start({ systemPrompt: "BASE" })).systemPrompt.includes(MINDSET));
  }

  console.log("end — tail injection (the prove-it check):");
  {
    const h = makeHarness("medium");
    const msgs = base();
    const res = await h.hooks.context({ messages: msgs });
    ok("appends one tail message", res && res.messages.length === msgs.length + 1);
    ok("tail is the CHECK (not the MINDSET)", res.messages.at(-1).content[0].text === CHECK);
    ok("tail role is user", res.messages.at(-1).role === "user");
    ok("prefix messages untouched (same refs)", res.messages[0] === msgs[0] && res.messages[1] === msgs[1]);
    const res2 = await h.hooks.context({ messages: base() });
    ok("tail text is byte-identical across turns", res2.messages.at(-1).content[0].text === res.messages.at(-1).content[0].text);
  }

  console.log("trivial-turn skip (tail only):");
  {
    for (const lvl of ["off", "minimal"]) {
      ok(`skips the tail check at level "${lvl}"`, (await makeHarness(lvl).hooks.context({ messages: base() })) === undefined);
    }
    ok("injects the tail at a real thinking level", (await makeHarness("high").hooks.context({ messages: base() })) !== undefined);
    ok("injects the tail when thinking level is unavailable", (await makeHarness("NONE").hooks.context({ messages: base() })) !== undefined);
  }

  console.log("integration (threaded context pipeline — runner.js order-independence):");
  {
    // Mirror runner.js emitContext threading: prove the CHECK survives no matter where
    // grounding sits relative to the other tail-injectors (load order is readdirSync).
    const thread = async (handlers) => {
      let cur = structuredClone(base());
      for (const hd of handlers) { const r = await hd({ type: "context", messages: cur }); if (r && r.messages) cur = r.messages; }
      return cur;
    };
    const g = makeHarness("medium").hooks.context;
    const stub = (label) => async (e) => ({ messages: [...e.messages, { role: "user", content: [{ type: "text", text: label }] }] });
    const plan = stub("## Active plan"), recall = stub("## Recalled");
    const hasCheck = (m) => m.some((x) => x.content?.[0]?.text === CHECK);
    for (const [name, order] of [["first", [g, plan, recall]], ["middle", [plan, g, recall]], ["last", [plan, recall, g]]]) {
      ok(`CHECK present when grounding runs ${name}`, hasCheck(await thread(order)));
    }
    const out = await thread([g, plan, recall]);
    ok("composes with neighbors (all three land)", hasCheck(out) && out.some((m) => m.content?.[0]?.text === "## Active plan") && out.some((m) => m.content?.[0]?.text === "## Recalled"));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

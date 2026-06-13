// Tests the compaction-notice tracker + the real session_compact/context handlers.
// Run: node --experimental-strip-types compaction-notice/test.mjs
//
// Caveat these tests cannot remove: they drive the handlers with event shapes WE construct, so
// they prove the extension's internal logic, not that a real pi build (a) fires session_compact
// with compactionEntry.tokensBefore on compaction, (b) fires a context event afterward before the
// next response, or (c) renders the compaction summary into that context's message list. Those can
// only be confirmed live (/compact a real session; see the README "Verify it on your build").
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compactionNotice, makeCompactionTracker } from "./index.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("tracker:");
let t = makeCompactionTracker();
ok("nothing pending initially", t.consume().fired === false);

t = makeCompactionTracker();
ok("observe returns true when it arms", t.observe(12345) === true);
ok("observe returns false when already armed (idempotent)", t.observe(999) === false);
let c = t.consume();
ok("consume reports fired after a compaction", c.fired === true);
ok("consume carries the first tokensBefore (coalesced)", c.tokensBefore === 12345);
ok("consumed exactly once (self-limiting)", t.consume().fired === false);

t = makeCompactionTracker();
ok("observe with no token count still arms", t.observe() === true);
ok("consume carries undefined tokensBefore", t.consume().tokensBefore === undefined);

console.log("notice text:");
ok("mentions compaction", compactionNotice().toLowerCase().includes("compact"));
ok("tells the model what to do", compactionNotice().includes("re-read") || compactionNotice().includes("re-derive"));
ok("warns detail may be lost", compactionNotice().toLowerCase().includes("lost") || compactionNotice().toLowerCase().includes("imprecise"));
ok("no size clause when token count unknown", !compactionNotice().includes("~"));
ok("rounds tokensBefore to ~k", compactionNotice(12345).includes("~12k"));
ok("omits size clause for zero/negative/NaN", !compactionNotice(0).includes("~") && !compactionNotice(-5).includes("~") && !compactionNotice(NaN).includes("~"));

// --- real handlers --- a fresh factory call per scenario keeps tracker state isolated.
console.log("handlers:");
const mk = () => {
  let onCompact, onContext;
  factory({
    on: (ev, h) => {
      if (ev === "session_compact") onCompact = h;
      if (ev === "context") onContext = h;
    },
    registerTool() {}, registerCommand() {},
  });
  return { onCompact, onContext };
};

const baseMsgs = () => [{ role: "user", content: [{ type: "text", text: "do the thing" }] }];
const ctxEvent = () => ({ type: "context", messages: baseMsgs() });
const compactEvent = (tokensBefore) => ({ type: "session_compact", compactionEntry: { type: "compaction", tokensBefore }, fromExtension: false });

{
  const { onCompact, onContext } = mk();
  ok("registers session_compact handler", typeof onCompact === "function");
  ok("registers context handler", typeof onContext === "function");
}

const run = async () => {
  {
    const { onCompact, onContext } = mk();
    let r = await onContext(ctxEvent());
    ok("no compaction yet: context unchanged", r === undefined);

    await onCompact(compactEvent(48000));
    r = await onContext(ctxEvent());
    ok("after compaction: a note is folded in (no new message)", !!r && Array.isArray(r.messages) && r.messages.length === 1);
    const note = r.messages.at(-1).content.at(-1);
    ok("note rides as a wrapped <reminder> block on the user turn", r.messages.at(-1).role === "user" && note.type === "text" && note.text.startsWith("<reminder>"));
    ok("note carries the rounded token anchor", note.text.includes("~48k"));
    ok("original user text preserved (stays first)", r.messages[0].content[0].text === "do the thing");

    r = await onContext(ctxEvent());
    ok("note fires once, not every turn", r === undefined);
  }
  {
    // compaction without a token count still injects, just without the size clause
    const { onCompact, onContext } = mk();
    await onCompact(compactEvent(undefined));
    let r = await onContext(ctxEvent());
    ok("compaction w/o token count: still injects", !!r && r.messages.length === 1);
    ok("…and omits the ~k size clause", !r.messages.at(-1).content.at(-1).text.includes("~"));
  }

  // ---- PI_COMPACTION_DEBUG writes a log file ----
  // DEBUG_LOG is fixed at module-load from the env, so this needs a child process with the env set
  // before the import. Verifies the diagnostic plumbing the README documents — not pi's real
  // compaction behavior (see the file header caveat).
  console.log("debug:");
  {
    const here = dirname(fileURLToPath(import.meta.url));
    const logPath = join(mkdtempSync(join(tmpdir(), "compactnotice-")), "dbg.log");
    const driver = `
      const f = (await import(${JSON.stringify(join(here, "index.ts"))})).default;
      let onCompact, onContext;
      f({ on:(e,h)=>{ if(e==='session_compact')onCompact=h; if(e==='context')onContext=h; },
          registerTool(){}, registerCommand(){} });
      await onCompact({ type:'session_compact', compactionEntry:{ type:'compaction', tokensBefore:5000 }, fromExtension:false });
      await onContext({ type:'context', messages:[{ role:'user', content:[{ type:'text', text:'x' }] }] });
    `;
    const res = spawnSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", driver], {
      env: { ...process.env, PI_COMPACTION_DEBUG: logPath },
      encoding: "utf8",
    });
    const log = res.status === 0 && existsSync(logPath) ? readFileSync(logPath, "utf8") : "";
    ok("PI_COMPACTION_DEBUG=<path> logs the arm", log.includes("armed via session_compact"));
    ok("PI_COMPACTION_DEBUG=<path> logs the inject", log.includes("injected compaction notice"));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

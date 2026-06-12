// Tests the toolsets extension: config loading, the pure gating helpers, the session_start
// gate, and the /tools command. No live model. Run: node --experimental-strip-types toolsets/test.mjs
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import factory, { DEFAULT_GROUPS, loadConfig, toolsForGroups, applyDisabled } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

const BUILTINS = ["read", "edit", "bash", "grep"];
const CORE = ["get_symbols", "plan_set", "goal_set"];
const ALL = [...BUILTINS, ...CORE, "web_search", "fetch_page", "remember", "recall", "forget", "advisor"];

console.log("pure helpers:");
ok("default groups cover web/memory/advisor", DEFAULT_GROUPS.web.includes("web_search") && DEFAULT_GROUPS.memory.includes("recall") && DEFAULT_GROUPS.advisor.includes("advisor"));
ok("toolsForGroups flattens + dedups", toolsForGroups(["web", "advisor"], DEFAULT_GROUPS).sort().join() === ["advisor", "fetch_page", "web_search"].sort().join());
ok("toolsForGroups ignores unknown groups", toolsForGroups(["nope"], DEFAULT_GROUPS).length === 0);
const gated = applyDisabled(ALL, ["web"], DEFAULT_GROUPS);
ok("applyDisabled removes only the group's tools", !gated.includes("web_search") && !gated.includes("fetch_page"));
ok("applyDisabled never drops built-ins or core", BUILTINS.every((t) => gated.includes(t)) && CORE.every((t) => gated.includes(t)));
ok("applyDisabled keeps other groups", gated.includes("remember") && gated.includes("advisor"));
ok("disabling nothing is a no-op", applyDisabled(ALL, [], DEFAULT_GROUPS).length === ALL.length);

console.log("config loading:");
ok("defaults: no disabled, default groups", (() => { const c = loadConfig({}); return c.disabled.length === 0 && c.groups.web.length === 2; })());
ok("env PI_TOOLSETS_DISABLED overrides", loadConfig({ PI_TOOLSETS_DISABLED: "web, advisor" }).disabled.join() === "web,advisor");
{
  const d = mkdtempSync(join(tmpdir(), "toolsets-"));
  const f = join(d, "c.json");
  writeFileSync(f, JSON.stringify({ disabled: ["memory"], groups: { extra: ["foo"] } }));
  const c = loadConfig({ PI_TOOLSETS_CONFIG: f });
  ok("file: disabled + custom group merged", c.disabled.join() === "memory" && c.groups.extra.join() === "foo" && c.groups.web.length === 2);
  ok("env beats file for disabled", loadConfig({ PI_TOOLSETS_CONFIG: f, PI_TOOLSETS_DISABLED: "web" }).disabled.join() === "web");
  rmSync(d, { recursive: true, force: true });
}

// --- harness ---
function makeHarness(activeList) {
  const hooks = {}, commands = {};
  let active = [...activeList];
  const calls = [];
  const fakePi = {
    on: (e, h) => { hooks[e] = h; },
    registerCommand: (n, o) => { commands[n] = o; },
    getActiveTools: () => [...active],
    setActiveTools: (names) => { active = [...names]; calls.push([...names]); },
  };
  factory(fakePi);
  return { hooks, commands, getActive: () => active, calls };
}

const run = async () => {
  console.log("session_start gate:");
  {
    process.env.PI_TOOLSETS_DISABLED = "web";
    const h = makeHarness(ALL);
    await h.hooks.session_start({});
    ok("disabled group removed from active set", !h.getActive().includes("web_search") && !h.getActive().includes("fetch_page"));
    ok("built-ins + core untouched", BUILTINS.every((t) => h.getActive().includes(t)) && h.getActive().includes("goal_set"));
    delete process.env.PI_TOOLSETS_DISABLED;
  }
  {
    const h = makeHarness(ALL); // no PI_TOOLSETS_DISABLED → opt-in no-op
    await h.hooks.session_start({});
    ok("nothing disabled → setActiveTools never called", h.calls.length === 0 && h.getActive().length === ALL.length);
  }

  console.log("/tools command:");
  {
    const h = makeHarness(ALL);
    ok("registers /tools, no model-facing tool", !!h.commands.tools);
    await h.commands.tools.handler("off web", {});
    ok("/tools off hides the group", !h.getActive().includes("web_search") && !h.getActive().includes("fetch_page"));
    await h.commands.tools.handler("on web", {});
    ok("/tools on reveals it again", h.getActive().includes("web_search") && h.getActive().includes("fetch_page"));
    const before = h.calls.length;
    await h.commands.tools.handler("off bogus", {});
    ok("unknown group → no setActiveTools call", h.calls.length === before);
    await h.commands.tools.handler("", {});
    ok("bare /tools lists without changing tools", h.calls.length === before);
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

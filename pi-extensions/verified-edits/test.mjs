// Tests verified-edits by driving the real extension through a fake pi.
// Run: node --experimental-strip-types verified-edits/test.mjs
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import factory, { pickChecker } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

// --- pickChecker unit checks ---
console.log("pickChecker:");
ok("python file picks a checker", pickChecker("/x/foo.py")?.label !== undefined);
ok("unknown extension → null", pickChecker("/x/foo.xyz") === null);
ok("json picks json checker", pickChecker("/x/foo.json")?.label === "json");

// --- end-to-end through a fake pi ---
// Capture the tool_result handler the extension registers.
let handler;
const fakePi = { on: (ev, h) => { if (ev === "tool_result") handler = h; }, registerTool() {}, registerCommand() {} };
factory(fakePi);
console.log("registration:");
ok("registers a tool_result handler", typeof handler === "function");

const dir = mkdtempSync(join(tmpdir(), "ve-"));
const ctx = { cwd: dir, signal: undefined };
const mkEvent = (toolName, path, isError = false) => ({
  type: "tool_result", toolName, toolCallId: "t1", isError,
  input: { path }, content: [{ type: "text", text: "wrote file" }], details: undefined,
});

const run = async () => {
  console.log("behavior:");

  // 1. Clean Python edit → no appended error (returns undefined / unchanged).
  writeFileSync(join(dir, "clean.py"), "x = 1\nprint(x)\n");
  let r = await handler(mkEvent("edit", "clean.py"), ctx);
  ok("clean python edit adds no error note", r === undefined || !JSON.stringify(r).includes("CHECK FAILED"));

  // 2. Broken Python write → error note appended in same result.
  writeFileSync(join(dir, "bad.py"), "def f(:\n  pass\n");
  r = await handler(mkEvent("write", "bad.py"), ctx);
  ok("broken python surfaces CHECK FAILED", !!r && JSON.stringify(r).includes("CHECK FAILED"));
  ok("error note preserves original content", !!r && r.content[0].text === "wrote file");

  // 3. A failed edit (isError) is never re-checked.
  r = await handler(mkEvent("edit", "bad.py", true), ctx);
  ok("isError result is skipped", r === undefined);

  // 4. Non-source file (no checker) → silent.
  writeFileSync(join(dir, "notes.xyz"), "::::");
  r = await handler(mkEvent("write", "notes.xyz"), ctx);
  ok("unknown extension stays silent", r === undefined);

  // 5. Broken JSON → caught.
  writeFileSync(join(dir, "bad.json"), "{ not: valid }");
  r = await handler(mkEvent("write", "bad.json"), ctx);
  ok("broken json surfaces CHECK FAILED", !!r && JSON.stringify(r).includes("CHECK FAILED"));

  // 6. Non-edit/write tool result (e.g. grep) ignored.
  r = await handler({ ...mkEvent("grep", "x"), toolName: "grep" }, ctx);
  ok("non-edit tool ignored", r === undefined);

  rmSync(dir, { recursive: true, force: true });
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

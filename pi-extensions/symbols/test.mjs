// Tests symbol extraction + the two tools + the read-redirect hook.
// Run: node --experimental-strip-types symbols/test.mjs
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { extractSymbols, langForExt } from "./extract.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

console.log("extract (python):");
const py = `import os
from sys import path

class Foo:
    def bar(self, x):
        return x

async def baz():
    pass
`;
let s = extractSymbols("python", py);
ok("finds 2 imports", s.filter((x) => x.kind === "import").length === 2);
ok("finds class Foo", s.some((x) => x.kind === "class" && x.name === "Foo"));
ok("finds def bar", s.some((x) => x.name === "bar"));
ok("finds async def baz", s.some((x) => x.name === "baz"));
ok("line numbers are 1-based", s.find((x) => x.name === "Foo")?.line === 4);

console.log("extract (ts/js):");
const ts = `import { x } from "./m";
export function compute(a: number): number { return a; }
export const handler = async (req) => { return req; };
export class Service {
  start() {}
}
export interface Opts { a: number; }
`;
s = extractSymbols("js", ts);
ok("finds import", s.some((x) => x.kind === "import"));
ok("finds function compute", s.some((x) => x.name === "compute"));
ok("finds arrow handler", s.some((x) => x.name === "handler"));
ok("finds class Service", s.some((x) => x.name === "Service"));
ok("finds interface Opts", s.some((x) => x.name === "Opts"));

console.log("extract (rust/go/c):");
ok("rust fn", extractSymbols("rust", "pub fn main() {}").some((x) => x.name === "main"));
ok("rust struct", extractSymbols("rust", "struct Point { x: i32 }").some((x) => x.name === "Point"));
ok("go func", extractSymbols("go", "func Handler(w http.ResponseWriter) {}").some((x) => x.name === "Handler"));
ok("c func", extractSymbols("c", "int main(int argc, char **argv) {").some((x) => x.name === "main"));

console.log("langForExt:");
ok(".py → python", langForExt(".py") === "python");
ok(".tsx → js", langForExt(".tsx") === "js");
ok(".txt → null", langForExt(".txt") === null);

// --- tool registration + behavior ---
console.log("tools:");
const tools = {};
let readHook;
const fakePi = {
  registerTool: (t) => { tools[t.name] = t; },
  registerCommand() {},
  on: (ev, h) => { if (ev === "tool_call") readHook = h; },
};
factory(fakePi);
ok("registers get_symbols", typeof tools.get_symbols?.execute === "function");
ok("registers find_symbol", typeof tools.find_symbol?.execute === "function");
ok("registers read hook", typeof readHook === "function");

const dir = mkdtempSync(join(tmpdir(), "sym-"));
const origCwd = process.cwd();
process.chdir(dir);

const run = async () => {
  // get_symbols on a real file
  writeFileSync(join(dir, "mod.py"), py);
  let r = await tools.get_symbols.execute("t", { path: "mod.py" });
  ok("get_symbols outlines mod.py", r.content[0].text.includes("class") && r.content[0].text.includes("Foo"));

  // get_symbols on unsupported file
  writeFileSync(join(dir, "data.txt"), "hello");
  r = await tools.get_symbols.execute("t", { path: "data.txt" });
  ok("get_symbols rejects unsupported type", r.content[0].text.includes("unsupported"));

  // find_symbol across project
  mkdirSync(join(dir, "pkg"));
  writeFileSync(join(dir, "pkg", "svc.ts"), "export function parseConfig(p) { return p; }\n");
  r = await tools.find_symbol.execute("t", { name: "parseConfig" }, { aborted: false });
  ok("find_symbol locates parseConfig", r.content[0].text.includes("svc.ts") && r.content[0].text.includes("parseConfig"));

  r = await tools.find_symbol.execute("t", { name: "doesNotExist" }, { aborted: false });
  ok("find_symbol reports miss helpfully", r.content[0].text.includes("No definition"));

  r = await tools.find_symbol.execute("t", { name: "  " }, { aborted: false });
  ok("find_symbol rejects empty name with teaching error", r.isError === true && r.content[0].text.includes("find_symbol("));

  // read redirect: big code file → blocked with outline
  const big = "import os\n" + Array.from({ length: 400 }, (_, i) => `def fn${i}(x):\n    return x`).join("\n");
  writeFileSync(join(dir, "big.py"), big);
  const ctx = { cwd: dir, ui: { notify() {} } };
  r = await readHook({ type: "tool_call", toolName: "read", toolCallId: "r1", input: { path: "big.py" } }, ctx);
  ok("big code read is blocked", r?.block === true);
  ok("block reason carries outline", r?.reason.includes("symbols") || r?.reason.includes("fn0"));

  // small file → not redirected
  r = await readHook({ type: "tool_call", toolName: "read", toolCallId: "r2", input: { path: "mod.py" } }, ctx);
  ok("small file read passes through", r === undefined);

  // explicit offset → never redirected (the escape hatch)
  r = await readHook({ type: "tool_call", toolName: "read", toolCallId: "r3", input: { path: "big.py", offset: 1, limit: 50 } }, ctx);
  ok("ranged read passes through", r === undefined);

  // non-code file read → not redirected
  r = await readHook({ type: "tool_call", toolName: "read", toolCallId: "r4", input: { path: "data.txt" } }, ctx);
  ok("non-code read passes through", r === undefined);

  process.chdir(origCwd);
  rmSync(dir, { recursive: true, force: true });
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

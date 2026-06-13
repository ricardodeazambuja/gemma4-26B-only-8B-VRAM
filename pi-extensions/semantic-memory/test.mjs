// Tests semantic-memory: search math, store I/O, embed parsing/fallback, and the
// full tool/hook pipeline with a fake embedding server (stubbed fetch).
// Run: node --experimental-strip-types semantic-memory/test.mjs
import { mkdtempSync, rmSync, existsSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cosine, cosineSearch, substringSearch } from "./search.ts";
import { encodeVector, decodeVector, appendChunk, loadChunks, removeChunks, appendMemoryLine, readMemoryMd, removeMemoryLine, memoryDir } from "./store.ts";
import { parseEmbeddingResponse, embed, loadFileConfig, defaultConfig } from "./embed.ts";
import { lastUserText, wrapReminder } from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

// --- config file precedence (env > file > default) ---
console.log("config precedence:");
const cfgFile = join(mkdtempSync(join(tmpdir(), "ecfg-")), "embed-config.json");
process.env.PI_EMBED_CONFIG = cfgFile;
delete process.env.PI_EMBED_URL; delete process.env.PI_EMBED_MODEL;
ok("no file → built-in default :8081", defaultConfig().url.includes(":8081"));
ok("missing file → empty partial", Object.keys(loadFileConfig()).length === 0);
writeFileSync(cfgFile, JSON.stringify({ url: "http://127.0.0.1:11434/v1/embeddings", model: "embeddinggemma" }));
ok("file config is read", defaultConfig().url.includes(":11434") && defaultConfig().model === "embeddinggemma");
process.env.PI_EMBED_URL = "http://env-wins:9/v1/embeddings";
ok("env overrides file", defaultConfig().url.includes("env-wins"));
delete process.env.PI_EMBED_URL; delete process.env.PI_EMBED_CONFIG;
rmSync(cfgFile, { force: true });

// --- search ---
console.log("search:");
ok("cosine of identical = 1", Math.abs(cosine([1, 2, 3], [1, 2, 3]) - 1) < 1e-9);
ok("cosine of orthogonal = 0", Math.abs(cosine([1, 0], [0, 1])) < 1e-9);
const cs = cosineSearch([1, 0, 0], [
  { id: "a", text: "a", source: "", date: "", vector: [1, 0, 0] },
  { id: "b", text: "b", source: "", date: "", vector: [0, 1, 0] },
  { id: "c", text: "c", source: "", date: "", vector: [0.9, 0.1, 0] },
], 2, 0.5);
ok("cosineSearch ranks + thresholds", cs.length === 2 && cs[0].chunk.id === "a" && cs[1].chunk.id === "c");
ok("cosineSearch skips vectorless chunks", cosineSearch([1, 0], [{ id: "x", text: "x", source: "", date: "" }], 3, 0).length === 0);
const ss = substringSearch("database schema design", [
  { id: "1", text: "the database schema", source: "", date: "" },
  { id: "2", text: "unrelated note", source: "", date: "" },
], 5);
ok("substringSearch ranks by term overlap", ss[0].chunk.id === "1");

// --- lastUserText (skips injected <reminder> blocks so the query is the real request) ---
console.log("lastUserText:");
ok("plain string content works", lastUserText([{ role: "user", content: "plain string" }]) === "plain string");
ok("returns the real text when reminders ride underneath it (folded turn)",
  lastUserText([{ role: "user", content: [{ type: "text", text: "the real question" }, { type: "text", text: wrapReminder("## Active plan\n[ ] step") }] }]) === "the real question");
ok("walks back past an all-reminder tail user turn (tool loop)",
  lastUserText([
    { role: "user", content: [{ type: "text", text: "where is the schema?" }] },
    { role: "toolResult", content: [{ type: "text", text: "out" }] },
    { role: "user", content: [{ type: "text", text: wrapReminder("## Possibly relevant memory\n- noise") }] },
  ]) === "where is the schema?");

// --- store ---
console.log("store:");
const v = [0.1, -0.2, 0.333, 1e-3, -5];
const rt = decodeVector(encodeVector(v));
ok("vector base64 roundtrip (Float32 precision)", rt.every((x, i) => Math.abs(x - v[i]) < 1e-4));
// alignment torture: many encodes shouldn't throw
let aligned = true;
try { for (let i = 0; i < 20; i++) decodeVector(encodeVector(Array.from({ length: 768 }, (_, j) => (i + j) * 0.01))); } catch { aligned = false; }
ok("decode never hits alignment error", aligned);

const dir = mkdtempSync(join(tmpdir(), "mem-"));
appendChunk(dir, { id: "1", text: "first", source: "remember", date: "2026-06-10", vector: [1, 0] });
appendChunk(dir, { id: "2", text: "second", source: "remember", date: "2026-06-10" });
let loaded = loadChunks(dir);
ok("append+load roundtrips chunks", loaded.length === 2 && loaded[0].text === "first");
ok("vectorless chunk loads without vector", loaded[1].vector === undefined);
const removed = removeChunks(dir, (c) => c.id === "1");
ok("removeChunks removes + rewrites", removed === 1 && loadChunks(dir).length === 1);

appendMemoryLine(dir, "Entry point is src/main.py");
appendMemoryLine(dir, "Uses Python 3.11");
ok("MEMORY.md gets a header + lines", readMemoryMd(dir).includes("# Project memory") && readMemoryMd(dir).includes("src/main.py"));
ok("readMemoryMd caps size", readMemoryMd(dir, 10).length <= 30);
ok("removeMemoryLine removes matching", removeMemoryLine(dir, "Python 3.11") && !readMemoryMd(dir).includes("3.11"));

// --- embed ---
console.log("embed:");
ok("parses OpenAI shape", JSON.stringify(parseEmbeddingResponse({ data: [{ embedding: [1, 2] }] })) === "[1,2]");
ok("parses llama.cpp shape", JSON.stringify(parseEmbeddingResponse({ embedding: [3, 4] })) === "[3,4]");
ok("parses batch array shape", JSON.stringify(parseEmbeddingResponse([{ embedding: [5, 6] }])) === "[5,6]");
ok("returns null on junk", parseEmbeddingResponse({ nope: 1 }) === null);

// dead server → null, no throw
const deadCfg = { url: "http://127.0.0.1:1/embeddings", timeoutMs: 500 };
const deadResult = await embed("hi", deadCfg);
ok("embed returns null when server is down", deadResult === null);

// injected fetch → works
const fakeFetch = async () => ({ ok: true, json: async () => ({ data: [{ embedding: [0.5, 0.5] }] }) });
ok("embed works with a live (fake) server", JSON.stringify(await embed("hi", deadCfg, fakeFetch)) === "[0.5,0.5]");

// --- full pipeline with fake embedding server (stub global fetch) ---
console.log("pipeline:");
// deterministic embedder: 26-dim letter-frequency bag, so identical text → identical vector.
function vecOf(text) {
  const v = new Array(26).fill(0);
  for (const ch of text.toLowerCase()) { const c = ch.charCodeAt(0) - 97; if (c >= 0 && c < 26) v[c]++; }
  return v;
}
globalThis.fetch = async (_url, opts) => {
  const body = JSON.parse(opts.body);
  return { ok: true, json: async () => ({ data: [{ embedding: vecOf(body.input || body.content || "") }] }) };
};

const mod = (await import("./index.ts")).default;
const tools = {}; const hooks = {};
mod({ registerTool: (t) => (tools[t.name] = t), registerCommand() {}, on: (e, h) => (hooks[e] = h) });

const pdir = mkdtempSync(join(tmpdir(), "memx-"));
const ctx = { cwd: pdir, sessionManager: { getSessionDir: () => pdir, getSessionId: () => "s" } };

const run = async () => {
  await hooks.session_start({}, ctx);

  let r = await tools.remember.execute("t", { fact: "" });
  ok("remember rejects empty fact", r.isError);
  r = await tools.remember.execute("t", { fact: "The database schema lives in db/schema.sql" });
  ok("remember stores a fact", !r.isError && r.content[0].text.startsWith("Remembered"));
  await tools.remember.execute("t", { fact: "The web server runs on port 8080" });

  r = await tools.recall.execute("t", { query: "database schema" });
  ok("recall finds the relevant fact", r.content[0].text.includes("db/schema.sql"));

  // auto-recall via context hook — folds into the trailing user turn as a wrapped reminder
  const msgs = [
    { role: "user", content: [{ type: "text", text: "first question" }] },
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
    { role: "user", content: [{ type: "text", text: "where is the database schema?" }] },
  ];
  r = await hooks.context({ messages: msgs });
  ok("auto-recall folds into the user turn (no new message)", r && r.messages.length === msgs.length);
  ok("the user's question stays first", r.messages.at(-1).content[0].text === "where is the database schema?");
  ok("recall rides as a wrapped <reminder> block", r.messages.at(-1).content.at(-1).text.includes("schema") && r.messages.at(-1).content.at(-1).text.startsWith("<reminder>"));
  ok("earlier messages untouched (same refs)", r.messages[0] === msgs[0] && r.messages[1] === msgs[1]);

  // MEMORY.md passive injection (uses session_start snapshot; re-run session_start to refresh)
  await hooks.session_start({}, ctx);
  r = await hooks.before_agent_start({ systemPrompt: "BASE" });
  ok("before_agent_start injects MEMORY.md", r && r.systemPrompt.includes("BASE") && r.systemPrompt.includes("Project memory"));
  const r2 = await hooks.before_agent_start({ systemPrompt: "BASE" });
  ok("injection is byte-stable across turns", r2.systemPrompt === r.systemPrompt);

  // forget
  r = await tools.forget.execute("t", { match: "port 8080" });
  ok("forget removes matching fact", r.content[0].text.includes("Forgot"));
  r = await tools.recall.execute("t", { query: "web server port" });
  ok("forgotten fact no longer recalled", !r.content[0].text.includes("8080"));

  // snapshot ingestion
  const snapDir = join(memoryDir(pdir), "snapshots");
  mkdirSync(snapDir, { recursive: true });
  writeFileSync(join(snapDir, "s_1.md"), "Task: build the parser\nFiles touched: parser.ts");
  await hooks.session_start({}, ctx); // triggers ingestSnapshots
  await new Promise((res) => setTimeout(res, 50));
  r = await tools.recall.execute("t", { query: "build the parser" });
  ok("ingested snapshot becomes recallable", r.content[0].text.includes("parser"));
  ok("snapshot moved to ingested/", existsSync(join(snapDir, "ingested")) && readdirSync(join(snapDir, "ingested")).length === 1);

  rmSync(dir, { recursive: true, force: true });
  rmSync(pdir, { recursive: true, force: true });
  rmSync(memoryDir(pdir), { recursive: true, force: true });
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
};
run();

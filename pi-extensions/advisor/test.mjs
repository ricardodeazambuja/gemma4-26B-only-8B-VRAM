// Tests the advisor extension: config precedence, transcript formatting,
// prompt building, reply capping, and the consult() flow with a fake runner
// (never touches tmux/tui-driver).
// Run: node --experimental-strip-types advisor/test.mjs
import { mkdtempSync, writeFileSync, rmSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  configPath, loadFileConfig, defaultConfig, DEFAULT_PROMPT_TEMPLATE,
  formatTranscript, buildPrompt, capReply, consult,
} from "./index.ts";
import factory from "./index.ts";

let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log(`  ✓ ${name}`); } else { fail++; console.log(`  ✗ ${name}`); } };

const tmp = mkdtempSync(join(tmpdir(), "advisor-test-"));

// --- config precedence: env > file > default -------------------------------
console.log("config:");
const cfgFile = join(tmp, "advisor-config.json");
const savedEnv = { ...process.env };
const clearEnv = () => {
  for (const k of Object.keys(process.env)) if (k.startsWith("PI_ADVISOR")) delete process.env[k];
};

clearEnv();
process.env.PI_ADVISOR_CONFIG = join(tmp, "missing.json");
let cfg = defaultConfig();
ok("no file, no env → built-in defaults", cfg.command === "" && cfg.tuiDriver === "tui-driver" && cfg.timeoutSec === 600);
ok("default keepSession true, inline false", cfg.keepSession === true && cfg.inlineTranscript === false);
ok("default template used", cfg.promptTemplate === DEFAULT_PROMPT_TEMPLATE);

writeFileSync(cfgFile, JSON.stringify({ command: "gemini", timeoutSec: 99, keepSession: false, promptTemplate: "T {transcript} {focus}" }));
process.env.PI_ADVISOR_CONFIG = cfgFile;
cfg = defaultConfig();
ok("file layer wins over defaults", cfg.command === "gemini" && cfg.timeoutSec === 99 && cfg.keepSession === false);
ok("file template wins", cfg.promptTemplate === "T {transcript} {focus}");

process.env.PI_ADVISOR_CMD = "claude";
process.env.PI_ADVISOR_TIMEOUT_SEC = "42";
process.env.PI_ADVISOR_KEEP_SESSION = "1";
cfg = defaultConfig();
ok("env layer wins over file", cfg.command === "claude" && cfg.timeoutSec === 42 && cfg.keepSession === true);

writeFileSync(cfgFile, "not json {");
ok("corrupt config file → {}", Object.keys(loadFileConfig(cfgFile)).length === 0);
ok("configPath honors PI_ADVISOR_CONFIG", configPath() === cfgFile);
clearEnv();
Object.assign(process.env, savedEnv);

// --- transcript formatting --------------------------------------------------
console.log("transcript:");
const entries = [
  { type: "message", message: { role: "user", content: "fix the bug" } },
  { type: "message", message: { role: "assistant", content: [
    { type: "thinking", thinking: "hmm ".repeat(400) },
    { type: "text", text: "Looking at it." },
    { type: "toolCall", id: "t1", name: "bash", arguments: { command: "x".repeat(2000) } },
  ] } },
  { type: "message", message: { role: "toolResult", toolName: "bash", isError: true,
    content: [{ type: "text", text: "boom ".repeat(1000) }] } },
  { type: "message", message: { role: "bashExecution", command: "ls", output: "a\nb" } },
  { type: "custom_message", customType: "plan", content: "step 1" },
  { type: "message", message: { role: "compactionSummary", summary: "earlier stuff" } },
];
const t = formatTranscript(entries, 200);
ok("header counts entries", t.includes("SESSION TRANSCRIPT (6 entries)"));
ok("user text present", t.includes("[user]\nfix the bug"));
ok("assistant text present", t.includes("Looking at it."));
ok("thinking clipped to ~600", t.includes("<thinking>") && !t.includes("hmm ".repeat(200)));
ok("tool call args clipped to ~400", t.includes("→ tool bash(") && !t.includes("x".repeat(500)));
ok("tool result marked ERROR and clipped", t.includes("[tool result: bash (ERROR)]") && t.includes("chars truncated"));
ok("bashExecution rendered", t.includes("[user ran: ls]"));
ok("custom_message rendered", t.includes("[extension:plan]\nstep 1"));
ok("compaction summary rendered", t.includes("[compactionSummary]\nearlier stuff"));
ok("entries without message skipped", formatTranscript([{ type: "label" }]).includes("(1 entries)"));

// --- prompt building / reply capping ----------------------------------------
console.log("prompt + caps:");
ok("placeholders substituted", buildPrompt("read {transcript} now. {focus}", "/tmp/x.md", "perf")
  === "read /tmp/x.md now. Focus on: perf");
ok("empty focus leaves no residue", buildPrompt("a {transcript} {focus}", "p", "") === "a p");
ok("short reply untouched", capReply("hi", 100, "/f") === "hi");
const capped = capReply("z".repeat(200), 100, "/full.md");
ok("long reply capped with pointer", capped.startsWith("z".repeat(100)) && capped.includes("/full.md"));

// --- consult() flow with a fake runner ---------------------------------------
console.log("consult:");
const baseCfg = {
  command: "fakeagent", tuiDriver: "tui-driver", timeoutSec: 5, keepSession: true,
  inlineTranscript: false, promptTemplate: "Read {transcript}. {focus}",
  maxToolResultChars: 1500, maxInlineChars: 60000, maxReplyChars: 12000,
};
const fakeRunner = (script) => {
  const calls = [];
  const run = async (args) => {
    calls.push(args);
    const action = args[1];
    const r = script[action] ?? { stdout: "", stderr: "", code: 0 };
    return typeof r === "function" ? r(args) : r;
  };
  return { run, calls };
};

let r = await consult({ ...baseCfg, command: "" }, async () => ({ stdout: "", stderr: "", code: 0 }), [], "", "t0");
ok("unconfigured → teaching error", r.isError && r.text.includes("advisor-config.json") && r.text.includes('"command"'));

let f = fakeRunner({ status: { stdout: "stopped\n", code: 0, stderr: "" },
                     start: { stdout: "", code: 0, stderr: "" },
                     send: { stdout: "Solid plan. Watch the lock file.\n", code: 0, stderr: "" } });
r = await consult(baseCfg, f.run, entries, "locking", "t1");
ok("happy path: not an error", !r.isError);
ok("reply surfaced with agent name", r.text.includes("ADVISOR (fakeagent)") && r.text.includes("Watch the lock file."));
ok("stopped → status, start, send (no stop)", f.calls.map(c => c[1]).join(",") === "status,start,send");
const sendPrompt = f.calls[2][2];
ok("prompt has transcript path + focus", /Read \/.*pi-advisor-t1\.md\. Focus on: locking/.test(sendPrompt));
const written = readFileSync(sendPrompt.match(/Read (\S+)\./)[1], "utf8");
ok("transcript file written with session content", written.includes("fix the bug"));

f = fakeRunner({ status: { stdout: "running\n", code: 0, stderr: "" },
                 send: { stdout: "ok\n", code: 0, stderr: "" } });
r = await consult({ ...baseCfg, keepSession: false }, f.run, [], "", "t2");
ok("running → no start; keepSession=false → stop", f.calls.map(c => c[1]).join(",") === "status,send,stop");

f = fakeRunner({ status: { stdout: "stopped\n", code: 0, stderr: "" },
                 start: { stdout: "", stderr: "tmux: not found", code: 1 } });
r = await consult(baseCfg, f.run, [], "", "t3");
ok("start failure → teaching error", r.isError && r.text.includes("failed to start") && r.text.includes("tmux"));

f = fakeRunner({ status: { stdout: "running\n", code: 0, stderr: "" },
                 send: { stdout: "   \n", stderr: "timeout", code: 1 } });
r = await consult(baseCfg, f.run, [], "", "t4");
ok("empty reply → error with retry hint", r.isError && r.text.includes("no reply") && r.text.includes("timeoutSec"));

f = fakeRunner({ status: { stdout: "running\n", code: 0, stderr: "" },
                 send: { stdout: "partial answer", code: 1, stderr: "" } });
r = await consult(baseCfg, f.run, [], "", "t5");
ok("timeout with partial reply → returned with note", !r.isError && r.text.includes("may be incomplete"));

f = fakeRunner({ status: { stdout: "running\n", code: 0, stderr: "" },
                 send: { stdout: "ok", code: 0, stderr: "" } });
r = await consult({ ...baseCfg, inlineTranscript: true, maxInlineChars: 80 }, f.run, entries, "", "t6");
ok("inline mode pastes (clipped) transcript text", f.calls[1][2].includes("SESSION TRANSCRIPT") && f.calls[1][2].includes("chars truncated"));

f = fakeRunner({ status: { stdout: "running\n", code: 0, stderr: "" },
                 send: { stdout: "y".repeat(20000), code: 0, stderr: "" } });
r = await consult(baseCfg, f.run, [], "", "t7");
ok("long reply capped with saved-file pointer", r.text.includes("reply truncated") && /pi-advisor-t7-reply\.md/.test(r.text));
ok("full reply persisted", readFileSync(join(tmpdir(), "pi-advisor-t7-reply.md"), "utf8").length === 20000);

// --- registration -------------------------------------------------------------
console.log("registration:");
let tool;
factory({ registerTool: (def) => { tool = def; }, on() {}, registerCommand() {} });
ok("registers a tool named advisor", tool?.name === "advisor");
ok("only optional focus param", JSON.stringify(Object.keys(tool.parameters.properties)) === '["focus"]');
ok("one-line description", typeof tool.description === "string" && !tool.description.includes("\n"));
ok("guideline names the tool", tool.promptGuidelines?.[0]?.includes("advisor"));

rmSync(tmp, { recursive: true, force: true });
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

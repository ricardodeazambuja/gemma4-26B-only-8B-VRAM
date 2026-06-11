#!/usr/bin/env node
// diffusion-shim — OpenAI-compatible HTTP front for llama-diffusion-cli (PR #24423).
//
// llama-server cannot generate from the diffusiongemma arch yet; the PR ships a
// CLI only. This shim keeps ONE llama-diffusion-cli process alive (model stays
// loaded) in our patched JSONL stdio mode and translates:
//
//   POST /v1/chat/completions  <->  {"messages":[...]}\n on the child's stdin
//                                    {"content":...,"ms":...}\n on its stdout
//
// pi then talks to it as a normal openai-completions provider. Zero npm deps.
//
// Env:
//   DGEMMA_BIN    path to llama-diffusion-cli            (required)
//   DGEMMA_MODEL  path to the diffusiongemma GGUF        (required)
//   DGEMMA_ARGS   extra CLI args, space-separated        (default: "")
//   DGEMMA_PORT   listen port                            (default: 8082)
//
// Limitations (deliberate, first pass):
//   - no tool calls: "tools" in requests are ignored; tool-role messages are
//     folded into user messages so the template never sees an unknown role.
//   - one request at a time (single canvas); concurrent requests queue.
//   - streaming is emulated: one SSE chunk with the whole reply.

import { spawn } from "node:child_process";
import { createServer } from "node:http";

const BIN = process.env.DGEMMA_BIN;
const MODEL = process.env.DGEMMA_MODEL;
const PORT = Number(process.env.DGEMMA_PORT || 8082);
const EXTRA = (process.env.DGEMMA_ARGS || "").split(/\s+/).filter(Boolean);
const MODEL_ID = "diffusiongemma-26b-a4b";

if (!BIN || !MODEL) {
  console.error("DGEMMA_BIN and DGEMMA_MODEL are required");
  process.exit(1);
}

// -cnv is required: the JSONL branch lives inside conversation mode.
const args = ["-m", MODEL, "-cnv", ...EXTRA];
console.error(`[shim] spawning: ${BIN} ${args.join(" ")}`);
const child = spawn(BIN, args, {
  env: { ...process.env, LLAMA_DIFFUSION_JSONL: "1" },
  stdio: ["pipe", "pipe", "inherit"], // stderr (model load logs) passes through
});
child.on("exit", (code) => {
  console.error(`[shim] llama-diffusion-cli exited (${code}); shutting down`);
  process.exit(code ?? 1);
});

let ready = false;
let buf = "";
const pending = []; // FIFO of resolve callbacks awaiting one JSON line each

child.stdout.on("data", (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line.startsWith("{")) continue; // load-progress noise etc.
    let obj;
    try { obj = JSON.parse(line); } catch { continue; }
    if (obj.ready) { ready = true; console.error("[shim] model ready"); continue; }
    pending.shift()?.(obj);
  }
});

// Serialize requests: one canvas, one generation at a time.
let chain = Promise.resolve();
function generate(messages) {
  const p = chain.then(() => new Promise((resolve) => {
    pending.push(resolve);
    child.stdin.write(JSON.stringify({ messages }) + "\n");
  }));
  chain = p.catch(() => {});
  return p;
}

// pi sends content as a string or as [{type:"text",text}...] blocks; the
// template only takes strings, and only user/assistant/system roles.
function flatten(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((b) => (b?.type === "text" ? b.text : "")).filter(Boolean).join("\n");
  }
  return String(content ?? "");
}
function normalize(messages) {
  const out = [];
  for (const m of messages || []) {
    let role = m.role;
    let text = flatten(m.content);
    if (role === "tool") { role = "user"; text = `Tool result:\n${text}`; }
    if (m.tool_calls) text += `\n[requested tool calls: ${JSON.stringify(m.tool_calls)}]`;
    if (!["system", "user", "assistant"].includes(role)) role = "user";
    if (!text) continue;
    // Gemma templates reject consecutive same-role turns; merge them.
    const prev = out[out.length - 1];
    if (prev && prev.role === role) prev.content += `\n\n${text}`;
    else out.push({ role, content: text });
  }
  return out;
}

// DiffusionGemma emits its reasoning in-band as <|channel>thought ... <channel|>;
// strip it so pi sees only the visible reply.
function stripThought(s) {
  return s.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, "").trim();
}

const estTok = (s) => Math.max(1, Math.round((s?.length ?? 0) / 4));

function completionBody(content, promptChars) {
  return {
    id: `chatcmpl-dg-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: MODEL_ID,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
    usage: {
      prompt_tokens: estTok(promptChars),
      completion_tokens: estTok(content),
      total_tokens: estTok(promptChars) + estTok(content),
    },
  };
}

const server = createServer(async (req, res) => {
  const send = (code, obj) => {
    res.writeHead(code, { "Content-Type": "application/json" });
    res.end(JSON.stringify(obj));
  };

  if (req.method === "GET" && req.url === "/health") {
    return send(ready ? 200 : 503, { status: ready ? "ok" : "loading" });
  }
  if (req.method === "GET" && req.url?.startsWith("/v1/models")) {
    return send(200, { object: "list", data: [{ id: MODEL_ID, object: "model" }] });
  }
  if (req.method !== "POST" || !req.url?.startsWith("/v1/chat/completions")) {
    return send(404, { error: { message: "not found" } });
  }
  if (!ready) return send(503, { error: { message: "model still loading" } });

  let body = "";
  for await (const c of req) body += c;
  let parsed;
  try { parsed = JSON.parse(body); } catch { return send(400, { error: { message: "bad json" } }); }

  const messages = normalize(parsed.messages);
  const promptChars = messages.map((m) => m.content).join("");
  const t0 = Date.now();
  const result = await generate(messages);
  if (result.error) return send(500, { error: { message: result.error } });
  result.content = stripThought(result.content ?? "");
  console.error(`[shim] turn done in ${((Date.now() - t0) / 1000).toFixed(1)}s (model ${result.ms?.toFixed(0)}ms)`);

  if (parsed.stream) {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const base = { id: `chatcmpl-dg-${t0}`, object: "chat.completion.chunk", created: Math.floor(t0 / 1000), model: MODEL_ID };
    const chunk = (delta, finish = null) =>
      res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
    chunk({ role: "assistant" });
    chunk({ content: result.content });
    chunk({}, "stop");
    res.write("data: [DONE]\n\n");
    return res.end();
  }
  return send(200, completionBody(result.content, promptChars));
});

server.listen(PORT, "127.0.0.1", () => console.error(`[shim] listening on http://127.0.0.1:${PORT}/v1`));
process.on("SIGINT", () => { child.kill(); process.exit(0); });
process.on("SIGTERM", () => { child.kill(); process.exit(0); });

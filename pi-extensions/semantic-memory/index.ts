import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { join } from "node:path";
import { readdirSync, existsSync, readFileSync, renameSync, mkdirSync } from "node:fs";
import { embed, defaultConfig, type EmbedConfig } from "./embed.ts";
import { cosineSearch, substringSearch, type Chunk } from "./search.ts";
import {
  memoryDir, loadChunks, appendChunk, removeChunks,
  appendMemoryLine, readMemoryMd, removeMemoryLine, newId,
} from "./store.ts";

// semantic-memory — cross-session memory with NO retrieval burden on the model.
// remember() writes a fact; recall() searches; but the real work is auto-recall:
// the extension embeds each incoming user message itself and injects the most
// relevant chunks at the TAIL, so Gemma never has to know to ask. MEMORY.md is
// injected passively (byte-stable) every session. PLAN.md item 5.

const RECALL_K = 3;
const AUTO_RECALL_K = 2;
const AUTO_RECALL_THRESHOLD = 0.55;
const CHUNK_INJECT_CHARS = 600; // ~150 tokens per chunk cap (R3)

function lastUserText(messages: { role: string; content: unknown }[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "user") continue;
    if (typeof m.content === "string") return m.content;
    if (Array.isArray(m.content)) {
      const t = m.content.find((c: any) => c?.type === "text");
      if (t) return (t as any).text || "";
    }
  }
  return "";
}

function clip(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

export default function (pi: ExtensionAPI) {
  const cfg: EmbedConfig = defaultConfig();
  let dir = "";
  let chunks: Chunk[] = [];
  let memoryMdSnapshot = ""; // frozen at session_start so injection stays byte-stable

  const reload = () => { chunks = dir ? loadChunks(dir) : []; };

  // Add a fact/snapshot to the index. Embeds if the server is up; stores without a
  // vector otherwise (lazily embeddable on a later pass).
  const addChunk = async (text: string, source: string): Promise<Chunk> => {
    const vector = (await embed(text, cfg)) ?? undefined;
    const c: Chunk = { id: newId(), text: text.trim(), source, date: new Date().toISOString().slice(0, 10), vector };
    appendChunk(dir, c);
    chunks.push(c);
    return c;
  };

  // Ingest plan's pre-compaction snapshots once, then move them aside so they
  // aren't ingested again.
  const ingestSnapshots = async () => {
    const snapDir = join(dir, "snapshots");
    if (!existsSync(snapDir)) return;
    let files: string[] = [];
    try { files = readdirSync(snapDir).filter((f) => f.endsWith(".md")); } catch { return; }
    if (!files.length) return;
    const doneDir = join(snapDir, "ingested");
    mkdirSync(doneDir, { recursive: true });
    for (const f of files) {
      try {
        const text = readFileSync(join(snapDir, f), "utf8");
        await addChunk(text, `snapshot:${f}`);
        renameSync(join(snapDir, f), join(doneDir, f));
      } catch {}
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    dir = memoryDir(ctx.cwd);
    reload();
    memoryMdSnapshot = readMemoryMd(dir, 1024);
    // best-effort: pull in any snapshots left by previous sessions
    ingestSnapshots().catch(() => {});
  });

  // Passive injection of MEMORY.md into the system prompt. Uses the session_start
  // snapshot (not a re-read) so the text is byte-stable all session → cache holds.
  pi.on("before_agent_start", async (event) => {
    if (!memoryMdSnapshot.trim()) return;
    return {
      systemPrompt: `${event.systemPrompt}\n\n## Project memory (from past sessions)\n${memoryMdSnapshot}`,
    };
  });

  // Auto-recall: embed the latest user message, inject top matches at the TAIL.
  pi.on("context", async (event) => {
    if (!chunks.length) return;
    const query = lastUserText(event.messages as any);
    if (!query.trim()) return;
    const qv = await embed(query, cfg);
    const hits = qv
      ? cosineSearch(qv, chunks, AUTO_RECALL_K, AUTO_RECALL_THRESHOLD)
      : []; // no embedding server → skip auto-recall (don't spam substring noise unprompted)
    if (!hits.length) return;
    const body = hits.map((h) => `- ${clip(h.chunk.text, CHUNK_INJECT_CHARS)}`).join("\n");
    const text = `## Possibly relevant memory\n${body}`;
    const reminder = { role: "user" as const, content: [{ type: "text" as const, text }] };
    return { messages: [...(event.messages as any), reminder] };
  });

  pi.registerTool({
    name: "remember",
    label: "Remember",
    description: "Save a durable fact for future sessions. One concrete line; include file paths/versions where relevant.",
    parameters: Type.Object({ fact: Type.String({ description: "The fact to remember, as a single concrete line" }) }),
    async execute(_id, params) {
      const fact = (params.fact || "").trim();
      if (!fact) return err(`remember needs a non-empty fact, e.g. remember(fact="Entry point is src/main.py; Python 3.11").`);
      if (fact.length > 300) return err(`Fact too long (${fact.length} chars). Keep it to one line under 300 chars; split big notes.`);
      appendMemoryLine(dir, fact);
      const c = await addChunk(fact, "remember");
      const note = c.vector ? "" : " (stored; embedding server offline, will index later)";
      return { content: [{ type: "text", text: `Remembered${note}.` }] };
    },
  });

  pi.registerTool({
    name: "recall",
    label: "Recall",
    description: "Search memory from past sessions for facts relevant to a query.",
    parameters: Type.Object({
      query: Type.String({ description: "What to look up" }),
      k: Type.Optional(Type.Number({ description: `Max results (default ${RECALL_K})` })),
    }),
    async execute(_id, params) {
      const query = (params.query || "").trim();
      if (!query) return err(`recall needs a query, e.g. recall(query="database schema").`);
      if (!chunks.length) return { content: [{ type: "text", text: "Memory is empty — nothing remembered yet." }] };
      const k = Math.min(Math.max(1, params.k || RECALL_K), 10);
      const qv = await embed(query, cfg);
      const hits = qv ? cosineSearch(qv, chunks, k, 0) : substringSearch(query, chunks, k);
      if (!hits.length) return { content: [{ type: "text", text: `No memory matches "${query}".` }] };
      const mode = qv ? "" : " (substring fallback — embedding server offline)";
      const body = hits.map((h) => `- [${h.chunk.date}] ${h.chunk.text}`).join("\n");
      return { content: [{ type: "text", text: `Recall${mode}:\n${body}` }] };
    },
  });

  pi.registerTool({
    name: "forget",
    label: "Forget",
    description: "Remove a remembered fact by a substring of its text. Wrong facts are worse than missing ones.",
    parameters: Type.Object({ match: Type.String({ description: "Substring identifying the fact to remove" }) }),
    async execute(_id, params) {
      const match = (params.match || "").trim();
      if (!match) return err(`forget needs a substring, e.g. forget(match="Python 3.11").`);
      const removedChunks = removeChunks(dir, (c) => c.text.toLowerCase().includes(match.toLowerCase()));
      const removedLine = removeMemoryLine(dir, match);
      reload();
      if (!removedChunks && !removedLine) return { content: [{ type: "text", text: `Nothing matched "${match}".` }] };
      return { content: [{ type: "text", text: `Forgot ${removedChunks} indexed item(s)${removedLine ? " and 1 MEMORY.md line" : ""}.` }] };
    },
  });
}

function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}

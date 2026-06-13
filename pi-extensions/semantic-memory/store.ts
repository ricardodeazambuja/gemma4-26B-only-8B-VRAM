// Persistence for memory chunks: one JSON object per line in chunks.jsonl, with
// the vector stored as base64-packed Float32 to keep the file compact. Also owns
// the curated MEMORY.md file.

import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync, existsSync, readFileSync, writeFileSync, appendFileSync } from "node:fs";
import type { Chunk } from "./search.ts";

export function memoryDir(cwd: string): string {
  const slug = cwd.replace(/^\/+/, "").replace(/[^a-zA-Z0-9]+/g, "-").replace(/-+$/g, "") || "root";
  return join(homedir(), ".pi", "memory", slug);
}

export function chunksPath(dir: string): string { return join(dir, "chunks.jsonl"); }
export function memoryMdPath(dir: string): string { return join(dir, "MEMORY.md"); }

// --- vector <-> base64 (Float32) ---
export function encodeVector(vec: number[]): string {
  const f = new Float32Array(vec);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength).toString("base64");
}
export function decodeVector(b64: string): number[] {
  const buf = Buffer.from(b64, "base64");
  // Copy into a fresh, 4-byte-aligned buffer: Buffer's pooled .buffer may have a
  // byteOffset that isn't a multiple of 4, which would make the Float32Array view throw.
  const copy = new Uint8Array(buf.byteLength);
  copy.set(buf);
  const f = new Float32Array(copy.buffer, 0, Math.floor(buf.byteLength / 4));
  return Array.from(f);
}

// On-disk record shape (vector packed as base64 string `v`).
interface RawChunk { id: string; text: string; source: string; date: string; v?: string; }

function toRaw(c: Chunk): RawChunk {
  return { id: c.id, text: c.text, source: c.source, date: c.date, v: c.vector ? encodeVector(c.vector) : undefined };
}
function fromRaw(r: RawChunk): Chunk {
  return { id: r.id, text: r.text, source: r.source, date: r.date, vector: r.v ? decodeVector(r.v) : undefined };
}

export function ensureDir(dir: string): void {
  mkdirSync(dir, { recursive: true });
}

export function loadChunks(dir: string): Chunk[] {
  const p = chunksPath(dir);
  if (!existsSync(p)) return [];
  const out: Chunk[] = [];
  for (const line of readFileSync(p, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try { out.push(fromRaw(JSON.parse(line))); } catch {}
  }
  return out;
}

export function appendChunk(dir: string, c: Chunk): void {
  ensureDir(dir);
  appendFileSync(chunksPath(dir), JSON.stringify(toRaw(c)) + "\n");
}

/** Remove chunks matching predicate; rewrites the file. Returns count removed. */
export function removeChunks(dir: string, pred: (c: Chunk) => boolean): number {
  const all = loadChunks(dir);
  const kept = all.filter((c) => !pred(c));
  const removed = all.length - kept.length;
  if (removed > 0) writeFileSync(chunksPath(dir), kept.map((c) => JSON.stringify(toRaw(c)) + "\n").join(""));
  return removed;
}

// --- MEMORY.md (curated, human-editable, passively injected) ---
export function appendMemoryLine(dir: string, line: string): void {
  ensureDir(dir);
  const p = memoryMdPath(dir);
  const header = existsSync(p) ? "" : "# Project memory\n\n";
  appendFileSync(p, `${header}- ${line.trim()}\n`);
}

export function readMemoryMd(dir: string, capBytes = 1024): string {
  const p = memoryMdPath(dir);
  if (!existsSync(p)) return "";
  let text = readFileSync(p, "utf8");
  if (text.length > capBytes) text = text.slice(0, capBytes) + "\n…(truncated)";
  return text;
}

export function removeMemoryLine(dir: string, match: string): boolean {
  const p = memoryMdPath(dir);
  if (!existsSync(p)) return false;
  const lines = readFileSync(p, "utf8").split("\n");
  const idx = lines.findIndex((l) => l.startsWith("- ") && l.toLowerCase().includes(match.toLowerCase()));
  if (idx === -1) return false;
  lines.splice(idx, 1);
  writeFileSync(p, lines.join("\n"));
  return true;
}

let counter = 0;
export function newId(): string {
  counter = (counter + 1) % 100000;
  return `${Date.now().toString(36)}-${counter.toString(36)}`;
}

// Embedding client for a local llama-server (EmbeddingGemma). Talks to either the
// OpenAI-compatible `/v1/embeddings` or llama.cpp's native `/embedding`. Designed
// to fail soft: any error returns null and the caller degrades to substring
// search rather than erroring Gemma's turn. PLAN.md item 5.

import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";

export interface EmbedConfig {
  url: string;
  model?: string;
  timeoutMs: number;
}

// Persistent config written by setup-embeddings.sh, so the backend survives across
// pi launches without env vars or start.sh edits. Precedence: env > file > default.
export function configPath(): string {
  return process.env.PI_EMBED_CONFIG || join(homedir(), ".pi", "agent", "embed-config.json");
}

export function loadFileConfig(path = configPath()): Partial<EmbedConfig> {
  if (!existsSync(path)) return {};
  try {
    const c = JSON.parse(readFileSync(path, "utf8"));
    const out: Partial<EmbedConfig> = {};
    if (typeof c.url === "string") out.url = c.url;
    if (typeof c.model === "string") out.model = c.model;
    if (Number.isFinite(c.timeoutMs)) out.timeoutMs = c.timeoutMs;
    return out;
  } catch { return {}; }
}

export function defaultConfig(): EmbedConfig {
  const file = loadFileConfig();
  return {
    url: process.env.PI_EMBED_URL || file.url || "http://127.0.0.1:8081/v1/embeddings",
    model: process.env.PI_EMBED_MODEL || file.model || "embeddinggemma",
    timeoutMs: Number(process.env.PI_EMBED_TIMEOUT_MS || file.timeoutMs || 4000),
  };
}

/** Parse either {data:[{embedding:[…]}]} (OpenAI) or {embedding:[…]} (llama.cpp),
 * or [{embedding:[…]}] (llama.cpp batch). Returns null if no vector is found. */
export function parseEmbeddingResponse(body: unknown): number[] | null {
  if (!body || typeof body !== "object") return null;
  const b = body as Record<string, unknown>;
  if (Array.isArray(b.data) && b.data[0] && typeof b.data[0] === "object") {
    const e = (b.data[0] as Record<string, unknown>).embedding;
    if (Array.isArray(e)) return e as number[];
  }
  if (Array.isArray(b.embedding)) return b.embedding as number[];
  if (Array.isArray(body) && body[0] && typeof body[0] === "object") {
    const e = (body[0] as Record<string, unknown>).embedding;
    if (Array.isArray(e)) return e as number[];
  }
  return null;
}

/** Returns the embedding vector, or null on any failure (server down, timeout,
 * bad response). Never throws. `fetchImpl` is injectable for testing. */
export async function embed(
  text: string,
  cfg: EmbedConfig = defaultConfig(),
  fetchImpl: typeof fetch = fetch,
): Promise<number[] | null> {
  if (!text.trim()) return null;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
  try {
    const res = await fetchImpl(cfg.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: text, content: text, model: cfg.model }),
      signal: ctrl.signal,
    });
    if (!res.ok) return null;
    const json = await res.json();
    const vec = parseEmbeddingResponse(json);
    return vec && vec.length ? vec : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

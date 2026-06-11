import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { resolve, extname, relative, join } from "node:path";
import { readFile, readdir, stat } from "node:fs/promises";
import { extractSymbols, langForExt, type Symbol } from "./extract.ts";

// symbols — give Gemma a code outline instead of making it read whole files to
// find one signature. The biggest recurring prefill saving: 30 lines instead of
// 800, every time it needs to locate something. PLAN.md item 2.

const MAX_OUTPUT_LINES = 60;       // R3 output cap
const REDIRECT_THRESHOLD = 200;    // R4: reads of code files larger than this get the outline
const INDEX_MAX_FILES = 4000;      // safety bound for find_symbol crawl
const SKIP_DIRS = new Set(["node_modules", ".git", "dist", "build", "target", "__pycache__", ".venv", "venv", "vendor", ".next", "out"]);

function formatSymbols(rel: string, syms: Symbol[]): string {
  if (!syms.length) return `${rel}: no symbols found (not a recognized language, or empty).`;
  const shown = syms.slice(0, MAX_OUTPUT_LINES);
  const body = shown.map((s) => `${String(s.line).padStart(4)}  ${s.kind.padEnd(7)} ${s.text}`).join("\n");
  const more = syms.length > MAX_OUTPUT_LINES
    ? `\n… ${syms.length - MAX_OUTPUT_LINES} more symbols (read the file directly for the full list).`
    : "";
  return `${rel} — ${syms.length} symbols:\n${body}${more}`;
}

async function symbolsForFile(absPath: string, cwd: string): Promise<{ text: string; count: number }> {
  const lang = langForExt(extname(absPath));
  const rel = relative(cwd, absPath) || absPath;
  if (!lang) return { text: `${rel}: unsupported file type for outline. Read it directly.`, count: 0 };
  let source: string;
  try {
    source = await readFile(absPath, "utf8");
  } catch (e) {
    return { text: `Could not read ${rel}: ${e instanceof Error ? e.message : String(e)}`, count: 0 };
  }
  const syms = extractSymbols(lang, source);
  return { text: formatSymbols(rel, syms), count: syms.length };
}

async function* walk(dir: string, depth = 0): AsyncGenerator<string> {
  if (depth > 12) return;
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    if (e.name.startsWith(".") && e.name !== ".") continue;
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name)) continue;
      yield* walk(join(dir, e.name), depth + 1);
    } else if (langForExt(extname(e.name))) {
      yield join(dir, e.name);
    }
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "get_symbols",
    label: "Get Symbols",
    description: "Outline a code file: function/class/type signatures and imports with line numbers. Use before reading a whole file.",
    parameters: Type.Object({
      path: Type.String({ description: "File path (relative to cwd or absolute)" }),
    }),
    async execute(_id, params, _signal) {
      // ctx.cwd isn't passed to execute; resolve against process.cwd() which pi sets per session.
      const abs = resolve(params.path);
      const { text } = await symbolsForFile(abs, process.cwd());
      return { content: [{ type: "text", text }] };
    },
  });

  pi.registerTool({
    name: "find_symbol",
    label: "Find Symbol",
    description: "Find where a function/class/type is defined across the project. Returns file:line for each definition.",
    parameters: Type.Object({
      name: Type.String({ description: "Exact symbol name to locate" }),
    }),
    async execute(_id, params, signal) {
      const target = params.name.trim();
      if (!target) {
        return { content: [{ type: "text", text: "find_symbol needs a non-empty name, e.g. find_symbol(name=\"parseConfig\")." }], isError: true };
      }
      const cwd = process.cwd();
      const hits: string[] = [];
      let scanned = 0;
      for await (const file of walk(cwd)) {
        if (signal?.aborted) break;
        if (++scanned > INDEX_MAX_FILES) break;
        const lang = langForExt(extname(file));
        if (!lang) continue;
        let src: string;
        try { src = await readFile(file, "utf8"); } catch { continue; }
        // Cheap pre-filter before regex work.
        if (!src.includes(target)) continue;
        for (const s of extractSymbols(lang, src)) {
          if (s.name === target || s.name.split(/[\s.<(]/)[0] === target) {
            hits.push(`${relative(cwd, file)}:${s.line}  ${s.kind}  ${s.text}`);
            if (hits.length >= MAX_OUTPUT_LINES) break;
          }
        }
        if (hits.length >= MAX_OUTPUT_LINES) break;
      }
      const text = hits.length
        ? `Definitions of "${target}":\n${hits.join("\n")}${hits.length >= MAX_OUTPUT_LINES ? "\n… (more; narrow the name)" : ""}`
        : `No definition of "${target}" found. Check spelling, or it may be imported from a dependency / defined dynamically.`;
      return { content: [{ type: "text", text }] };
    },
  });

  // R4 enforce>persuade: when Gemma reads a large code file in full, hand back the
  // outline instead and tell it how to get the real content. Small models will
  // reach for `read` reflexively; this redirects without needing a prompt rule.
  pi.on("tool_call", async (event, ctx) => {
    if (!isToolCallEventType("read", event)) return;
    const input = event.input as { path?: string; offset?: number; limit?: number };
    if (!input.path || input.offset || input.limit) return; // partial reads are intentional — leave them
    const abs = resolve(ctx.cwd, input.path);
    if (!langForExt(extname(abs))) return;
    let size = 0;
    try { size = (await stat(abs)).size; } catch { return; }
    // ~ bytes/line heuristic avoids a full read just to count lines.
    if (size < REDIRECT_THRESHOLD * 40) return;
    const { text, count } = await symbolsForFile(abs, ctx.cwd);
    if (count === 0) return; // nothing useful to show; let the real read proceed
    ctx.ui.notify(`symbols: outline shown for ${input.path} instead of full read`, "info");
    return {
      block: true,
      reason:
        `${text}\n\n(This is the outline, shown to save context instead of the whole file. ` +
        `If you need the actual lines, read a specific range with offset/limit — ` +
        `e.g. read(path="${input.path}", offset=1, limit=200) — which is not redirected.)`,
    };
  });
}

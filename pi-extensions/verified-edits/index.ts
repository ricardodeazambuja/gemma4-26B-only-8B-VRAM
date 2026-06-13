import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isEditToolResult, isWriteToolResult } from "@earendil-works/pi-coding-agent";
import { resolve, extname } from "node:path";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";

// verified-edits — after every edit/write to a source file, run the cheapest
// available checker for that language and append any error to the tool result,
// so Gemma sees ground truth in the same turn instead of trusting its own
// (weak) self-verification — enforce > persuade (design rule R4).

interface Checker {
  /** Build the argv for checking `absPath`, or null if this checker can't run. */
  cmd: (absPath: string) => string[] | null;
  /** Human label shown in the failure line, e.g. "py_compile". */
  label: string;
}

// First checker whose binary exists wins, per extension. Ordered cheapest-first.
const CHECKERS: Record<string, Checker[]> = {
  ".py": [
    { label: "ruff", cmd: (p) => has("ruff") ? ["ruff", "check", "--quiet", p] : null },
    { label: "py_compile", cmd: (p) => has("python3") ? ["python3", "-m", "py_compile", p] : null },
  ],
  ".ts": [{ label: "tsc", cmd: (p) => tscCmd(p) }],
  ".tsx": [{ label: "tsc", cmd: (p) => tscCmd(p) }],
  ".js": [{ label: "node-check", cmd: (p) => has("node") ? ["node", "--check", p] : null }],
  ".mjs": [{ label: "node-check", cmd: (p) => has("node") ? ["node", "--check", p] : null }],
  ".cjs": [{ label: "node-check", cmd: (p) => has("node") ? ["node", "--check", p] : null }],
  ".rs": [{ label: "rustc", cmd: (p) => has("rustc") ? ["rustc", "--edition", "2021", "--emit", "metadata", "-o", "/dev/null", p] : null }],
  ".go": [{ label: "gofmt", cmd: (p) => has("gofmt") ? ["gofmt", "-e", p] : null }],
  ".json": [{ label: "json", cmd: (p) => has("node") ? ["node", "-e", `JSON.parse(require('fs').readFileSync(${JSON.stringify(p)},'utf8'))`] : null }],
  ".sh": [{ label: "bash-n", cmd: (p) => has("bash") ? ["bash", "-n", p] : null }],
};

const PATH_DIRS = (process.env.PATH || "").split(":");
const binCache = new Map<string, boolean>();
function has(bin: string): boolean {
  if (binCache.has(bin)) return binCache.get(bin)!;
  const found = PATH_DIRS.some((d) => d && existsSync(resolve(d, bin)));
  binCache.set(bin, found);
  return found;
}

function tscCmd(p: string): string[] | null {
  // tsc on a lone file ignores project config; --noEmit just syntax/type-checks it.
  if (has("tsc")) return ["tsc", "--noEmit", "--allowJs", "--skipLibCheck", p];
  if (has("npx")) return ["npx", "--no-install", "tsc", "--noEmit", "--allowJs", "--skipLibCheck", p];
  return null;
}

const CHECK_TIMEOUT_MS = 5000;
const MAX_ERR_LINES = 8;

function runCheck(argv: string[], signal?: AbortSignal): Promise<{ ok: boolean; out: string }> {
  return new Promise((res) => {
    let stdout = "";
    let stderr = "";
    let done = false;
    const finish = (ok: boolean) => { if (!done) { done = true; res({ ok, out: (stderr || stdout).trim() }); } };
    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(argv[0], argv.slice(1), { stdio: ["ignore", "pipe", "pipe"] });
    } catch {
      return finish(true); // can't spawn → stay silent, never block the edit
    }
    const timer = setTimeout(() => { try { child.kill("SIGKILL"); } catch {} finish(true); }, CHECK_TIMEOUT_MS);
    const onAbort = () => { try { child.kill("SIGKILL"); } catch {} };
    signal?.addEventListener("abort", onAbort, { once: true });
    child.stdout?.on("data", (d) => (stdout += d.toString()));
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("error", () => { clearTimeout(timer); finish(true); });
    child.on("close", (code) => { clearTimeout(timer); signal?.removeEventListener("abort", onAbort); finish(code === 0); });
  });
}

/** Pick the first runnable checker for a path. Exported shape mirrors test.mjs. */
export function pickChecker(absPath: string): { argv: string[]; label: string } | null {
  const ext = extname(absPath).toLowerCase();
  for (const c of CHECKERS[ext] || []) {
    const argv = c.cmd(absPath);
    if (argv) return { argv, label: c.label };
  }
  return null;
}

function trimError(label: string, out: string): string {
  const lines = out.split("\n").filter((l) => l.trim());
  const head = lines.slice(0, MAX_ERR_LINES).join("\n");
  const more = lines.length > MAX_ERR_LINES ? `\n… (${lines.length - MAX_ERR_LINES} more lines)` : "";
  return `\n\n⚠ CHECK FAILED (${label}) — fix this before continuing:\n${head}${more}`;
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_result", async (event, ctx) => {
    if (event.isError) return; // the edit itself failed; nothing to verify
    if (!isEditToolResult(event) && !isWriteToolResult(event)) return;

    const rel = (event.input as { path?: string }).path;
    if (!rel) return;
    const absPath = resolve(ctx.cwd, rel);

    const picked = pickChecker(absPath);
    if (!picked) return; // unknown language or no checker installed → silent

    const { ok, out } = await runCheck(picked.argv, ctx.signal);
    if (ok || !out) return;

    const note = { type: "text" as const, text: trimError(picked.label, out) };
    return { content: [...event.content, note] };
  });
}

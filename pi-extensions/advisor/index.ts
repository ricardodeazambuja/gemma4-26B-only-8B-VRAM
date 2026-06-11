import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

// advisor — consult a stronger external agent mid-session, like Claude Code's
// advisor tool. Serializes the current session branch to a transcript file,
// drives a user-configured TUI agent via tui-driver (tmux), and returns its
// reply as the tool result. The advised agent is whatever the config says
// (agy, claude, aichat, ...) — Gemma never picks it.

// ---------------------------------------------------------------------------
// Config: env > ~/.pi/agent/advisor-config.json > defaults (same precedence
// pattern as semantic-memory's embed-config). `command` has NO default on
// purpose — consulting an external agent may cost money, so it must be an
// explicit user decision.
// ---------------------------------------------------------------------------

export interface AdvisorConfig {
  command: string;          // TUI command tui-driver drives ("" = unconfigured)
  tuiDriver: string;        // tui-driver executable (path or on $PATH)
  timeoutSec: number;       // max wait for the advisor's reply (TUI_TIMEOUT)
  keepSession: boolean;     // leave the TUI running for follow-up calls
  inlineTranscript: boolean;// paste transcript text into the prompt instead of a file path
  promptTemplate: string;   // {transcript} = path (or text when inline), {focus} = optional focus
  maxToolResultChars: number; // per tool-result truncation in the transcript
  maxInlineChars: number;   // transcript cap when inlineTranscript is true
  maxReplyChars: number;    // reply cap before the continuation hint kicks in
}

export const DEFAULT_PROMPT_TEMPLATE =
  "You are a senior engineer advising another AI coding agent mid-session. " +
  "Read the full session transcript in the file {transcript} first. " +
  "Then advise: is the current approach sound, what was missed or assumed wrongly, " +
  "and what concrete next step should be taken? Be specific, under 30 lines. {focus}";

export function configPath(): string {
  return process.env.PI_ADVISOR_CONFIG || join(homedir(), ".pi", "agent", "advisor-config.json");
}

export function loadFileConfig(path = configPath()): Partial<AdvisorConfig> {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return {};
  }
}

export function defaultConfig(): AdvisorConfig {
  const file = loadFileConfig();
  const envBool = (v: string | undefined, fallback: boolean) =>
    v === undefined ? fallback : v !== "0" && v.toLowerCase() !== "false";
  return {
    command: process.env.PI_ADVISOR_CMD || file.command || "",
    tuiDriver:
      process.env.PI_ADVISOR_TUI_DRIVER ||
      file.tuiDriver ||
      join(homedir(), ".local", "bin", "tui-driver"),
    timeoutSec: Number(process.env.PI_ADVISOR_TIMEOUT_SEC || file.timeoutSec || 600),
    keepSession: envBool(process.env.PI_ADVISOR_KEEP_SESSION, file.keepSession ?? true),
    inlineTranscript: envBool(process.env.PI_ADVISOR_INLINE, file.inlineTranscript ?? false),
    promptTemplate: file.promptTemplate || DEFAULT_PROMPT_TEMPLATE,
    maxToolResultChars: Number(file.maxToolResultChars || 1500),
    maxInlineChars: Number(file.maxInlineChars || 60000),
    maxReplyChars: Number(file.maxReplyChars || 12000),
  };
}

export const EXAMPLE_CONFIG = `{
  "command": "agy",
  "tuiDriver": "tui-driver",
  "timeoutSec": 600,
  "keepSession": true,
  "inlineTranscript": false
}`;

// ---------------------------------------------------------------------------
// Transcript serialization: session entries → readable markdown-ish text.
// Tool results are the bulk of a session; truncate each one hard so the
// advisor sees the whole arc instead of one giant bash dump.
// ---------------------------------------------------------------------------

function clip(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + ` …[${s.length - max} chars truncated]`;
}

function blocksToText(content: unknown, toolResultMax: number): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content as any[]) {
    if (!b || typeof b !== "object") continue;
    if (b.type === "text") parts.push(b.text ?? "");
    else if (b.type === "thinking") parts.push(`<thinking> ${clip(b.thinking ?? "", 600)}`);
    else if (b.type === "toolCall")
      parts.push(`→ tool ${b.name}(${clip(JSON.stringify(b.arguments ?? {}), 400)})`);
    else if (b.type === "image") parts.push("[image]");
  }
  return parts.join("\n");
}

export function formatTranscript(entries: any[], maxToolResultChars = 1500): string {
  const out: string[] = [`=== SESSION TRANSCRIPT (${entries.length} entries) ===`];
  for (const e of entries) {
    if (e?.type === "custom_message") {
      out.push(`\n[extension:${e.customType}]`, blocksToText(e.content, maxToolResultChars));
      continue;
    }
    const m = e?.message;
    if (!m) continue;
    switch (m.role) {
      case "user":
        out.push("\n[user]", blocksToText(m.content, maxToolResultChars));
        break;
      case "assistant":
        out.push("\n[assistant]", blocksToText(m.content, maxToolResultChars));
        break;
      case "toolResult":
        out.push(
          `\n[tool result: ${m.toolName}${m.isError ? " (ERROR)" : ""}]`,
          clip(blocksToText(m.content, maxToolResultChars), maxToolResultChars),
        );
        break;
      case "bashExecution":
        out.push(
          `\n[user ran: ${m.command}]`,
          clip(m.output ?? "", maxToolResultChars),
        );
        break;
      case "custom":
        out.push(`\n[extension:${m.customType ?? "?"}]`, blocksToText(m.content, maxToolResultChars));
        break;
      case "branchSummary":
      case "compactionSummary":
        out.push(`\n[${m.role}]`, clip(m.summary ?? blocksToText(m.content, maxToolResultChars), 4000));
        break;
    }
  }
  out.push("\n=== END OF TRANSCRIPT ===");
  return out.join("\n");
}

export function buildPrompt(template: string, transcript: string, focus: string): string {
  return template
    .replaceAll("{transcript}", transcript)
    .replaceAll("{focus}", focus ? `Focus on: ${focus}` : "")
    .trim();
}

export function capReply(reply: string, max: number, fullPath: string): string {
  if (reply.length <= max) return reply;
  return reply.slice(0, max) + `\n…[reply truncated — full text saved at ${fullPath}]`;
}

// Transcripts can contain anything the session saw (including secrets that
// leaked into tool output), so they never go into world-readable /tmp:
// one private 0o700 dir per pi process, files written 0o600.
let advisorDirCache: string | null = null;
export function advisorDir(): string {
  if (!advisorDirCache || !existsSync(advisorDirCache)) {
    advisorDirCache = mkdtempSync(join(tmpdir(), "pi-advisor-"));
  }
  return advisorDirCache;
}

// ---------------------------------------------------------------------------
// tui-driver invocations. Injectable so tests never touch tmux.
// ---------------------------------------------------------------------------

export type Runner = (
  args: string[],
  env: Record<string, string>,
  timeoutMs: number,
  signal?: AbortSignal,
) => Promise<{ stdout: string; stderr: string; code: number }>;

function makeRealRunner(tuiDriver: string): Runner {
  return (args, env, timeoutMs, signal) =>
    new Promise((resolve) => {
      execFile(
        tuiDriver,
        args,
        {
          env: { ...process.env, ...env },
          timeout: timeoutMs,
          maxBuffer: 16 * 1024 * 1024,
          signal,
        },
        (err: any, stdout, stderr) => {
          resolve({
            stdout: stdout?.toString() ?? "",
            stderr: stderr?.toString() ?? "",
            code: err ? (typeof err.code === "number" ? err.code : 1) : 0,
          });
        },
      );
    });
}

// ---------------------------------------------------------------------------
// The tool itself.
// ---------------------------------------------------------------------------

export async function consult(
  cfg: AdvisorConfig,
  run: Runner,
  entries: any[],
  focus: string,
  tag: string,
  signal?: AbortSignal,
): Promise<{ text: string; isError: boolean }> {
  if (!cfg.command) {
    return {
      isError: true,
      text:
        `advisor is not configured: no external agent command set.\n` +
        `Create ${configPath()} with e.g.:\n${EXAMPLE_CONFIG}\n` +
        `(or set PI_ADVISOR_CMD). "command" is the TUI tui-driver should drive.`,
    };
  }

  // tui-driver is a hard prerequisite. When configured as a path (the default
  // is ~/.local/bin/tui-driver), verify it exists before driving tmux; a bare
  // command name is left to PATH resolution at exec time.
  if (cfg.tuiDriver.includes("/") && !existsSync(cfg.tuiDriver)) {
    return {
      isError: true,
      text:
        `advisor: tui-driver not found at ${cfg.tuiDriver}.\n` +
        `Install it: from your tui-driver checkout run './tui-driver.sh install' ` +
        `(copies it to ~/.local/bin/tui-driver; needs tmux), ` +
        `or set "tuiDriver" in ${configPath()} to where the script lives.`,
    };
  }

  const transcript = formatTranscript(entries, cfg.maxToolResultChars);
  const transcriptPath = join(advisorDir(), `${tag}.md`);
  writeFileSync(transcriptPath, transcript, { encoding: "utf8", mode: 0o600 });

  const transcriptArg = cfg.inlineTranscript
    ? clip(transcript, cfg.maxInlineChars)
    : transcriptPath;
  const prompt = buildPrompt(cfg.promptTemplate, transcriptArg, focus);

  const env = {
    TUI_QUIET: "true",
    TUI_TIMEOUT: String(cfg.timeoutSec),
    TUI_DIR: process.cwd(),
  };
  const slackMs = 90_000; // child timeout > TUI_TIMEOUT so tui-driver times out first

  // Idempotent session bring-up: start only when status says stopped.
  const status = await run([cfg.command, "status"], env, 30_000, signal);
  if (status.stdout.trim() !== "running") {
    const started = await run([cfg.command, "start"], env, 120_000, signal);
    if (started.code !== 0) {
      return {
        isError: true,
        text:
          `advisor: failed to start '${cfg.command}' via ${cfg.tuiDriver}.\n` +
          clip(started.stderr || started.stdout, 1000) +
          `\nCheck: is tmux installed, is '${cfg.command}' on PATH, does it start as a TUI?`,
      };
    }
  }

  const sent = await run(
    [cfg.command, "send", prompt],
    env,
    cfg.timeoutSec * 1000 + slackMs,
    signal,
  );

  if (!cfg.keepSession) {
    await run([cfg.command, "stop"], env, 30_000).catch(() => {});
  }

  const reply = sent.stdout.trim();
  if (!reply) {
    return {
      isError: true,
      text:
        `advisor: '${cfg.command}' returned no reply` +
        (sent.code !== 0 ? ` (tui-driver exit ${sent.code})` : "") +
        `.\n${clip(sent.stderr, 800)}\n` +
        `The transcript is still at ${transcriptPath}; try again or raise timeoutSec.`,
    };
  }

  const replyPath = join(advisorDir(), `${tag}-reply.md`);
  writeFileSync(replyPath, reply, { encoding: "utf8", mode: 0o600 });

  const note = sent.code !== 0 ? "\n\n(note: advisor timed out — reply may be incomplete)" : "";
  return {
    isError: false,
    text: `ADVISOR (${cfg.command}) says:\n\n${capReply(reply, cfg.maxReplyChars, replyPath)}${note}`,
  };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "advisor",
    label: "Advisor",
    description:
      "Consult an external senior advisor agent that reads a transcript of this whole session and reviews your approach.",
    promptGuidelines: [
      "Use advisor before committing to a big decision, when stuck after repeated failures, or before declaring a long task done.",
    ],
    parameters: Type.Object({
      focus: Type.Optional(
        Type.String({ description: "Optional: what the advisor should focus on" }),
      ),
    }),
    async execute(toolCallId, params, signal, _onUpdate, ctx) {
      const cfg = defaultConfig();
      const run = makeRealRunner(cfg.tuiDriver);
      const entries = ctx?.sessionManager?.getBranch?.() ?? [];
      const { text, isError } = await consult(
        cfg,
        run,
        entries,
        (params as any).focus ?? "",
        toolCallId.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 24) || "call",
        signal,
      );
      return { content: [{ type: "text", text }], isError };
    },
  });
}

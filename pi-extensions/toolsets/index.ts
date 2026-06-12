import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";

// toolsets — context economy: don't announce situational tools a session doesn't need.
// Tool definitions are the standing prefill tax (R5); this shrinks the *set*, not just the
// wording. PLAN.md item 12.
//
// R1 IS THE WHOLE CONSTRAINT. Tool schemas live in the KV-cached prefix, so changing the
// active set RE-PREFILLS from the tools onward. Naive per-turn toggling trades schema tokens
// for re-prefill tokens — usually a loss. So the set is chosen ONCE per session (session_start)
// and only changes on an explicit `/tools` action (a bounded, understood cost). Per-turn
// auto-gating is deliberately NOT done here (future work).
//
// Safety: it only ever REMOVES known group tools by name from the current active set, so pi's
// built-ins and any unrecognised tools are never dropped. Default disables nothing (opt-in).

// Groups of OUR situational tools. Everything else (built-ins, symbols, plan, goal) is always
// active and never touched. Hiding the `recall` TOOL doesn't break semantic-memory's
// auto-recall — that's a context injection, not a tool call.
export const DEFAULT_GROUPS: Record<string, string[]> = {
  web: ["web_search", "fetch_page"],
  memory: ["remember", "recall", "forget"],
  advisor: ["advisor"],
};

export interface ToolsetsConfig {
  groups: Record<string, string[]>;
  disabled: string[];
}

export function loadConfig(env: Record<string, string | undefined> = process.env): ToolsetsConfig {
  let groups: Record<string, string[]> = { ...DEFAULT_GROUPS };
  let disabled: string[] = [];
  const file = env.PI_TOOLSETS_CONFIG || join(homedir(), ".pi", "agent", "toolsets-config.json");
  try {
    if (existsSync(file)) {
      const c = JSON.parse(readFileSync(file, "utf8"));
      if (c && typeof c.groups === "object" && c.groups) groups = { ...groups, ...c.groups };
      if (Array.isArray(c?.disabled)) disabled = c.disabled.map(String);
    }
  } catch {}
  if (env.PI_TOOLSETS_DISABLED) disabled = env.PI_TOOLSETS_DISABLED.split(",").map((s) => s.trim()).filter(Boolean);
  return { groups, disabled };
}

// Flatten the named groups to a deduped tool list (unknown group names contribute nothing).
export function toolsForGroups(names: string[], groups: Record<string, string[]>): string[] {
  const out = new Set<string>();
  for (const n of names) for (const t of groups[n] || []) out.add(t);
  return [...out];
}

// The active set with the disabled groups' tools removed. Never touches non-group tools.
export function applyDisabled(active: string[], disabled: string[], groups: Record<string, string[]>): string[] {
  const hide = new Set(toolsForGroups(disabled, groups));
  return active.filter((t) => !hide.has(t));
}

export default function (pi: ExtensionAPI) {
  const cfg = loadConfig();

  const canManage = () => typeof pi.getActiveTools === "function" && typeof pi.setActiveTools === "function";

  // Apply the configured gating ONCE per session (R1: set the prefix and leave it stable).
  pi.on("session_start", async () => {
    try {
      if (!canManage() || !cfg.disabled.length) return; // opt-in: nothing disabled → no-op
      pi.setActiveTools(applyDisabled(pi.getActiveTools(), cfg.disabled, cfg.groups));
    } catch {}
  });

  // Human control: list groups, or reveal/hide one live (each change re-prefills once).
  pi.registerCommand("tools", {
    description: "List tool groups and which are active; /tools on|off <group> to reveal or hide one.",
    handler: async (args, ctx) => {
      const notify = (m: string) => { try { (ctx as any)?.ui?.notify?.(m); } catch {} };
      if (!canManage()) { notify("This pi build doesn't expose tool management."); return; }
      const parts = (args ?? "").trim().split(/\s+/).filter(Boolean);
      const verb = (parts[0] || "").toLowerCase();
      const group = parts[1];

      if (verb === "on" || verb === "off") {
        if (!group || !cfg.groups[group]) {
          notify(`Unknown group "${group ?? ""}". Groups: ${Object.keys(cfg.groups).join(", ")}.`);
          return;
        }
        const gTools = cfg.groups[group];
        const active = pi.getActiveTools();
        const next = verb === "on"
          ? [...new Set([...active, ...gTools])]
          : active.filter((t) => !gTools.includes(t));
        pi.setActiveTools(next);
        notify(`${verb === "on" ? "Revealed" : "Hid"} "${group}" (${gTools.join(", ")}). This changes the tool prefix → one re-prefill.`);
        return;
      }

      const active = new Set(pi.getActiveTools());
      const lines = Object.entries(cfg.groups).map(([g, ts]) => `  [${ts.some((t) => active.has(t)) ? "on " : "off"}] ${g}: ${ts.join(", ")}`);
      notify(`Tool groups (everything else is always on):\n${lines.join("\n")}\nChange with /tools on|off <group> (re-prefills once).`);
    },
  });
}

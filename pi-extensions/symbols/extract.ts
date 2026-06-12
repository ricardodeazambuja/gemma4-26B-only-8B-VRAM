// Language-agnostic-ish symbol extraction by line regex. Deliberately
// dependency-free: ctags isn't always installed and tree-sitter needs native
// or WASM builds. Precision is "good enough to navigate" — signatures, classes,
// imports — which is all an outline needs. Swap this module for tree-sitter
// later without touching index.ts (it only imports `extractSymbols`).

export interface Symbol {
  line: number;       // 1-based
  kind: string;       // func | class | method | import | const | type | …
  text: string;       // the trimmed signature line
  name: string;       // best-effort identifier (for find_symbol)
}

type Rule = { kind: string; re: RegExp; name: (m: RegExpMatchArray) => string };

const RULES: Record<string, Rule[]> = {
  python: [
    { kind: "import", re: /^\s*(?:import|from)\s+\S/, name: (m) => m[0].trim() },
    { kind: "class", re: /^\s*class\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "func", re: /^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
  ],
  js: [
    { kind: "import", re: /^\s*(?:import\s.+from\s|import\s+['"]|const\s+\w+\s*=\s*require\()/, name: (m) => m[0].trim() },
    { kind: "export", re: /^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|interface|type|enum)\s+([A-Za-z_$][\w$]*)/, name: (m) => m[1] },
    { kind: "class", re: /^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)/, name: (m) => m[1] },
    { kind: "func", re: /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)/, name: (m) => m[1] },
    { kind: "arrow", re: /^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/, name: (m) => m[1] },
    { kind: "type", re: /^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)/, name: (m) => m[1] },
    { kind: "method", re: /^\s{2,}(?:public|private|protected|static|async|get|set|\s)*\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*[:{]/, name: (m) => m[1] },
  ],
  rust: [
    { kind: "use", re: /^\s*use\s+\S/, name: (m) => m[0].trim() },
    { kind: "fn", re: /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "struct", re: /^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "enum", re: /^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "trait", re: /^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "impl", re: /^\s*impl(?:<[^>]*>)?\s+(.+?)\s*\{/, name: (m) => m[1].trim() },
  ],
  go: [
    { kind: "import", re: /^\s*import\s+[("]/, name: (m) => m[0].trim() },
    { kind: "func", re: /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "type", re: /^\s*type\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
  ],
  c: [
    { kind: "include", re: /^\s*#\s*include\s/, name: (m) => m[0].trim() },
    { kind: "define", re: /^\s*#\s*define\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    { kind: "struct", re: /^\s*(?:typedef\s+)?struct\s+([A-Za-z_]\w*)/, name: (m) => m[1] },
    // function definition: type name(args) { ... at column 0
    { kind: "func", re: /^[A-Za-z_][\w\s*]*\s+\*?([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$/, name: (m) => m[1] },
  ],
};

const EXT_LANG: Record<string, string> = {
  ".py": "python",
  ".js": "js", ".mjs": "js", ".cjs": "js", ".jsx": "js", ".ts": "js", ".tsx": "js",
  ".rs": "rust",
  ".go": "go",
  ".c": "c", ".h": "c", ".cpp": "c", ".cc": "c", ".hpp": "c", ".cxx": "c",
};

export function langForExt(ext: string): string | null {
  return EXT_LANG[ext.toLowerCase()] || null;
}

export function extractSymbols(lang: string, source: string): Symbol[] {
  const rules = RULES[lang];
  if (!rules) return [];
  const out: Symbol[] = [];
  const lines = source.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    for (const rule of rules) {
      const m = line.match(rule.re);
      if (m) {
        out.push({ line: i + 1, kind: rule.kind, text: line.trim().slice(0, 200), name: rule.name(m) });
        break; // one symbol kind per line
      }
    }
  }
  return out;
}

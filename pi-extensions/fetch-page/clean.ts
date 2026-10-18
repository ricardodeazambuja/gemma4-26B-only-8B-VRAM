// Pure text post-processing for fetch-page: collapse browser innerText into tidy
// lines and paginate with an offset, so a long page can't blow the context window
// (rule R3). The DOM extraction itself runs in-browser (see index.ts).

export function collapseText(raw: string): string {
  return raw
    .replace(/\r/g, "")
    .split("\n")
    .map((l) => l.replace(/[ \t]+/g, " ").trim())
    .filter((l, i, arr) => l.length > 0 || (i > 0 && arr[i - 1].length > 0)) // collapse blank runs to one
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export interface Page { body: string; shownLines: number; totalLines: number; nextOffset: number | null; }

// Lines longer than MAX_LINE_CHARS are cut into pieces first: a JSON API response or a
// minified page arrives as one huge line, and a cap counted in lines alone would let
// it through whole (a 233 KB PyPI JSON once filled a 16K context in one call). Each
// page also stops at maxChars. Offsets count the cut lines, so paging stays consistent.
export const MAX_LINE_CHARS = 200;

export function wrapLines(text: string, width = MAX_LINE_CHARS): string[] {
  const out: string[] = [];
  for (const line of text.split("\n")) {
    if (line.length <= width) { out.push(line); continue; }
    for (let i = 0; i < line.length; i += width) out.push(line.slice(i, i + width));
  }
  return out;
}

export function paginate(text: string, offset: number, maxLines: number, maxChars = 6000): Page {
  const lines = wrapLines(text);
  const start = Math.max(0, offset | 0);
  const slice: string[] = [];
  let chars = 0;
  for (let i = start; i < lines.length && slice.length < maxLines; i++) {
    if (slice.length > 0 && chars + lines[i].length + 1 > maxChars) break;
    slice.push(lines[i]);
    chars += lines[i].length + 1;
  }
  const end = start + slice.length;
  return {
    body: slice.join("\n"),
    shownLines: slice.length,
    totalLines: lines.length,
    nextOffset: end < lines.length ? end : null,
  };
}

export function formatPage(url: string, title: string, p: Page): string {
  const header = title ? `# ${title}\n${url}\n` : `${url}\n`;
  const more = p.nextOffset !== null
    ? `\n\n… ${p.totalLines - p.nextOffset} more lines — call again with offset=${p.nextOffset}.`
    : "";
  return `${header}\n${p.body}${more}`;
}

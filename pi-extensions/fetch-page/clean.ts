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

export function paginate(text: string, offset: number, maxLines: number): Page {
  const lines = text.split("\n");
  const start = Math.max(0, offset | 0);
  const slice = lines.slice(start, start + maxLines);
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

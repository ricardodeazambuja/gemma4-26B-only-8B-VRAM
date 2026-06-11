#!/usr/bin/env bash
# stats.sh — summarize /gemma-draft usage from stats.jsonl.
# Run directly or via the /spec-stats slash command.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if [ ! -s "$STATS_FILE" ]; then
  echo "gemma-draft: no events yet ($STATS_FILE empty)."
  echo "Run /gemma-draft <task> a few times to populate it."
  exit 0
fi

jq -rs '
  def c(f): map(select(f)) | length;
  (map(select(.event=="draft")))                                    as $d
  | ($d | c(.mode=="text" and .result=="ok"))                       as $txt
  | ($d | c(.mode=="text" and .result=="fail"))                     as $fail
  | ($d | c(.result=="offline"))                                    as $off
  | ($d | map(select(.mode=="image")))                              as $imgcalls
  | ($imgcalls | length)                                            as $icall
  | ($imgcalls | map(.ok // 0) | add // 0)                          as $iok
  | ($d | length)                                                   as $tot
  | ([ $d[] | select(.result=="ok" or .mode=="image") | .ms // empty ]) as $lat
  | (if ($lat|length)>0 then ($lat|add/(($lat|length))/1000*10|floor/10) else 0 end) as $avgs
  | (map(select(.event=="autostart")) | length)                     as $auto
  | (map(select(.event != "draft" and .event != "autostart")) | length) as $legacy
  | ($iok * 1000)                                                   as $imgsave
  | "gemma-draft usage  (\($tot) invocations)",
  "────────────────────────────────────────────",
  "  text drafts            : \($txt)",
  "  image reads (local OCR): \($iok) files across \($icall) calls",
  "  draft failed / empty   : \($fail)",
  "  server was down        : \($off)",
  "  server auto-launches   : \($auto)",
  "  avg draft latency      : \($avgs)s",
  "────────────────────────────────────────────",
  "  est. big-model image tokens kept off the bill: ~\($imgsave)   (rough heuristic)"
  + (if $legacy > 0 then
       "\n  (plus \($legacy) legacy hook-era events from before the /gemma-draft pivot)"
     else "" end)
' "$STATS_FILE"

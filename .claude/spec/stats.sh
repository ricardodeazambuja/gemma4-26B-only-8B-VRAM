#!/usr/bin/env bash
# stats.sh — summarize the speculative agent's predictor accuracy from stats.jsonl.
# Run directly or via the /spec-stats slash command.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if [ ! -s "$STATS_FILE" ]; then
  echo "speculative-agent: no events yet ($STATS_FILE empty)."
  echo "Start llama-server (scripts/start.sh) and run a few prompts to populate it."
  exit 0
fi

jq -rs '
  def c(f): map(select(f)) | length;
  (map(select(.event=="predict")))                                  as $p
  | ($p | c(.result=="hit"))                                        as $hit
  | ($p | c(.result=="hit_predicted"))                              as $bp
  | ($p | c(.result=="miss_drafted"))                               as $md
  | ($p | c(.result=="miss_nodraft"))                               as $mn
  | ($p | c(.result=="miss_offline"))                               as $off
  | ($p | length)                                                   as $tot
  | ($hit + $bp)                                                    as $hits
  | ($tot - $off)                                                   as $online
  | ([ $p[] | select(.result=="hit_predicted") | .score ])          as $scores
  | (if ($scores|length)>0 then ($scores|add/(($scores|length))|floor) else 0 end) as $avg
  | (map(select(.event=="speculate")) | length)                     as $spec
  | (map(select(.event=="image_offload")))                          as $imgs
  | ($imgs | length)                                                as $nimg
  | ($imgs | map(.bytes // 0) | add // 0)                           as $ibytes
  | def pct($n;$d): if $d>0 then (($n*1000/$d|floor)/10) else 0 end;
  "speculative-agent stats  (\($tot) prompts seen)",
  "────────────────────────────────────────────",
  "  cache hit (exact)        : \($hit)",
  "  branch predicted (fuzzy) : \($bp)   avg match \($avg)%",
  "  inline draft (miss)      : \($md)",
  "  no draft  (miss)         : \($mn)",
  "  offline   (server down)  : \($off)",
  "────────────────────────────────────────────",
  "  BRANCH-PREDICTION HIT RATE",
  "    overall : \($hits)/\($tot) = \(pct($hits;$tot))%",
  "    online  : \($hits)/\($online) = \(pct($hits;$online))%   (excludes offline turns)",
  "  background speculations  : \($spec)",
  "  image offloads           : \($nimg)   (~\(($ibytes/1024)|floor) KB kept off Opus)"
' "$STATS_FILE"

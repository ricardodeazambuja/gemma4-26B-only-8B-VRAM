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
  | ($p | c(.result=="miss_hard"))                                  as $mh
  | ($p | c(.result=="miss_skipped_long"))                          as $ms
  | ($p | c(.result=="miss_nodraft"))                               as $mn
  | ($p | c(.result=="miss_offline"))                               as $off
  | ($p | length)                                                   as $tot
  | ($hit + $bp)                                                    as $hits
  | ($tot - $off)                                                   as $online
  | ([ $p[] | select(.result=="hit_predicted") | .score ])          as $scores
  | (if ($scores|length)>0 then ($scores|add/(($scores|length))|floor) else 0 end) as $avg
  | ($p | c(.result=="miss_image_prewarm"))                         as $mi
  | ($p | c(.result=="miss_verb_gate"))                             as $vg
  | (map(select(.event=="outcome")))                                as $o
  | ($o | c(.verdict=="accepted"))                                  as $acc
  | ($o | c(.verdict=="superseded"))                                as $sup
  | (map(select(.event=="speculate")) | length)                     as $spec
  | (map(select(.event=="image_offload")))                          as $imgs
  | ($imgs | length)                                                as $nimg
  | ($imgs | map(select(.cached==true)) | length)                   as $nimgc
  | (map(select(.event=="image_prewarm")) | length)                 as $npre
  | ($imgs | map(.bytes // 0) | add // 0)                           as $ibytes
  | (map(select(.event=="log_offload")))                            as $logs
  | ($logs | length)                                                as $nlog
  | ($logs | map(((.bytes // 0)/4 - 1000) | if . < 0 then 0 else . end) | add // 0 | floor) as $logsave
  | ($nimg * 1000)                                                  as $imgsave
  | (($hit + $bp) * 200)                                            as $hitsave
  | ($logsave + $imgsave + $hitsave)                                as $totsave
  | def pct($n;$d): if $d>0 then (($n*1000/$d|floor)/10) else 0 end;
  "speculative-agent stats  (\($tot) prompts seen)",
  "────────────────────────────────────────────",
  "  cache hit (exact)        : \($hit)",
  "  branch predicted (fuzzy) : \($bp)   avg match \($avg)%",
  "  inline draft, easy (miss): \($md)",
  "  hard — left to big model : \($mh)",
  "  long — inline skipped    : \($ms)",
  "  image — prewarm instead  : \($mi)",
  "  verb-gated (overlap, ≠verb): \($vg)",
  "  no draft  (miss)         : \($mn)",
  "  offline   (server down)  : \($off)",
  "────────────────────────────────────────────",
  "  BRANCH-PREDICTION HIT RATE",
  "    overall : \($hits)/\($tot) = \(pct($hits;$tot))%",
  "    online  : \($hits)/\($online) = \(pct($hits;$online))%   (excludes offline turns)",
  "  DRAFT OUTCOMES (containment heuristic)",
  "    accepted : \($acc) · superseded : \($sup)" +
    (if ($acc+$sup)>0 then "   (\(pct($acc;$acc+$sup))% accepted)" else "   (none judged yet)" end),
  "  background speculations  : \($spec)",
  "  image offloads           : \($nimg) (\($nimgc) instant from prewarm) · prewarms: \($npre)",
  "  log digests              : \($nlog)",
  "────────────────────────────────────────────",
  "  est. big-model tokens kept off the bill: ~\($totsave)",
  "    logs ~\($logsave) · images ~\($imgsave) · prediction hits ~\($hitsave)   (rough heuristics)"
' "$STATS_FILE"

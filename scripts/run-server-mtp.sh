#!/usr/bin/env bash
# Interactive launcher for the Gemma 4 **MTP** server. Prompts for each tunable with a
# sensible [default] (just press Enter) and a one-line hint on safe values, then hands
# off to scripts/run-server.sh (so it inherits the tested backend / MoE / KV / banner
# logic). The server runs in the foreground; start pi separately (see the note at launch).
#
# What MTP does here — MEASURED, see docs/mtp-benchmark.md:
#   • Lossless self-speculative decoding; the 0.25 GB draft head fits at the same NCMOE.
#   • Real speedup ONLY at low temperature: ~+15–30 % at greedy/coding temp, and
#     ~nothing at the default temp 1.0 (acceptance-limited).
#   • 64k context is essentially free vs 32k (Gemma 4 sliding-window attention).
#   • --spec-draft-p-min is FORCED to 0: any p_min>0 degenerates the output (known bug).
#
# Non-interactive: set DRY_RUN=1 to print the resolved command without launching.
# Any value you pass as an env var is used as that prompt's default (e.g. CTX=32768 …).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD="$REPO_ROOT/models/gemma4-26b-a4b-qat/mtp-gemma-4-26B-A4B-it.gguf"

# ask "question" "default" [validator-regex] -> sets ANS (Enter = default; re-asks on bad input; Ctrl-C/D cancels)
ask() {
  local q="$1" def="$2" re="${3:-}" a
  while :; do
    read -r -p "$q
   [$def] > " a || { echo; echo "cancelled."; exit 130; }
    a="${a:-$def}"
    if [ -z "$re" ] || printf '%s' "$a" | grep -qE "$re"; then ANS="$a"; return; fi
    echo "   ! '$a' doesn't look valid — try again."
  done
}

echo "=========================================================="
echo "  Gemma 4 MTP server — interactive launch"
echo "  (Enter = accept [default]; Ctrl-C to cancel)"
echo "=========================================================="

ask "Enable MTP speculative decoding?  y = MTP (helps at LOW temp) · n = plain baseline" "y" '^[yYnN]$'
MTP="$ANS"
NMAX=""
if [ "$MTP" = y ] || [ "$MTP" = Y ]; then
  [ -f "$HEAD" ] || { echo "ERROR: MTP draft head missing: $HEAD  (run scripts/setup.sh)"; exit 1; }
  ask "  Draft tokens per step (n-max)?  safe = 2 · usable range 1–6 (higher rarely helps here)" "${NMAX:-2}" '^[1-9]$'
  NMAX="$ANS"
fi

ask "Temperature?  0 = greedy (fastest, deterministic) · 0.7 = coding · 1.0 = Gemma default (MTP gives ~no gain)" "${TEMP:-0.7}" '^[0-9]+([.][0-9]+)?$'
TEMP="$ANS"

ask "Context window (CTX)?  32768 · 65536 = free & recommended · 131072 (uses more RAM)" "${CTX:-65536}" '^[0-9]+$'
CTX="$ANS"

ask "KV-cache quant?  q8_0 = recommended (frees VRAM, forces flash-attn) · f16 = full precision, more VRAM" "${KVQUANT:-q8_0}" '^(q8_0|q5_1|q5_0|q4_1|q4_0|f16|bf16|f32)$'
KVQUANT="$ANS"

ask "Experts kept on CPU (NCMOE)?  lower = faster but risks VRAM OOM · 27 = tuned for 64k/q8 (leave unless tuning)" "${NCMOE:-27}" '^[0-9]+$'
NCMOE="$ANS"

ask "Backend?  cuda = fast (recommended) · vulkan = fallback · cpu = ~2 tok/s (testing only)" "${BACKEND:-cuda}" '^(cuda|vulkan|cpu)$'
BACKEND="$ANS"

ask "Port?" "${PORT:-8080}" '^[0-9]+$'
PORT="$ANS"

ask "Enable vision / multimodal?  y / n" "n" '^[yYnN]$'
IMG="$ANS"; IMG_FLAG=(); { [ "$IMG" = y ] || [ "$IMG" = Y ]; } && IMG_FLAG=(--image)

# Build MTP flags (p_min hard-pinned to 0 — see header).
SPEC=""
if [ "$MTP" = y ] || [ "$MTP" = Y ]; then
  SPEC="--spec-type draft-mtp --spec-draft-model $HEAD --spec-draft-n-max $NMAX --spec-draft-p-min 0"
fi

echo
echo "---------------------------------------------------------"
echo " Launching:  MTP=${MTP^^}${NMAX:+  n-max=$NMAX}  temp=$TEMP  ctx=$CTX  kv=$KVQUANT  ncmoe=$NCMOE  backend=$BACKEND  port=$PORT${IMG_FLAG:+  +image}"
[ -n "$SPEC" ] && echo " MTP flags:  $SPEC"
echo " (server runs here in the foreground — in another terminal run:  bash scripts/run-pi.sh )"
echo "---------------------------------------------------------"

export BACKEND CTX NCMOE KVQUANT TEMP PORT
export EXTRA_ARGS="${EXTRA_ARGS:-} $SPEC"

if [ "${DRY_RUN:-}" = 1 ]; then
  echo "[DRY_RUN] would exec: BACKEND=$BACKEND CTX=$CTX NCMOE=$NCMOE KVQUANT=$KVQUANT TEMP=$TEMP PORT=$PORT EXTRA_ARGS='$EXTRA_ARGS' bash scripts/run-server.sh ${IMG_FLAG[*]}"
  exit 0
fi
exec bash "$REPO_ROOT/scripts/run-server.sh" "${IMG_FLAG[@]}"

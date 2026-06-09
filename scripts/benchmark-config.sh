#!/usr/bin/env bash
#
# benchmark-config.sh — find the fastest NCMOE / CTX config for YOUR GPU.
#
# The two knobs that decide speed-vs-context on an 8 GB card (see the README's
# "Performance & tuning") trade against the same VRAM budget:
#   * CTX    bigger context  -> bigger KV cache in VRAM -> less room for experts
#   * NCMOE  fewer layers' experts on CPU (lower N) -> more on GPU -> faster,
#            but more VRAM, so it OOMs sooner.
# The sweet spot is GPU-specific, so this probes real configs on YOUR hardware.
#
# How it works: for each (CTX, NCMOE) pair it launches the *real* server
# (scripts/run-server.sh, so the measured tok/s is exactly what pi will see),
# waits for /health, times one short /completion, then shuts it down. A config
# that dies while loading is reported as OOM/fail, not a crash. At the end it
# prints, per context size, the fastest NCMOE that fit = your recommended
# start.sh setting.
#
# Two ways to use it:
#   * Pin a context, optimise NCMOE for it (set CTX):
#       CTX=32768 bash scripts/benchmark-config.sh
#   * Sweep several contexts (default; uses CTX_LIST):
#       bash scripts/benchmark-config.sh
#
# Config (override via env vars):
#   CTX        a single context to optimise for (takes precedence over CTX_LIST)
#   CTX_LIST   contexts to sweep when CTX is unset   (default: 16384,32768,65536)
#   NCMOE_LIST NCMOE values to try, low=faster/more-VRAM (default: 20,22,24,27,30)
#              30 = all 30 layers' experts on CPU (== --cpu-moe, slowest/safest).
#   BACKEND    cuda | vulkan | cpu                   (default: auto — cuda if built)
#   KVQUANT    KV-cache quant to measure under (q8_0, q5_1, ...; default f16/off).
#              Tuning is keyed by it, so each KV setting gets its own results.
#   MODEL      path to the .gguf                     (default: ./models/.../UD-Q4_K_XL.gguf)
#   PORT       probe port — kept off 8080 so it never touches a real server (default: 8099)
#   N_PREDICT  tokens to generate per timed run (after a discarded warmup) (default: 128)
#   RUNS       timed runs per config; the MEDIAN is reported (default: 5). The GPU's
#              boost clock bounces between probes, so one sample is noisy; the median
#              of several makes the ranking stop depending on which probe clocked high.
#   PROMPT     prompt used for the timed generation  (default: a short fixed string)
#   LOAD_TIMEOUT  seconds to wait for a config to load before calling it failed (default: 180)
#   ENV_NAME / CUDA_ENV / CUDA_BIN  passed through to run-server.sh as-is.
#
# Notes:
#   * tok/s is measured at LOW context fill; real throughput drops as the context
#     fills up (that deep-context cost is exactly why this doesn't prefill 128K).
#     Use the numbers to RANK configs (the ranking holds); the absolute value is
#     an optimistic ceiling.
#   * Each config is timed RUNS times and reported as the median, with the min–max
#     spread shown alongside so you can see how noisy that config was.
#   * Probing reloads the ~14 GB model once per config (the slow part); the RUNS
#     extra generations are cheap (~seconds each). The default grid is ~15 configs
#     (~12 min). Trim CTX_LIST / NCMOE_LIST, or lower RUNS, to go faster.
#
# Examples:
#   CTX=32768 bash scripts/benchmark-config.sh              # optimise NCMOE for 32K
#   CTX=131072 NCMOE_LIST=27,30 bash scripts/benchmark-config.sh
#   CTX_LIST=8192,16384,32768 bash scripts/benchmark-config.sh
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared backend banner + resolution (same source as run-server.sh — no dupes).
source "$REPO_ROOT/scripts/_banner.sh"
# Shared tuning cache so start.sh can reuse what we measure (tune_get/tune_set).
source "$REPO_ROOT/scripts/_tuning.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8099}"                       # off 8080 so we never disturb a real server
NCMOE_LIST="${NCMOE_LIST:-20,22,24,27,30}"
N_PREDICT="${N_PREDICT:-128}"
RUNS="${RUNS:-5}"                          # timed generations per config; report the MEDIAN
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. Tell me a short story.}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-180}"
# A 2070's boost clock bounces between probes (each reloads 14 GB, GPU idles, ramps
# back unevenly), so a single timed run is noisy. Take RUNS samples and report the
# median — the ranking stops depending on which probe happened to clock high.
case "$RUNS" in ''|*[!0-9]*|0) RUNS=5 ;; esac

# CTX (single, pinned) wins over CTX_LIST (sweep).
if [ -n "${CTX:-}" ]; then
  CTX_LIST="$CTX"
else
  CTX_LIST="${CTX_LIST:-16384,32768,65536}"
fi

BACKEND="$(resolve_backend)"
KV="${KVQUANT:-f16}"                        # tuning-cache dimension (f16 = unquantized KV)
SERVER_LOG="${SERVER_LOG:-/tmp/gemma4-benchmark-server.log}"
RESULTS="$(mktemp)"                        # ctx|ncmoe|status|tokps|pp|vram

command -v curl >/dev/null 2>&1 || { echo "ERROR: 'curl' not found on PATH."; exit 1; }

# Don't trample a server the user already has on the probe port.
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "ERROR: something is already listening on $HOST:$PORT."
  echo "       Set PORT= to a free port (the benchmark needs one to itself)."
  exit 1
fi

# --- cleanup: never orphan a 14 GB probe server -----------------------------
cleanup() { PORT="$PORT" bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1 || true; rm -f "$RESULTS"; }
trap cleanup EXIT INT TERM

# nvidia-smi VRAM-used sampler (optional; blank if unavailable).
gpu_mib() { command -v nvidia-smi >/dev/null 2>&1 &&
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true; }

splash "$_FG_GREEN" "📊" "GEMMA 4 CONFIG BENCHMARK" "backend: $BACKEND  ·  KV: $KV  ·  probe port: $PORT"
echo ">> contexts: $CTX_LIST    NCMOE: $NCMOE_LIST    ($RUNS runs × $N_PREDICT tok, median per config)"
echo ">> tok/s is measured at low context fill — use it to rank, not as a deep-context promise."
echo ">> each config runs $RUNS times (RUNS=) and reports the median — smooths out the GPU's clock bounce."
echo

# --- probe one (CTX, NCMOE) -------------------------------------------------
# Echoes (stdout): "<status> <med_tps> <med_pp> <vram> <lo_tps> <hi_tps>"
#   status: ok | oom | fail.  Live progress goes to stderr, never to stdout.
probe() {
  local ctx="$1" ncmoe="$2"
  : > "$SERVER_LOG"
  CTX="$ctx" NCMOE="$ncmoe" PORT="$PORT" BACKEND="$BACKEND" HOST="$HOST" KVQUANT="${KVQUANT:-}" \
    nohup bash "$REPO_ROOT/scripts/run-server.sh" > "$SERVER_LOG" 2>&1 &
  local pid=$!
  printf 'loading… ' >&2          # live progress (stderr → shown but not captured)

  # Wait for /health, or the process to die (OOM/fail), or timeout.
  local waited=0 ready=0
  while [ "$waited" -lt "$LOAD_TIMEOUT" ]; do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi   # exited before becoming healthy
    sleep 2; waited=$((waited + 2))
  done

  if [ "$ready" != 1 ]; then
    # Distinguish out-of-memory from any other load failure for a useful label.
    local status=fail
    if grep -qiE 'out of memory|failed to allocate|cudaMalloc|ggml_backend.*alloc|insufficient' "$SERVER_LOG"; then
      status=oom
    fi
    kill "$pid" 2>/dev/null || true
    PORT="$PORT" bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1 || true
    echo "$status   -   -   -   -   -"
    return
  fi

  local vram; vram="$(gpu_mib)"

  # Warm up first (discarded): the very first decode pays one-time CUDA kernel /
  # graph-capture cost that would unfairly penalise GPU-heavy (low-NCMOE) configs.
  curl -fsS "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(json_str "$PROMPT"),\"n_predict\":24,\"ignore_eos\":true,\"cache_prompt\":false,\"temperature\":0}" \
    >/dev/null 2>&1 || true

  # Then time RUNS generations and keep the MEDIAN — one sample is noisy because the
  # GPU's clock bounces between runs. ignore_eos => exactly N tokens for a stable
  # rate. Progress prints to stderr ("runs: 1/5 2/5 …") so it shows live but never
  # lands in the result the caller captures via $(probe ...).
  local tps_list="" pp_list="" r resp flat tps pp
  printf 'runs:' >&2
  for r in $(seq 1 "$RUNS"); do
    printf ' %d/%d' "$r" "$RUNS" >&2
    resp="$(curl -fsS "http://$HOST:$PORT/completion" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(json_str "$PROMPT"),\"n_predict\":$N_PREDICT,\"ignore_eos\":true,\"cache_prompt\":false,\"temperature\":0}" \
      2>/dev/null || true)"
    flat="$(printf '%s' "$resp" | tr -d ' \n')"
    tps="$(printf '%s' "$flat" | grep -oE '"predicted_per_second":[0-9.]+' | head -1 | cut -d: -f2)"
    pp="$( printf '%s' "$flat" | grep -oE '"prompt_per_second":[0-9.]+'    | head -1 | cut -d: -f2)"
    [ -n "$tps" ] && tps_list+="$tps"$'\n'
    [ -n "$pp"  ] && pp_list+="$pp"$'\n'
  done
  printf ' ' >&2   # one space so the caller's result doesn't butt against "5/5"

  # Collapse the samples: median for the headline, min/max to show the spread.
  local med_tps med_pp lo hi
  med_tps="$(printf '%s' "$tps_list" | _stat median)"
  med_pp="$( printf '%s' "$pp_list"  | _stat median)"
  lo="$(printf '%s' "$tps_list" | _stat min)"
  hi="$(printf '%s' "$tps_list" | _stat max)"

  PORT="$PORT" bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1 || true
  wait "$pid" 2>/dev/null || true

  if [ -n "$med_tps" ]; then
    echo "ok   $med_tps   ${med_pp:--}   ${vram:--}   ${lo:--}   ${hi:--}"
  else
    echo "fail   -   -   ${vram:--}   -   -"   # loaded but no usable generation
  fi
}

# JSON-encode a string (so prompts with quotes/specials are safe).
json_str() { printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
  || printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }

# Round a numeric field to 1 decimal; pass non-numbers ("-") through untouched.
round1() { case "$1" in ''|*[!0-9.]*) printf '%s' "${1:--}";; *) printf '%.1f' "$1";; esac; }

# _stat <median|min|max>: read numbers (one per line) on stdin, print the stat to
# 1 decimal (empty if there were none). Used to collapse the RUNS samples per probe.
_stat() {
  awk -v what="$1" '
    /[0-9]/ { a[n++] = $1 + 0 }
    END {
      if (n == 0) exit
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (a[j] < a[i]) { t = a[i]; a[i] = a[j]; a[j] = t }
      if      (what == "min") printf "%.1f", a[0]
      else if (what == "max") printf "%.1f", a[n - 1]
      else if (n % 2)         printf "%.1f", a[(n - 1) / 2]
      else                    printf "%.1f", (a[n/2 - 1] + a[n/2]) / 2
    }'
}

# Wait for the probe port to free up before the next launch.
wait_port_free() {
  local n=0
  while curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    [ "$n" -ge 15 ] && break; sleep 2; n=$((n + 1))
  done
}

# --- the sweep --------------------------------------------------------------
IFS=',' read -r -a CTXS <<< "$CTX_LIST"
IFS=',' read -r -a NCMOES <<< "$NCMOE_LIST"

for ctx in "${CTXS[@]}"; do
  ctx="$(printf '%s' "$ctx" | tr -d ' ')"
  echo "── CTX=$ctx ──────────────────────────────────────────────"
  for ncmoe in "${NCMOES[@]}"; do
    ncmoe="$(printf '%s' "$ncmoe" | tr -d ' ')"
    printf "   NCMOE=%-3s … " "$ncmoe"
    wait_port_free
    read -r status tokps pp vram lo hi <<< "$(probe "$ctx" "$ncmoe")"
    case "$status" in
      ok)
        if [ "${lo:--}" != "-" ] && [ "$lo" != "$hi" ]; then
          printf "✅ %s tok/s  (median of %s: %s–%s; pp %s, VRAM %s MiB)\n" \
            "$(round1 "$tokps")" "$RUNS" "$lo" "$hi" "$(round1 "$pp")" "$vram"
        else
          printf "✅ %s tok/s  (median of %s; pp %s, VRAM %s MiB)\n" \
            "$(round1 "$tokps")" "$RUNS" "$(round1 "$pp")" "$vram"
        fi ;;
      oom)  printf "🔴 OOM — doesn't fit\n" ;;
      *)    printf "⚠️  failed (see %s)\n" "$SERVER_LOG" ;;
    esac
    echo "$ctx|$ncmoe|$status|$tokps|$pp|$vram" >> "$RESULTS"
  done
  echo
done

# --- recommendation ---------------------------------------------------------
splash "$_FG_GREEN" "🏁" "RECOMMENDED CONFIGS" "fastest NCMOE that fit, per context"
printf "   %-9s %-7s %-9s %s\n" "CTX" "NCMOE" "tok/s" "start.sh command"
printf "   %-9s %-7s %-9s %s\n" "---" "-----" "-----" "----------------"
for ctx in "${CTXS[@]}"; do
  ctx="$(printf '%s' "$ctx" | tr -d ' ')"
  # Best = highest tok/s among the 'ok' rows for this ctx.
  best="$(awk -F'|' -v c="$ctx" '$1==c && $3=="ok"{print $4"\t"$2}' "$RESULTS" | sort -rn | head -1)"
  if [ -n "$best" ]; then
    tokps="$(printf '%s' "$best" | cut -f1)"
    ncmoe="$(printf '%s' "$best" | cut -f2)"
    printf "   %-9s %-7s %-9s %s\n" "$ctx" "$ncmoe" "$(round1 "$tokps")" \
      "CTX=$ctx NCMOE=$ncmoe${KVQUANT:+ KVQUANT=$KVQUANT} bash scripts/start.sh"
    tune_set "$(tune_key "$BACKEND" "$ctx" "$KV")" "$ncmoe"   # remember it so start.sh can reuse it
  else
    printf "   %-9s %-7s %-9s %s\n" "$ctx" "-" "-" "doesn't fit on this GPU — lower CTX or raise NCMOE"
    tune_set "$(tune_key "$BACKEND" "$ctx" "$KV")" nofit
  fi
done
echo
echo ">> Saved to $(tune_cache_file) — start.sh will reuse these (no re-run needed)."
echo ">> Remember to pass the SAME CTX to configure-pi.sh:  CTX=<ctx> bash scripts/configure-pi.sh"

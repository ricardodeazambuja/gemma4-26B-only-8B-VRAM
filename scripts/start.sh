#!/usr/bin/env bash
#
# start.sh — one command to bring up the model server (if not already running)
# and drop you into pi.
#
# Starts scripts/run-server.sh in the background (if nothing is serving yet),
# waits for it to finish loading, then launches pi in the foreground. When pi
# exits, if THIS script started the server it offers to stop it (a server that
# was already running when you invoked start.sh is always left alone).
#
# All run-server.sh knobs pass through via env, e.g.:
#   BACKEND=cuda NCMOE=22 bash scripts/start.sh
#   CTX=32768 bash scripts/start.sh
#   bash scripts/start.sh --image                     # forwarded to the server
#   CTX=131072 KVQUANT=q8_0 bash scripts/start.sh      # quantized KV for long context
#   bash scripts/start.sh -p "summarize @README.md"   # other args go to pi
#
#   KVQUANT also keys the auto-tune cache (backend × context × KV quant), so the
#   tuned expert split is measured per KV-quant setting — they don't collide.
#
#   STOP_ON_EXIT  control the shutdown offer after pi exits:
#                 unset = ask (only when interactive); 1 = always stop; 0 = never
#
#   AUTOTUNE      measure the fastest expert split (NCMOE) on YOUR GPU the first
#                 time you launch a fresh server, then REMEMBER it (gitignored
#                 cache) so later launches reuse it instantly. Two modes:
#                   * CTX pinned (you passed CTX=)  -> tune NCMOE for that context.
#                   * no CTX (the default)          -> sweep several contexts
#                       (CTX_LIST × NCMOE_LIST), show the fastest that fits at
#                       each, and let you PICK which context to launch; your
#                       pick is remembered for next time.
#                   unset = reuse a saved result, else ask (interactive only)
#                   1     = force a fresh measurement/sweep (ignore the cache)
#                   0     = never tune, never ask
#                 An explicit NCMOE= always wins and skips tuning entirely.
#                 Override the sweep grid with CTX_LIST= / NCMOE_LIST=.
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared backend banner + resolution (same source as run-server.sh — no dupes).
source "$REPO_ROOT/scripts/_banner.sh"
# Shared auto-tuning cache (tune_get/tune_set) — see scripts/_tuning.sh.
source "$REPO_ROOT/scripts/_tuning.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# Did the user pin a context, or are we on the default? Auto-tuning behaves
# differently: an explicit CTX tunes only NCMOE for it; no CTX offers a sweep
# across context sizes and lets you pick which to launch.
if [ -n "${CTX:-}" ]; then CTX_EXPLICIT=1; else CTX_EXPLICIT=0; fi
export CTX="${CTX:-32768}"         # effective context (also the auto-tune cache key); export so run-server.sh sees the same value
export KVQUANT="${KVQUANT:-}"      # KV-cache quant; a tuning-cache dimension too. export so run-server.sh + benchmark see it
SERVER_LOG="${SERVER_LOG:-/tmp/gemma4-server.log}"
STOP_ON_EXIT="${STOP_ON_EXIT:-}"   # unset = ask, 1 = always, 0 = never
STARTED_SERVER=0                   # set to 1 only if we launch it below

# Split args: server flags (currently just --image) go to run-server.sh, the
# rest go to pi. Without this, --image would be sent to pi and silently ignored.
SERVER_ARGS=()
PI_ARGS=()
for a in "$@"; do
  case "$a" in
    --image) SERVER_ARGS+=("$a") ;;
    *) PI_ARGS+=("$a") ;;
  esac
done

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> server already running at http://$HOST:$PORT — reusing it"
  # Surface the backend the *running* server reported, read from its log (its
  # banner went there). Skipped silently if the log is absent/stale.
  reused_be="$(sed -n 's/.*BACKEND: \([A-Za-z]*\).*/\1/p' "$SERVER_LOG" 2>/dev/null | tail -1 | tr '[:upper:]' '[:lower:]')"
  [ -n "$reused_be" ] && backend_banner "$reused_be"
  if [ ${#SERVER_ARGS[@]} -gt 0 ]; then
    echo ">> NOTE: ${SERVER_ARGS[*]} only applies to a fresh server; the running one is reused as-is."
    echo "         Restart it to apply: bash scripts/stop-server.sh && bash scripts/start.sh ${SERVER_ARGS[*]}"
  fi
else
  # Show which backend the server will come up on (same logic run-server.sh uses).
  backend_banner "$(resolve_backend)"

  # --- auto-tuning -----------------------------------------------------------
  # Results are remembered in $(tune_cache_file) so the slow measurement runs
  # once, not on every launch. Controlled by AUTOTUNE: unset = reuse a saved
  # result, else ask (interactive only); 1 = force a fresh measurement; 0 = off.
  # An explicit NCMOE= always wins (skips tuning). Two modes:
  #   * CTX pinned (you passed CTX=)  -> tune only NCMOE for that one context.
  #   * no CTX (the default)          -> sweep several contexts, then you pick
  #                                       which to launch (remembered as 'chosen').
  BE="$(resolve_backend)"
  KV="${KVQUANT:-f16}"             # tuning-cache dimension (f16 = unquantized KV)
  bench_port=8099; [ "$bench_port" = "$PORT" ] && bench_port=8100
  if [ -n "${NCMOE:-}" ]; then
    echo ">> NCMOE=$NCMOE set explicitly — skipping auto-tuning."
  elif [ "${AUTOTUNE:-}" = "0" ]; then
    : # auto-tuning disabled

  elif [ "$CTX_EXPLICIT" = 1 ]; then
    # ---- pinned context: tune the expert split for this CTX only ----
    saved="$(tune_get "$(tune_key "$BE" "$CTX" "$KV")")"
    [ "${AUTOTUNE:-}" = "1" ] && saved=""   # force a fresh measurement
    if [[ "$saved" =~ ^[0-9]+$ ]]; then
      export NCMOE="$saved"
      echo ">> auto-tune: reusing saved NCMOE=$NCMOE for BACKEND=$BE CTX=$CTX KV=$KV (re-measure: AUTOTUNE=1 bash scripts/start.sh)"
    elif [ "$saved" = "declined" ]; then
      : # the user previously said no for this key — don't nag, launch as-is
    elif [ "$saved" = "nofit" ]; then
      echo ">> auto-tune: no expert split fit CTX=$CTX here before — launching with all experts on CPU (may still OOM; consider a smaller CTX)."
    elif [ -t 0 ]; then
      nlist="${NCMOE_LIST:-22,27,30}"
      ncount="$(awk -F, '{print NF}' <<< "$nlist")"
      echo
      echo ">> No tuned config yet for BACKEND=$BE CTX=$CTX."
      echo "   Auto-tuning briefly launches the server $ncount time(s) (splits: $nlist) to find the"
      echo "   fastest one that fits — roughly ${ncount}–$((ncount * 2)) min. The result is saved and reused next time."
      read -r -p ">> Run auto-tuning now? [y/N] " ans || ans=""
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        echo ">> measuring (this is the slow part — once) ..."
        CTX="$CTX" BACKEND="$BE" KVQUANT="$KVQUANT" NCMOE_LIST="$nlist" PORT="$bench_port" \
          bash "$REPO_ROOT/scripts/benchmark-config.sh" || true
        best="$(tune_get "$(tune_key "$BE" "$CTX" "$KV")")"
        if [[ "$best" =~ ^[0-9]+$ ]]; then
          export NCMOE="$best"
          echo ">> auto-tune: applying NCMOE=$NCMOE for CTX=$CTX KV=$KV."
        else
          echo ">> auto-tune: no fitting split measured — launching with the default expert split."
        fi
      else
        tune_set "$(tune_key "$BE" "$CTX" "$KV")" declined
        echo ">> skipped — launching with the default expert split. Re-enable later: AUTOTUNE=1 bash scripts/start.sh"
      fi
    fi

  else
    # ---- no context pinned: sweep contexts, then let the user pick one ----
    chosen="$(tune_get "$(tune_key "$BE" chosen "$KV")")"
    [ "${AUTOTUNE:-}" = "1" ] && chosen=""   # force a fresh exploration
    if [[ "$chosen" =~ ^[0-9]+$ ]]; then
      export CTX="$chosen"
      ncmoe_for="$(tune_get "$(tune_key "$BE" "$chosen" "$KV")")"
      [[ "$ncmoe_for" =~ ^[0-9]+$ ]] && export NCMOE="$ncmoe_for"
      echo ">> auto-tune: reusing your saved choice CTX=$CTX${NCMOE:+ NCMOE=$NCMOE} KV=$KV for BACKEND=$BE (re-explore: AUTOTUNE=1 bash scripts/start.sh)"
    elif [ "$chosen" = "declined" ]; then
      : # explored-and-declined before — launch the default, don't nag
    elif [ -t 0 ]; then
      clist="${CTX_LIST:-16384,32768,65536,131072}"
      nlist="${NCMOE_LIST:-22,27,30}"
      nprobes=$(( $(awk -F, '{print NF}' <<< "$clist") * $(awk -F, '{print NF}' <<< "$nlist") ))
      echo
      echo ">> No context tuned yet for BACKEND=$BE. Auto-tuning can sweep several context"
      echo "   sizes ($clist) × expert splits ($nlist) and show the fastest that fits at each,"
      echo "   so you can pick how much context to run. It launches the server up to $nprobes time(s)"
      echo "   — roughly ${nprobes}–$(( nprobes + nprobes/2 )) min, measured once and remembered."
      read -r -p ">> Run the context sweep now? [y/N] " ans || ans=""
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        echo ">> sweeping (this is the slow part — once) ..."
        # CTX= (empty) so benchmark-config.sh sweeps CTX_LIST instead of pinning our default.
        CTX= BACKEND="$BE" KVQUANT="$KVQUANT" CTX_LIST="$clist" NCMOE_LIST="$nlist" PORT="$bench_port" \
          bash "$REPO_ROOT/scripts/benchmark-config.sh" || true
        # Build the menu from what fit (cache holds backend:ctx:kv=NCMOE per swept ctx).
        opts=()
        IFS=',' read -r -a _cl <<< "$clist"
        for c in "${_cl[@]}"; do
          c="$(printf '%s' "$c" | tr -d ' ')"
          v="$(tune_get "$(tune_key "$BE" "$c" "$KV")")"
          [[ "$v" =~ ^[0-9]+$ ]] && opts+=("$c:$v")
        done
        if [ "${#opts[@]}" -eq 0 ]; then
          echo ">> auto-tune: nothing fit on this GPU — launching with the default expert split."
        else
          echo
          echo ">> Which context to launch with? (tok/s for each is in the table above)"
          i=1; for o in "${opts[@]}"; do printf "     [%d] CTX=%-7s NCMOE=%s\n" "$i" "${o%%:*}" "${o##*:}"; i=$((i+1)); done
          dflt="${#opts[@]}"   # default = the largest context that fit (last entry)
          read -r -p ">> choice [${dflt}=largest]: " pick || pick=""
          [[ "$pick" =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -le "${#opts[@]}" ] || pick="$dflt"
          sel="${opts[$((pick-1))]}"
          export CTX="${sel%%:*}"
          export NCMOE="${sel##*:}"
          tune_set "$(tune_key "$BE" chosen "$KV")" "$CTX"
          echo ">> auto-tune: launching at CTX=$CTX NCMOE=$NCMOE KV=$KV (re-explore: AUTOTUNE=1 bash scripts/start.sh)"
          echo ">> NOTE: keep pi in sync with this context:  CTX=$CTX bash scripts/configure-pi.sh"
        fi
      else
        tune_set "$(tune_key "$BE" chosen "$KV")" declined
        echo ">> skipped — launching at the default CTX=$CTX. Re-enable later: AUTOTUNE=1 bash scripts/start.sh"
      fi
    fi
  fi

  echo ">> starting server in background (logs: $SERVER_LOG) ..."
  nohup bash "$REPO_ROOT/scripts/run-server.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
  SRV_PID=$!
  printf ">> waiting for the model to load"
  for _ in $(seq 1 150); do          # up to ~5 min
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then printf " ready.\n"; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      printf "\nERROR: server exited while starting. Last log lines:\n"
      tail -20 "$SERVER_LOG"; exit 1
    fi
    printf "."; sleep 2
  done
  if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo; echo "ERROR: server did not become healthy in time. See $SERVER_LOG"; exit 1
  fi
  echo ">> server up (PID $SRV_PID). Stop it later with: bash scripts/stop-server.sh"
  STARTED_SERVER=1
fi

echo ">> launching pi ..."
# Run pi in the foreground (not exec) so control returns here when it exits.
# `|| PI_RC=$?` keeps `set -e` from aborting on a non-zero pi exit (e.g. Ctrl-C).
PI_RC=0
bash "$REPO_ROOT/scripts/run-pi.sh" "${PI_ARGS[@]}" || PI_RC=$?

# --- offer to shut the server down ------------------------------------------
# Only when we started it this run and it's still up. A reused server is the
# user's pre-existing process — leave it alone.
if [ "$STARTED_SERVER" = 1 ] && curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  do_stop=""
  case "$STOP_ON_EXIT" in
    1) do_stop=yes ;;
    0) do_stop=no ;;
    *)
      if [ -t 0 ]; then
        read -r -p $'\n>> pi exited. Stop the model server too? [y/N] ' ans || ans=""
        [[ "$ans" =~ ^[Yy]$ ]] && do_stop=yes || do_stop=no
      else
        do_stop=no   # non-interactive (e.g. -p one-shot): don't guess, leave it up
      fi
      ;;
  esac

  if [ "$do_stop" = yes ]; then
    bash "$REPO_ROOT/scripts/stop-server.sh"
  else
    echo ">> leaving the server running. Stop it with: bash scripts/stop-server.sh"
  fi
fi

exit "$PI_RC"

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
# Prefer a guided walk-through over remembering env vars?
#   bash scripts/start.sh --menu     # interactive setup: backend, auto-tune vs
#                                     # manual, context, KV-quant, sampling, image
#
# First interactive run also offers (once) to link this repo's pi-extensions into
# pi — local-Gemma tools (code outlines, verified edits, memory, web fetch). Decline
# and it won't ask again (rm ~/.pi/agent/.gemma4-ext-bootstrap-skip to re-enable).
#
# Settings (env vars; all forwarded to run-server.sh — see its -h for the rest):
#   BACKEND  cuda | vulkan | cpu          (default: auto-detect)
#   CTX      context window, in tokens    (default: 32768)
#   NCMOE    expert layers kept on CPU; lower = more on GPU = faster (default: all)
#   KVQUANT  KV-cache quant: f16(off) | q8_0 | q5_1 | q4_0 | ...
#            (default: the KV quant you picked in a previous sweep/--menu,
#             remembered in the tuning cache; f16/off if you never picked one)
#   TEMP / TOP_P / TOP_K  sampling        (defaults: 1.0 / 0.95 / 64)
#   MTP      enable MTP self-speculative decoding (MTP=1 | y | on; default off)
#   NMAX     MTP draft tokens per step    (default: 2; only used when MTP is on)
#            MTP is lossless and speeds up LOW-temp / coding (~+15-30%); ~no gain at
#            temp 1.0. p_min is pinned to 0 (any p_min>0 degenerates the output).
#   --image  load the vision projector so the server accepts images
#
#   KVQUANT also keys the auto-tune cache (backend × context × KV quant), so the
#   tuned expert split is measured per KV-quant setting — they don't collide.
#   The first sweep explores the KV dimension too (KVQUANT_LIST, default
#   f16,q8_0) and your pick — context AND KV quant — is what gets remembered.
#
#   STOP_ON_EXIT  control the shutdown offer after pi exits:
#                 unset = ask (only when interactive); 1 = always stop; 0 = never
#
#   AUTOTUNE      measure the fastest expert split (NCMOE) on YOUR GPU the first
#                 time you launch a fresh server, then REMEMBER it (gitignored
#                 cache) so later launches reuse it instantly. Two modes:
#                   * CTX pinned (you passed CTX=)  -> tune NCMOE for that context.
#                   * no CTX (the default)          -> sweep several contexts
#                       (CTX_LIST × NCMOE_LIST × KVQUANT_LIST — KV is part of
#                       the grid unless pinned by KVQUANT= or a remembered
#                       pick), show the fastest that fits at each, and let you
#                       PICK which config to launch; your pick — context AND
#                       KV quant — is remembered for next time.
#                   unset = reuse a saved result, else ask (interactive only)
#                   1     = force a fresh measurement/sweep (ignore the cache)
#                   0     = never tune, never ask
#                 An explicit NCMOE= always wins and skips tuning entirely.
#                 Override the sweep grid with CTX_LIST= / NCMOE_LIST= / KVQUANT_LIST=.
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

# Actionable hints printed after a server fails to start (OOM / missing build /
# port clash). Called from both startup-failure paths in the launch block below.
server_start_hints() {
  echo ">> Common causes & fixes:"
  echo "   • out of GPU memory — smaller context (CTX=32768), push more experts to CPU"
  echo "     (raise NCMOE, e.g. NCMOE=27), or quantize the KV cache (KVQUANT=q8_0)."
  echo "   • CUDA build missing — build it once:  bash scripts/build-llama-cuda.sh"
  echo "   • port $PORT already in use — stop a stale server:  bash scripts/stop-server.sh"
  echo "   • full log:  $SERVER_LOG"
}

# One-time bootstrap: a fresh clone has the pi-extensions in the repo but not yet
# linked into pi. On the first interactive launch, offer to run the installer.
# Guards (each a silent skip): no extensions in the repo · already linked ·
# previously declined (a marker so we never nag) · non-interactive (never block
# -p / piped runs). Explicit if-blocks, not `&& return`, to stay set -e safe.
maybe_bootstrap_extensions() {
  local dest="${PI_EXT_DIR:-$HOME/.pi/agent/extensions}"
  local setup="$REPO_ROOT/pi-extensions/setup.sh"
  local skip_marker="$HOME/.pi/agent/.gemma4-ext-bootstrap-skip"

  if [ ! -f "$setup" ]; then return 0; fi              # repo has no extensions
  if [ -e "$dest/verified-edits" ]; then return 0; fi  # already linked -> done
  if [ -f "$skip_marker" ]; then return 0; fi          # declined before -> hush
  if [ ! -t 0 ]; then return 0; fi                     # non-interactive -> skip

  echo
  echo ">> The pi-extensions in this repo aren't linked into pi yet —"
  echo "   local-Gemma tools: code outlines, verified edits, memory, web fetch, …"
  local ans
  read -r -p ">> Install them now? (one-time) [Y/n] " ans || ans=""
  if [[ "$ans" =~ ^[Nn] ]]; then
    mkdir -p "$(dirname "$skip_marker")"
    : > "$skip_marker"
    echo ">> skipped — won't ask again. Install later:  bash pi-extensions/setup.sh"
    echo "   (re-enable this prompt:  rm $skip_marker)"
  else
    echo ">> running pi-extensions/setup.sh …"
    bash "$setup" || echo ">> setup didn't finish cleanly — re-run anytime: bash pi-extensions/setup.sh"
  fi
  echo
}

# --- interactive setup (--menu) ---------------------------------------------
# A guided walk-through of every knob and the auto-tune-vs-manual fork. It sets
# the SAME env vars the equivalent command line would (BACKEND/CTX/NCMOE/KVQUANT/
# TEMP.../AUTOTUNE) and toggles --image, then the normal launch path below runs
# unchanged. Reuses the read-prompt style of the auto-tune block (one universe).

# --- menu input helpers (every prompt can be cancelled) ---------------------
# All three honor a universal escape hatch: typing q / quit / cancel — or hitting
# Ctrl-D (EOF) — aborts the whole setup cleanly. They return via the global
# MENU_REPLY (a function in $(...) can't exit the script, so we don't use it).

_menu_cancel() { echo; echo ">> cancelled — nothing started."; exit 0; }

# Numbered choice in 1..$3. Enter = $2 (default). Re-asks on out-of-range input
# so a typo never silently selects the wrong option.
_menu_choice() {
  local prompt="$1" def="$2" max="$3" ans
  while true; do
    read -r -p "$prompt" ans || _menu_cancel
    case "$ans" in q|Q|quit|cancel) _menu_cancel ;; esac
    [ -z "$ans" ] && { MENU_REPLY="$def"; return; }
    if [[ "$ans" =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le "$max" ]; then
      MENU_REPLY="$ans"; return
    fi
    echo "   ↳ please enter a number 1-$max (or q to cancel)."
  done
}

# Free-text answer. Enter = $2 (default, may be empty). q / Ctrl-D = cancel.
_menu_text() {
  local prompt="$1" def="$2" ans
  read -r -p "$prompt" ans || _menu_cancel
  case "$ans" in q|Q|quit|cancel) _menu_cancel ;; esac
  [ -z "$ans" ] && MENU_REPLY="$def" || MENU_REPLY="$ans"
}

# Yes/No. Enter = $2 ("yes"|"no"). q / Ctrl-D = cancel; unrecognized = default.
_menu_yesno() {
  local prompt="$1" def="$2" ans
  read -r -p "$prompt" ans || _menu_cancel
  case "$ans" in
    q|Q|quit|cancel) _menu_cancel ;;
    y|Y|yes|Yes)     MENU_REPLY=yes ;;
    n|N|no|No)       MENU_REPLY=no ;;
    *)               MENU_REPLY="$def" ;;
  esac
}

# Ask for a context size; export CTX (suggests common values; defaults to 32768).
_menu_ask_ctx() {
  echo "   Context size, in tokens — bigger = more room for history, but more VRAM."
  echo "     common: 16384   32768   65536   131072 (128K)"
  _menu_text "   CTX [default 32768]: " 32768
  if [[ "$MENU_REPLY" =~ ^[0-9]+$ ]]; then
    export CTX="$MENU_REPLY"
  else
    echo "   ↳ not a whole number — using 32768."; export CTX=32768
  fi
}

configure_menu() {
  splash "$_FG_GREEN" "🛠️" "GEMMA 4 INTERACTIVE SETUP" "configure the server before it launches"
  echo "   At any prompt:  Enter = the [default]   ·   q = cancel and quit."
  echo
  local _menu_strategy

  # 1) Backend ---------------------------------------------------------------
  local detected; detected="$(resolve_backend)"
  echo "1) Backend  —  the compute path  (auto-detected: $detected)"
  echo "     1) auto    use the detected backend ($detected)"
  echo "     2) cuda    NVIDIA GPU, fast path (needs the CUDA build)"
  echo "     3) vulkan  any GPU, slower MoE path"
  echo "     4) cpu     no GPU offload — very slow (testing only)"
  _menu_choice "   choice [1-4, default 1]: " 1 4
  case "$MENU_REPLY" in
    2) export BACKEND=cuda ;;
    3) export BACKEND=vulkan ;;
    4) export BACKEND=cpu ;;
    *) unset BACKEND || true ;;            # auto: let resolve_backend decide
  esac
  local BE; BE="$(resolve_backend)"

  # 2) Context & expert split: automatic vs manual ---------------------------
  echo
  echo "2) Context & expert split  —  how to size context and place experts"
  echo "     1) auto-tune  measure the fastest split that fits on YOUR GPU (recommended)"
  echo "     2) manual     I'll pick the context and expert split myself"
  _menu_choice "   choice [1-2, default 1]: " 1 2
  if [ "$MENU_REPLY" = 2 ]; then _menu_strategy=manual; else _menu_strategy=auto; fi

  if [ "$_menu_strategy" = auto ]; then
    # Reuse a saved tuned result if present, else MEASURE. The user explicitly
    # opted into auto-tuning here, so TUNE_YES tells the block below to skip its
    # "Run auto-tuning now? [y/N]" prompt (whose default-No would otherwise be a
    # sticky 'declined' on the no-cache path — e.g. the recommended q8_0 one).
    unset AUTOTUNE || true
    TUNE_YES=1
    echo
    echo "   ⏳ Heads-up: the FIRST time, auto-tuning launches the server several"
    echo "      times to measure what fits — a few minutes (longer for a sweep)."
    echo "      The result is saved, so every later launch is instant."
    echo "     1) sweep several contexts, then let me pick which to launch (default)"
    echo "     2) auto-tune the expert split for ONE context I choose"
    _menu_choice "   choice [1-2, default 1]: " 1 2
    if [ "$MENU_REPLY" = 2 ]; then
      _menu_ask_ctx; CTX_EXPLICIT=1
    else
      CTX_EXPLICIT=0                        # CTX stays the default; the sweep picks
    fi
  else
    _menu_ask_ctx; CTX_EXPLICIT=1
    echo
    echo "   Expert split (NCMOE) — how many of the 30 layers keep experts on CPU."
    echo "     lower  = more experts on GPU = faster, but more VRAM (can OOM)"
    echo "     higher = gentler on VRAM, slower    ·    blank = all on CPU (safest)"
    echo "     typical on an 8 GB card: 20-27."
    _menu_text "   NCMOE [blank = all on CPU]: " ""
    if [[ "$MENU_REPLY" =~ ^[0-9]+$ ]]; then export NCMOE="$MENU_REPLY"; else unset NCMOE || true; fi
    export AUTOTUNE=0                        # manual choice: never tune
  fi

  # 3) KV-cache quantization -------------------------------------------------
  echo
  echo "3) KV-cache quantization  —  frees VRAM; mainly a long-context lever"
  echo "     1) f16   off, full precision (default)"
  echo "     2) q8_0  near-lossless; recommended for 64K+ context"
  echo "     3) q4_0  aggressive: most VRAM saved, some quality cost"
  echo "     4) other pick another type (q5_1, q5_0, q4_1, iq4_nl, bf16, f32)"
  _menu_choice "   choice [1-4, default 1]: " 1 4
  case "$MENU_REPLY" in
    2) export KVQUANT=q8_0 ;;
    3) export KVQUANT=q4_0 ;;
    4) _menu_text "   KV type (q5_1|q5_0|q4_1|iq4_nl|bf16|f32) [f16]: " f16
       export KVQUANT="$MENU_REPLY" ;;
    *) export KVQUANT=f16 ;;   # explicit, so a cache-saved KV pick can't override the menu
  esac

  # 4) Sampling --------------------------------------------------------------
  echo
  echo "4) Sampling  —  generation randomness (server-wide default)"
  echo "     1) unsloth defaults — temp=1.0  top-p=0.95  top-k=64 (recommended)"
  echo "     2) custom — enter your own"
  _menu_choice "   choice [1-2, default 1]: " 1 2
  if [ "$MENU_REPLY" = 2 ]; then
    _menu_text "   temp  (0.0-2.0)  [1.0]:  " ""; [ -n "$MENU_REPLY" ] && export TEMP="$MENU_REPLY"
    _menu_text "   top-p (0.0-1.0)  [0.95]: " ""; [ -n "$MENU_REPLY" ] && export TOP_P="$MENU_REPLY"
    _menu_text "   top-k (integer)  [64]:   " ""; [ -n "$MENU_REPLY" ] && export TOP_K="$MENU_REPLY"
  fi

  # 5) Image input -----------------------------------------------------------
  echo
  echo "5) Image input  —  load the vision projector so the server accepts images"
  echo "     (~1.2 GB on CPU; leave off for text-only, the common case)"
  _menu_yesno "   enable images? [y/N] " no
  if [ "$MENU_REPLY" = yes ]; then
    case " ${SERVER_ARGS[*]} " in *" --image "*) : ;; *) SERVER_ARGS+=(--image) ;; esac
  fi

  # 6) MTP speculative decoding ----------------------------------------------
  echo
  echo "6) MTP speculative decoding  —  lossless self-speculation (Gemma 4 draft head)"
  echo "     speeds up LOW-temperature / coding generation (~+15-30%); ~no gain at temp 1.0"
  _menu_yesno "   enable MTP? [y/N] " no
  if [ "$MENU_REPLY" = yes ]; then
    MTP=1
    _menu_text "   draft tokens per step (n-max) [2]: " 2; NMAX="$MENU_REPLY"
  fi

  # Summary + confirm --------------------------------------------------------
  local img=off; case " ${SERVER_ARGS[*]} " in *" --image "*) img=on ;; esac
  echo
  echo "── Summary ──────────────────────────────────────────────"
  printf "   backend : %s\n"                 "$BE"
  printf "   strategy: %s\n"                 "$_menu_strategy"
  printf "   context : %s\n"                 "$([ "$CTX_EXPLICIT" = 1 ] && echo "$CTX" || echo 'auto (sweep & pick)')"
  printf "   NCMOE   : %s\n"                 "${NCMOE:-$([ "$_menu_strategy" = manual ] && echo 'all on CPU' || echo 'auto')}"
  printf "   KV quant: %s\n"                 "${KVQUANT:-f16 (off)}"
  printf "   sampling: temp=%s top-p=%s top-k=%s\n" "${TEMP:-1.0}" "${TOP_P:-0.95}" "${TOP_K:-64}"
  printf "   image   : %s\n"                 "$img"
  printf "   MTP     : %s\n"                 "$([ "${MTP:-}" = 1 ] && echo "on (n-max ${NMAX:-2})" || echo off)"
  echo "─────────────────────────────────────────────────────────"
  _menu_yesno ">> Launch with these settings? [Y/n]  (q cancels) " yes
  [ "$MENU_REPLY" = yes ] || _menu_cancel
  # The confirmed menu pick is an explicit choice — remember the KV quant so
  # later bare launches (no KVQUANT=) land on the same KV universe.
  tune_set "$(tune_kv_key "$BE")" "${KVQUANT:-f16}"
  echo
}

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
MTP="${MTP:-}"                     # MTP self-speculative decoding: 1|y|on enables it (default off)
NMAX="${NMAX:-2}"                  # MTP draft tokens per step (only used when MTP is on)

# Split args: server flags (--image) go to run-server.sh, --menu is consumed
# here (interactive setup), the rest go to pi. Without this split, --image/--menu
# would be sent to pi and silently ignored.
MENU=0
SERVER_ARGS=()
PI_ARGS=()
for a in "$@"; do
  case "$a" in
    --image) SERVER_ARGS+=("$a") ;;
    --menu)  MENU=1 ;;
    *) PI_ARGS+=("$a") ;;
  esac
done

# First-run turnkey step: offer to link this repo's pi-extensions into pi before
# we launch. Self-guards (already linked / declined / non-interactive), so it's a
# no-op on every launch after the first. See maybe_bootstrap_extensions above.
maybe_bootstrap_extensions

# --- interactive setup (--menu): only meaningful for a fresh server ----------
# A reused server can't be reconfigured, and the wizard needs a real terminal.
if [ "$MENU" = 1 ]; then
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo ">> NOTE: a server is already running — interactive setup only affects a"
    echo "         fresh server, so it's skipped. Reconfigure by restarting it:"
    echo "         bash scripts/stop-server.sh && bash scripts/start.sh --menu"
  elif [ ! -t 0 ]; then
    echo "ERROR: --menu needs an interactive terminal."; exit 1
  else
    configure_menu
  fi
fi

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
  # KV-quant resolution: an explicit KVQUANT= (env or --menu) wins; otherwise
  # reuse the KV quant picked in a previous sweep/--menu (chosen-kv in the
  # tuning cache); else default f16. KV_SOURCE also decides whether the sweep
  # below explores the KV dimension (default → yes; env/cache → pinned, except
  # AUTOTUNE=1 which re-explores everything not pinned by env).
  KV_SOURCE=default                # default | env | cache
  [ -n "${KVQUANT:-}" ] && KV_SOURCE=env
  if [ "$KV_SOURCE" = default ]; then
    _saved_kv="$(tune_get "$(tune_kv_key "$BE")")"
    if [ -n "$_saved_kv" ]; then
      export KVQUANT="$_saved_kv"; KV_SOURCE=cache
      echo ">> auto-tune: reusing saved KV quant KVQUANT=$KVQUANT (re-explore: AUTOTUNE=1 bash scripts/start.sh)"
    fi
  fi
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
    # --menu's auto-tune choice overrides an earlier 'declined' (the user is
    # actively asking to tune now); a numeric/nofit result is still respected.
    [ "${TUNE_YES:-}" = 1 ] && [ "$saved" = "declined" ] && saved=""
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
      if [ "${TUNE_YES:-}" = 1 ]; then
        ans=y                              # --menu auto-tune choice: don't re-ask
      else
        read -r -p ">> Run auto-tuning now? [y/N] " ans || ans=""
      fi
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        echo ">> measuring (this is the slow part — once) ..."
        CTX="$CTX" BACKEND="$BE" KVQUANT="$KV" NCMOE_LIST="$nlist" PORT="$bench_port" \
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
    # --menu's auto-tune choice overrides an earlier 'declined' (see pinned branch).
    [ "${TUNE_YES:-}" = 1 ] && [ "$chosen" = "declined" ] && chosen=""
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
      # KV grid: explicit env (or --menu) pins it; a cache-saved pick pins it too
      # unless AUTOTUNE=1 asked for a full re-exploration; otherwise sweep KV as
      # a third dimension so the pick below chooses context AND KV quant.
      if [ "$KV_SOURCE" = env ] || { [ "$KV_SOURCE" = cache ] && [ "${AUTOTUNE:-}" != 1 ]; }; then
        kvlist="$KV"
      else
        kvlist="${KVQUANT_LIST:-f16,q8_0}"
      fi
      nprobes=$(( $(awk -F, '{print NF}' <<< "$clist") * $(awk -F, '{print NF}' <<< "$nlist") * $(awk -F, '{print NF}' <<< "$kvlist") ))
      echo
      echo ">> No context tuned yet for BACKEND=$BE. Auto-tuning can sweep several context"
      echo "   sizes ($clist) × expert splits ($nlist) × KV quants ($kvlist) and show the"
      echo "   fastest that fits at each, so you can pick how much context to run (and under"
      echo "   which KV quant). It launches the server up to $nprobes time(s)"
      echo "   — roughly ${nprobes}–$(( nprobes + nprobes/2 )) min, measured once and remembered."
      if [ "${TUNE_YES:-}" = 1 ]; then
        ans=y                              # --menu auto-tune choice: don't re-ask
      else
        read -r -p ">> Run the context sweep now? [y/N] " ans || ans=""
      fi
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        echo ">> sweeping (this is the slow part — once) ..."
        # CTX=/KVQUANT= (empty) so benchmark-config.sh sweeps the LISTs instead of pinning.
        CTX= BACKEND="$BE" KVQUANT= KVQUANT_LIST="$kvlist" CTX_LIST="$clist" NCMOE_LIST="$nlist" PORT="$bench_port" \
          bash "$REPO_ROOT/scripts/benchmark-config.sh" || true
        # Build the menu from what fit (cache holds backend:ctx:kv=NCMOE per swept combo).
        opts=()                                  # entries: "ctx:ncmoe:kv"
        IFS=',' read -r -a _cl <<< "$clist"
        IFS=',' read -r -a _kl <<< "$kvlist"
        for c in "${_cl[@]}"; do
          c="$(printf '%s' "$c" | tr -d ' ')"
          for k in "${_kl[@]}"; do
            k="$(printf '%s' "$k" | tr -d ' ')"
            v="$(tune_get "$(tune_key "$BE" "$c" "$k")")"
            [[ "$v" =~ ^[0-9]+$ ]] && opts+=("$c:$v:$k")
          done
        done
        if [ "${#opts[@]}" -eq 0 ]; then
          echo ">> auto-tune: nothing fit on this GPU — launching with the default expert split."
        else
          echo
          echo ">> Which config to launch with? (tok/s for each is in the table above)"
          i=1
          for o in "${opts[@]}"; do
            IFS=':' read -r _c _n _k <<< "$o"
            printf "     [%d] CTX=%-7s KV=%-5s NCMOE=%s\n" "$i" "$_c" "$_k" "$_n"; i=$((i+1))
          done
          dflt="${#opts[@]}"   # default = the largest context that fit (last entry)
          read -r -p ">> choice [${dflt}=largest, q cancels]: " pick || pick=""
          case "$pick" in q|Q|quit|cancel) echo ">> cancelled — nothing started."; exit 0 ;; esac
          [[ "$pick" =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -le "${#opts[@]}" ] || pick="$dflt"
          IFS=':' read -r _c _n _k <<< "${opts[$((pick-1))]}"
          export CTX="$_c" NCMOE="$_n" KVQUANT="$_k"
          KV="$_k"
          tune_set "$(tune_key "$BE" chosen "$KV")" "$CTX"
          tune_set "$(tune_kv_key "$BE")" "$KV"    # remember the KV quant itself too
          echo ">> auto-tune: launching at CTX=$CTX NCMOE=$NCMOE KV=$KV (re-explore: AUTOTUNE=1 bash scripts/start.sh)"
          # --menu auto-syncs pi below; only nudge the env-var path here.
          [ "$MENU" = 1 ] || echo ">> NOTE: keep pi in sync with this context:  CTX=$CTX bash scripts/configure-pi.sh"
        fi
      else
        tune_set "$(tune_key "$BE" chosen "$KV")" declined
        echo ">> skipped — launching at the default CTX=$CTX. Re-enable later: AUTOTUNE=1 bash scripts/start.sh"
      fi
    fi
  fi

  # Keep pi's client context window in lockstep with the server's -c when the
  # user came through --menu. The server can serve 128K, but pi silently caps the
  # context at its own configured contextWindow, so without this you get a 128K
  # server and a 32K client. The menu is an explicit interactive opt-in, so we
  # sync it automatically (it edits pi's ~/.pi/agent/models.json). Non-menu
  # env-var launches are left untouched — there we only remind (see the sweep path).
  if [ "$MENU" = 1 ]; then
    echo ">> syncing pi's context window to $CTX (edits ~/.pi/agent/models.json) ..."
    if CTX="$CTX" bash "$REPO_ROOT/scripts/configure-pi.sh" >/dev/null 2>&1; then
      echo ">> pi context window set to $CTX."
    else
      echo ">> WARNING: could not sync pi automatically. Run it yourself:"
      echo "            CTX=$CTX bash scripts/configure-pi.sh"
    fi
  fi

  # MTP self-speculative decoding (optional). Built HERE — after auto-tune — so the
  # tuning probes measure the base config; the 0.25 GB draft head fits at the same
  # NCMOE. p_min is pinned to 0 (any p_min>0 degenerates the output). Appended to
  # EXTRA_ARGS, which run-server.sh reads from the env we export to it.
  case "${MTP:-}" in
    1|y|Y|yes|on)
      _mtp_head="$REPO_ROOT/models/gemma4-26b-a4b-qat/mtp-gemma-4-26B-A4B-it.gguf"
      if [ -f "$_mtp_head" ]; then
        export EXTRA_ARGS="${EXTRA_ARGS:-} --spec-type draft-mtp --spec-draft-model $_mtp_head --spec-draft-n-max ${NMAX:-2} --spec-draft-p-min 0"
        echo ">> MTP: enabled (n-max ${NMAX:-2}, p-min 0) — speeds up low-temp/coding; ~no gain at temp 1.0"
      else
        echo ">> WARNING: MTP requested but draft head missing ($_mtp_head) — launching WITHOUT MTP."
      fi ;;
  esac

  echo ">> starting server in background (logs: $SERVER_LOG) ..."
  # Launch in its OWN process group (setsid) so an abort can kill the whole tree.
  # run-server.sh exec's `mamba run … llama-server`, so llama-server is a grandchild:
  # killing only $SRV_PID would orphan it, and stop-server.sh can't help mid-load
  # (the port isn't bound until the model finishes loading).
  SRV_PGRP=0
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$REPO_ROOT/scripts/run-server.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SRV_PGRP=1
  else
    nohup bash "$REPO_ROOT/scripts/run-server.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
  fi
  SRV_PID=$!

  # If the user aborts (Ctrl-C) before pi comes up, stop the server we just started
  # rather than leaving a half-loaded model orphaned in the background eating RAM.
  # We own it (we launched it this run), so cleaning it up is the right default.
  _abort_during_load() {
    printf '\n>> aborted while the model was loading — stopping the server we started.\n'
    if kill -0 "$SRV_PID" 2>/dev/null; then
      if [ "$SRV_PGRP" = 1 ]; then
        kill -TERM -- -"$SRV_PID" 2>/dev/null || true   # whole process group
        sleep 1
        kill -KILL -- -"$SRV_PID" 2>/dev/null || true
      else
        # No group isolation: kill the wrapper's child (llama-server) then the wrapper,
        # plus a by-port sweep in case it had already bound the port.
        pkill -TERM -P "$SRV_PID" 2>/dev/null || true
        kill -TERM "$SRV_PID" 2>/dev/null || true
        bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1 || true
      fi
      echo "   server stopped."
    fi
    exit 130
  }
  trap _abort_during_load INT TERM

  # Live progress (elapsed seconds + the latest server log line) instead of a blank
  # wall of dots, so a multi-minute first load visibly advances rather than looking
  # hung. Falls back to plain dots when stdout isn't a terminal (e.g. piped/-p runs).
  echo ">> waiting for the model to load — first load can take a few minutes (Ctrl-C aborts)"
  _t0=$SECONDS; _tty=0; [ -t 1 ] && _tty=1
  for _ in $(seq 1 150); do          # up to ~5 min
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      [ "$_tty" = 1 ] && printf '\r\033[K'
      echo ">> model ready in $((SECONDS - _t0))s."
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      [ "$_tty" = 1 ] && printf '\r\033[K'
      echo "ERROR: the server exited while starting. Last log lines:"
      tail -20 "$SERVER_LOG"
      server_start_hints
      exit 1
    fi
    if [ "$_tty" = 1 ]; then
      _last="$(grep -av '^[[:space:]]*$' "$SERVER_LOG" 2>/dev/null | tail -1 || true)"
      printf '\r\033[K>> loading… %3ds  %s' "$((SECONDS - _t0))" "$(printf '%.68s' "$_last")"
    else
      printf '.'
    fi
    sleep 2
  done
  trap - INT TERM
  if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    [ "$_tty" = 1 ] || echo
    echo "ERROR: the server did not become healthy within ~5 min. Last log lines:"
    tail -20 "$SERVER_LOG"
    server_start_hints
    exit 1
  fi
  echo ">> server up at http://$HOST:$PORT  (PID $SRV_PID · backend $(resolve_backend) · CTX=$CTX). Stop later: bash scripts/stop-server.sh"
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

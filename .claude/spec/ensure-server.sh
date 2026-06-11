#!/usr/bin/env bash
# ensure-server.sh — idempotently bring up the OPTIMAL Gemma 4 llama.cpp server,
# non-blocking, single-flight. Called fire-and-forget by /gemma-draft (draft.sh) so a
# first invocation in a fresh session warms the server; the next one gets a draft.
#
# Design constraints (see docs/PRD-speculative-agent.md):
#   * NEVER block the user: caller-mode returns instantly; the slow model load runs in a
#     detached (setsid) worker. The triggering invocation just degrades (no draft) this turn.
#   * Single-flight: a lock dir stops every invocation from launching a second server.
#   * Reuse, don't reinvent "optimal": backend via resolve_backend, CTX/NCMOE from the
#     auto-tune cache (.gemma4-tuning) via _tuning.sh — exactly what start.sh would pick.
#   * Launch with --image when the mmproj exists, so /gemma-draft image reads work.
#
# Env:
#   SPEC_AUTOSTART=0          disable auto-launch entirely (kill switch; default on)
#   SPEC_AUTOSTART_DRYRUN=1   print the resolved launch command, do not launch
#   SPEC_SERVER_LOG=PATH      server log (default /tmp/gemma4-server.log — same as start.sh)
#   SPEC_AUTOSTART_WAIT=N     health-poll iterations (×2s) the worker waits (default 180)
#   CTX/NCMOE/KVQUANT/BACKEND override the tuned pick (same vars run-server.sh reads)
set -uo pipefail
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SELF")/lib.sh"

REPO_ROOT="$(cd -- "$(dirname -- "$SELF")/../.." && pwd -P)"
export REPO_ROOT
LOCK="$CACHE_DIR/.autostart.lock"
LOG="${SPEC_SERVER_LOG:-/tmp/gemma4-server.log}"
MMPROJ="${MMPROJ:-$REPO_ROOT/models/gemma4-26b-a4b-qat/mmproj-BF16.gguf}"

# Make `mamba` reachable even from a stripped hook environment (run-server needs it).
ensure_mamba() {
  command -v mamba >/dev/null 2>&1 && return 0
  local d
  for d in "$HOME/miniforge3/condabin" "$HOME/miniforge3/bin" \
           "$HOME/mambaforge/condabin"  "$HOME/mambaforge/bin" \
           "$HOME/miniconda3/condabin"  "$HOME/miniconda3/bin"; do
    [ -x "$d/mamba" ] && { PATH="$d:$PATH"; export PATH; return 0; }
  done
  return 1
}

# Resolve the optimal config the way start.sh's non-interactive path would, into
# globals CTX/NCMOE/KVQUANT/SERVER_ARGS + a human summary in $RESOLVED.
resolve_config() {
  source "$REPO_ROOT/scripts/_banner.sh"   # resolve_backend
  source "$REPO_ROOT/scripts/_tuning.sh"   # tune_get / tune_key
  local BE KV chosen saved
  BE="$(resolve_backend)"
  KV="${KVQUANT:-f16}"
  if [ -z "${CTX:-}" ]; then
    chosen="$(tune_get "$(tune_key "$BE" chosen "$KV")")"
    [[ "$chosen" =~ ^[0-9]+$ ]] && CTX="$chosen" || CTX=32768
  fi
  if [ -z "${NCMOE:-}" ]; then
    saved="$(tune_get "$(tune_key "$BE" "$CTX" "$KV")")"
    [[ "$saved" =~ ^[0-9]+$ ]] && NCMOE="$saved"
  fi
  export CTX KVQUANT
  [ -n "${NCMOE:-}" ] && export NCMOE
  SERVER_ARGS=()
  [ -f "$MMPROJ" ] && SERVER_ARGS+=(--image)   # enable image offload (G6) when possible
  RESOLVED="BACKEND=$BE CTX=$CTX NCMOE=${NCMOE:-all-on-cpu} KV=$KV ${SERVER_ARGS[*]:-(text-only)}"
}

# ---------------------------------------------------------------------------
# Worker: holds the lock, launches the server detached, waits until healthy.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--worker" ]; then
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
  if ! ensure_mamba; then
    echo "[ensure-server $(date -u +%H:%M:%SZ)] mamba not found on PATH; cannot launch. Start manually: scripts/start.sh" >> "$LOG"
    exit 0
  fi
  resolve_config
  # Truncate the log for this fresh launch (matches start.sh) so stale lines from a
  # previous server don't masquerade as current errors.
  { echo "[ensure-server $(date -u +%H:%M:%SZ)] auto-launching: $RESOLVED"; } > "$LOG"
  setsid bash "$REPO_ROOT/scripts/run-server.sh" "${SERVER_ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
  srv=$!
  spec_log "$(jq -cn --arg r "$RESOLVED" '{event:"autostart",resolved:$r}')"
  for _ in $(seq 1 "${SPEC_AUTOSTART_WAIT:-180}"); do
    spec_server_up && break
    kill -0 "$srv" 2>/dev/null || break      # server died while loading
    sleep 2
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Hook mode: cheap checks, then hand off to a detached worker and return now.
# ---------------------------------------------------------------------------
[ "${SPEC_AUTOSTART:-1}" = "1" ] || exit 0
spec_server_up && exit 0                       # already up — nothing to do

if [ "${SPEC_AUTOSTART_DRYRUN:-0}" = "1" ]; then
  ensure_mamba || echo "(note: mamba not on PATH — would fail to launch)"
  resolve_config
  echo "[dry-run] would launch: bash scripts/run-server.sh ${SERVER_ARGS[*]}"
  echo "[dry-run] resolved: $RESOLVED"
  echo "[dry-run] log: $LOG"
  exit 0
fi

# Single-flight: only one launcher at a time. Steal a stale lock (>10 min).
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -d "$LOCK" ] && find "$LOCK" -maxdepth 0 -mmin +10 >/dev/null 2>&1; then
    rmdir "$LOCK" 2>/dev/null || true
    mkdir "$LOCK" 2>/dev/null || exit 0
  else
    exit 0                                     # a launch is already in progress
  fi
fi

setsid bash "$SELF" --worker >/dev/null 2>&1 < /dev/null &
exit 0

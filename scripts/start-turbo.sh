#!/usr/bin/env bash
#
# start-turbo.sh — zero-config preset: the exact Twitter/@analogalok TurboQuant recipe
# for the Gemma 4 12B v2 "agentic" model, then drop into pi. Nothing to remember — just
# run it. Thin wrapper over scripts/start-agentic.sh with the tweet's settings baked in.
#
# Bakes in (== the tweet's server command):
#   TURBO=1        TheTom/llama-cpp-turboquant build + turbo3 V cache, full offload -ngl 99
#   QUANT=Q4_K_M   the recommended quant
#   CTX=25000      the tweet's context (verified to fit our 8 GB RTX 2070 — tight but works)
#   K=q8_0 V=turbo3, port 8080, temp 1.0 / top-p 0.95 / top-k 64   (start-agentic/TURBO defaults)
#     -> llama-server -m gemma4-v2-Q4_K_M.gguf -ngl 99 -c 25000 \
#          --cache-type-k q8_0 --cache-type-v turbo3 --port 8080
# Then launches pi (context window auto-synced) and offers to stop the server on exit.
#
# Just run it:
#   bash scripts/start-turbo.sh                       # server + interactive pi
#   bash scripts/start-turbo.sh -p "explain @README.md"   # one-shot (leaves server up)
#
# Every baked-in value is still overridable from the env if you want to deviate, e.g.:
#   CTX=16384 bash scripts/start-turbo.sh             # roomier VRAM headroom
#   QUANT=Q3_K_M bash scripts/start-turbo.sh          # smaller quant
#   STOP_ON_EXIT=1 bash scripts/start-turbo.sh        # always stop the server when pi exits
# Needs the turboquant build (vendor/llama-cpp-turboquant) — start-agentic.sh prints how
# to build it if it's missing. See docs/gemma4-12b-agentic-eval.md.
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The Twitter preset — exported so start-agentic.sh / run-12b-agentic.sh pick them up.
# Each uses ${VAR:-default} so you CAN override from the env, but never NEED to.
export TURBO="${TURBO:-1}"
export QUANT="${QUANT:-Q4_K_M}"
export CTX="${CTX:-25000}"
export PORT="${PORT:-8080}"

exec bash "$REPO_ROOT/scripts/start-agentic.sh" "$@"

#!/usr/bin/env bash
#
# stop-server.sh — stop the local llama.cpp server by the port it listens on.
#
#   PORT   port the server binds (default: 8080)
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

get_pids_on_port() {
  local port="$1"
  local pids=""
  if command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
  fi
  if [ -z "$pids" ] && command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true)"
  fi
  echo "$pids"
}

stop_pids() {
  local pids="$1"
  local port="$2"
  if [ -z "$pids" ]; then
    return 0
  fi
  echo ">> stopping server PID(s) on port $port: $pids"
  for p in $pids; do kill "$p" 2>/dev/null || true; done
  sleep 2
  # escalate if still alive
  for p in $pids; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true; done
  echo ">> stopped."
}

# Check if PORT is set in the environment or passed explicitly
ENV_PORT="${PORT:-}"

if [ -n "$ENV_PORT" ]; then
  # Port was pinned explicitly via environment, bypass interactive choice
  pids="$(get_pids_on_port "$ENV_PORT")"
  if [ -z "$pids" ]; then
    echo ">> no server listening on port $ENV_PORT"
  else
    stop_pids "$pids" "$ENV_PORT"
  fi
  exit 0
fi

# Detect active servers
std_pids="$(get_pids_on_port 8080)"
diff_pids="$(get_pids_on_port 8082)"

# Check if non-interactive
if [ ! -t 0 ]; then
  # Non-interactive fallback: stop whatever is running
  if [ -n "$std_pids" ]; then
    stop_pids "$std_pids" 8080
  fi
  if [ -n "$diff_pids" ]; then
    stop_pids "$diff_pids" 8082
    pkill -f "llama-diffusion-cli" || true
  fi
  if [ -z "$std_pids" ] && [ -z "$diff_pids" ]; then
    echo ">> no standard (8080) or diffusion (8082) servers are currently running."
  fi
  exit 0
fi

# Interactive menu
echo ">> Active server check:"
if [ -n "$std_pids" ]; then
  echo "   [Active] Standard server on port 8080"
else
  echo "   [Off]    Standard server on port 8080"
fi

if [ -n "$diff_pids" ]; then
  echo "   [Active] Diffusion server on port 8082"
else
  echo "   [Off]    Diffusion server on port 8082"
fi
echo

# If nothing is running, abort early
if [ -z "$std_pids" ] && [ -z "$diff_pids" ]; then
  echo ">> No servers are currently running on port 8080 or 8082."
  exit 0
fi

echo "Which server do you want to stop?"
echo "  1) Standard server (port 8080)"
echo "  2) Diffusion server (port 8082)"
echo "  3) Both"
echo "  4) Cancel"
read -r -p "choice [1-4, default 4]: " choice || choice=""
case "$choice" in
  1)
    if [ -n "$std_pids" ]; then
      stop_pids "$std_pids" 8080
    else
      echo ">> Standard server was not running."
    fi
    ;;
  2)
    if [ -n "$diff_pids" ]; then
      stop_pids "$diff_pids" 8082
    else
      echo ">> Diffusion server was not running."
    fi
    pkill -f "llama-diffusion-cli" || true
    ;;
  3)
    [ -n "$std_pids" ] && stop_pids "$std_pids" 8080
    [ -n "$diff_pids" ] && stop_pids "$diff_pids" 8082
    pkill -f "llama-diffusion-cli" || true
    ;;
  *)
    echo ">> Cancelled."
    ;;
esac

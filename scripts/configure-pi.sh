#!/usr/bin/env bash
#
# configure-pi.sh — register the local llama.cpp server as a provider in pi.
#
# Idempotently adds (or replaces) a "llamacpp" provider in pi's models.json,
# using the definition in config/pi-provider.json. Safe to re-run.
#
#   PI_MODELS   path to pi's models.json   (default: ~/.pi/agent/models.json)
#   CTX         override the model's contextWindow to match the server's -c
#               (default: keep the value in config/pi-provider.json, 32768).
#               Use the SAME value you pass as CTX to run-server.sh/start.sh.
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_MODELS="${PI_MODELS:-$HOME/.pi/agent/models.json}"
PROVIDER_JSON="$REPO_ROOT/config/pi-provider.json"
CTX="${CTX:-}"
PROVIDER="${PROVIDER:-llamacpp}"

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required."; exit 1; }
[ -f "$PROVIDER_JSON" ] || { echo "ERROR: missing $PROVIDER_JSON"; exit 1; }

mkdir -p "$(dirname "$PI_MODELS")"

python3 - "$PI_MODELS" "$PROVIDER_JSON" "$CTX" "$PROVIDER" <<'PY'
import json, os, sys
models_path, provider_path, ctx, provider_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

data = {"providers": {}}
if os.path.exists(models_path):
    try:
        with open(models_path) as f:
            data = json.load(f)
    except Exception:
        pass

if provider_name == "llamacpp":
    with open(provider_path) as f:
        provider = json.load(f)
    if ctx:
        for m in provider.get("models", []):
            m["contextWindow"] = int(ctx)
    data.setdefault("providers", {})["llamacpp"] = provider
    cw = provider["models"][0].get("contextWindow")
else:
    cw = None
    if ctx and data.get("providers", {}).get(provider_name):
        for m in data["providers"][provider_name].get("models", []):
            m["contextWindow"] = int(ctx)
            cw = m["contextWindow"]

with open(models_path, "w") as f:
    json.dump(data, f, indent=2)

if cw:
    print(f">> updated provider '{provider_name}' (contextWindow={cw}) -> {models_path}")
else:
    print(f">> wrote models.json -> {models_path}")
PY

cat <<EOF

>> Done. Start the server then run pi with --provider $PROVIDER.
EOF

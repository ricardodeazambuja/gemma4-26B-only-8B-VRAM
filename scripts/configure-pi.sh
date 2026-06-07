#!/usr/bin/env bash
#
# configure-pi.sh — register the local llama.cpp server as a provider in pi.
#
# Idempotently adds (or replaces) a "llamacpp" provider in pi's models.json,
# using the definition in config/pi-provider.json. Safe to re-run.
#
#   PI_MODELS   path to pi's models.json   (default: ~/.pi/agent/models.json)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_MODELS="${PI_MODELS:-$HOME/.pi/agent/models.json}"
PROVIDER_JSON="$REPO_ROOT/config/pi-provider.json"

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required."; exit 1; }
[ -f "$PROVIDER_JSON" ] || { echo "ERROR: missing $PROVIDER_JSON"; exit 1; }

mkdir -p "$(dirname "$PI_MODELS")"

python3 - "$PI_MODELS" "$PROVIDER_JSON" <<'PY'
import json, os, sys
models_path, provider_path = sys.argv[1], sys.argv[2]
with open(provider_path) as f:
    provider = json.load(f)
data = {"providers": {}}
if os.path.exists(models_path):
    try:
        with open(models_path) as f:
            data = json.load(f)
    except Exception:
        pass
data.setdefault("providers", {})["llamacpp"] = provider
with open(models_path, "w") as f:
    json.dump(data, f, indent=2)
print(f">> wrote provider 'llamacpp' -> {models_path}")
PY

cat <<EOF

>> Done. Start the server (scripts/run-server.sh) then run:
     pi --provider llamacpp --model gemma-4-26b-a4b-qat
EOF

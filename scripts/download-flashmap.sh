#!/usr/bin/env bash
# Download everything the FlashMap (flow-map speculative decoding) experiment needs:
#   models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf          target, served by llama-server
#   models/qwen3-8b/hf/                            target config + tokenizer + embed_tokens only
#                                                  (the drafter converter needs the tokenizer and
#                                                  bakes constants from the embedding matrix)
#   models/qwen3-8b/flashmap-hf/                   j-shen/FlashMap-Qwen3-8B drafter checkpoint
#
# Uses curl -4 (IPv6 to Hugging Face stalls on this network) and verifies sha256 of every
# LFS file against the Hub's lfs.oid. Re-running resumes partial downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/models/qwen3-8b"
HF="https://huggingface.co"
mkdir -p "$DIR/hf" "$DIR/flashmap-hf"

fetch() { # repo file dest [sha256]
  local repo="$1" file="$2" dest="$3" sha="${4:-}"
  if [ -n "$sha" ] && [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then
    echo "ok (cached): $dest"; return
  fi
  echo "downloading $repo/$file"
  curl -4 -fL --retry 5 -C - -o "$dest" "$HF/$repo/resolve/main/$file"
  if [ -n "$sha" ]; then
    echo "$sha  $dest" | sha256sum -c - || { echo "ERROR: sha256 mismatch for $dest"; exit 1; }
  fi
}

fetch Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf "$DIR/Qwen3-8B-Q4_K_M.gguf" \
  d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785

for f in config.json configuration_flashmap.py modeling_flashmap.py README.md; do
  fetch j-shen/FlashMap-Qwen3-8B "$f" "$DIR/flashmap-hf/$f"
done
fetch j-shen/FlashMap-Qwen3-8B model.safetensors "$DIR/flashmap-hf/model.safetensors" \
  19da1d6b1946648ff4b82f1831acfdd3f6f3b14620e6638b65efc904cac7c564

for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  fetch Qwen/Qwen3-8B "$f" "$DIR/hf/$f"
done

# embed_tokens only (1.2 GB of the 4 GB first shard): HTTP range reads of the safetensors
# header and of the one tensor, rewritten as a single-tensor safetensors file
EMB="$DIR/hf/embed_tokens.safetensors"
if [ ! -f "$EMB" ]; then
  python3 -I - "$HF/Qwen/Qwen3-8B/resolve/main/model-00001-of-00005.safetensors" "$EMB" <<'EOF'
import json, struct, subprocess, sys
url, out = sys.argv[1], sys.argv[2]
def get(a, b):  # inclusive byte range
    return subprocess.run(["curl", "-4", "-fsSL", "--retry", "5", "-r", f"{a}-{b}", url],
                          check=True, capture_output=True).stdout
n = struct.unpack("<Q", get(0, 7))[0]
header = json.loads(get(8, 8 + n - 1))
name = "model.embed_tokens.weight"
t = header[name]
a, b = t["data_offsets"]
print(f"{name}: {t['dtype']} {t['shape']} ({(b - a) / 2**30:.2f} GiB)")
data = get(8 + n + a, 8 + n + b - 1)
assert len(data) == b - a, (len(data), b - a)
h = json.dumps({name: {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [0, b - a]}}).encode()
h += b" " * (-len(h) % 8)
with open(out + ".part", "wb") as f:
    f.write(struct.pack("<Q", len(h)) + h + data)
import os; os.replace(out + ".part", out)
EOF
fi
echo "done: $DIR"

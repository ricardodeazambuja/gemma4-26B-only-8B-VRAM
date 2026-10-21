#!/usr/bin/env python3
"""Convert a FlashMap drafter (https://github.com/al-jshen/flashmap) to a GGUF for the
buun-llama.cpp fork patched with patches/buun-llama.cpp-flashmap.patch.

FlashMap is written as a variant of the fork's `dflash` arch (DSpark anchor-first block
contract, target hidden states injected into the drafter KV cache) and is detected at load
time by `dflash.flashmap.target_hidden_size`. Only the one-step draft (num_steps=1, the
upstream default) is exported, so everything that depends on the jump times alone is baked
here instead of evaluated per draft:

  * the AdaLN modulations of every norm for the jump s=0 -> t=1 (mod = 1 + scale, shift, gate)
  * the embedded uniform posterior through state_norm      (flashmap.state_const)
  * noise_norm.weight x noise_gate x (t - s)               (flashmap.noise_norm)
  * L with L L^T = E^T E, E = the target's embedding       (flashmap.noise_factor)

Usage:
  python scripts/convert-flashmap.py DRAFT_DIR TARGET_EMBED.safetensors TARGET.gguf OUT.gguf [--outtype f16|q8_0]
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "buun-llama.cpp" / "gguf-py"))
import gguf
from gguf.quants import quantize

ARCH = "dflash"
NUM_STEPS = 1


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """All tensors of a safetensors file as float32 (BF16/F16/F32 only)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, t in header.items():
            if name == "__metadata__":
                continue
            a, b = t["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if t["dtype"] == "BF16":
                arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif t["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif t["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                raise ValueError(f"{name}: unsupported dtype {t['dtype']}")
            out[name] = arr.reshape(t["shape"])
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def time_embedding(w: dict, prefix: str, t: float, dim: int, scale: float) -> np.ndarray:
    """FlashMapTimeEmbedding: sinusoidal features of t, then Linear -> SiLU -> Linear."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = (t * scale) * freqs
    emb = np.concatenate([np.cos(args), np.sin(args)])
    h = w[f"{prefix}.linear_1.weight"] @ emb + w[f"{prefix}.linear_1.bias"]
    return w[f"{prefix}.linear_2.weight"] @ silu(h) + w[f"{prefix}.linear_2.bias"]


def modulation(w: dict, prefix: str, cond: np.ndarray):
    """FlashMapAdaLayerNorm.modulation -> (scale, shift, gate)."""
    m = w[f"{prefix}.linear.weight"] @ silu(cond) + w[f"{prefix}.linear.bias"]
    return np.split(m, 3)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps).astype(np.float32) * weight


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("draft_dir", type=Path)
    ap.add_argument("target_embed", type=Path, help="safetensors holding model.embed_tokens.weight of the target")
    ap.add_argument("target_gguf", type=Path, help="target GGUF to copy the tokenizer from")
    ap.add_argument("out", type=Path)
    ap.add_argument("--outtype", choices=["f16", "q8_0"], default="f16",
                    help="type of the matmul weights (norms, biases and constants stay f32)")
    args = ap.parse_args()

    cfg = json.loads((args.draft_dir / "config.json").read_text())
    assert cfg["model_type"] == "flashmap", cfg["model_type"]
    grid = cfg["jump_grids"][str(NUM_STEPS)]
    s, t = float(grid[0]), float(grid[1])
    jump = 1.0 if cfg.get("noise_gate", "jump") == "constant" else t - s

    w = load_safetensors(args.draft_dir / "model.safetensors")
    E = load_safetensors(args.target_embed)["model.embed_tokens.weight"]
    n_vocab, n_embd_tgt = E.shape
    assert n_embd_tgt == cfg["target_hidden_size"], (E.shape, cfg["target_hidden_size"])
    n_embd = cfg["hidden_size"]
    n_layer = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    head_dim = n_embd // n_head
    rms_eps = float(cfg["rms_norm_eps"])

    # --- constants of the one-step jump -----------------------------------------------
    cond = time_embedding(w, "time_embed", s, cfg["time_embed_dim"], cfg["time_scale"]) \
         + time_embedding(w, "jump_embed", t, cfg["time_embed_dim"], cfg["time_scale"])

    # E^T E in float32 by row blocks (as upstream), its Cholesky factor in float64
    print(f"target embedding {E.shape}: computing E^T E and its Cholesky factor ...", flush=True)
    gram = np.zeros((n_embd_tgt, n_embd_tgt), dtype=np.float32)
    for i in range(0, n_vocab, 16384):
        rows = E[i:i + 16384]
        gram += rows.T @ rows
    L = np.linalg.cholesky(gram.astype(np.float64)).astype(np.float32)
    uniform_embeds = E.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_const = rms_norm(uniform_embeds, w["state_norm.weight"], rms_eps)
    noise_norm = w["noise_norm.weight"] * (w["noise_gate"].reshape(()) * jump)
    del E, gram

    tensors: list[tuple[str, np.ndarray, bool]] = []  # (name, data, is_matmul_weight)

    def add(name, data, mm=False):
        tensors.append((name, np.ascontiguousarray(data, dtype=np.float32), mm))

    add("fc.weight", w["fc.weight"], True)
    add("enc.output_norm.weight", w["hidden_norm.weight"])
    add("flashmap.anchor_proj.weight", w["anchor_proj.weight"], True)
    add("flashmap.anchor_proj.bias", w["anchor_proj.bias"])
    add("flashmap.anchor_norm.weight", w["anchor_norm.weight"])
    add("flashmap.state_const", state_const)
    add("flashmap.noise_norm.weight", noise_norm)
    add("flashmap.noise_factor", L, True)
    add("flashmap.x_embed.weight", w["x_embed.weight"], True)
    add("flashmap.x_embed.bias", w["x_embed.bias"])
    scale, shift, _ = modulation(w, "final_adaln", cond)
    add("flashmap.final_mod.weight", 1.0 + scale)
    add("flashmap.final_mod.bias", shift)
    add("flashmap.head.weight", w["head.weight"], True)
    add("flashmap.head.bias", w["head.bias"])

    for i in range(n_layer):
        p, b = f"layers.{i}", f"blk.{i}"
        for src, dst in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            add(f"{b}.{dst}.weight", w[f"{p}.cross_attn.{src}.weight"], True)
        add(f"{b}.attn_q_norm.weight", w[f"{p}.cross_attn.q_norm.weight"])
        add(f"{b}.attn_k_norm.weight", w[f"{p}.cross_attn.k_norm.weight"])
        for src, dst in (("q_proj", "fm_sa_q"), ("k_proj", "fm_sa_k"), ("v_proj", "fm_sa_v"), ("o_proj", "fm_sa_out")):
            add(f"{b}.{dst}.weight", w[f"{p}.self_attn.{src}.weight"], True)
        add(f"{b}.fm_sa_q_norm.weight", w[f"{p}.self_attn.q_norm.weight"])
        add(f"{b}.fm_sa_k_norm.weight", w[f"{p}.self_attn.k_norm.weight"])
        for src, dst in (("self_attn_adaln", "fm_sa"), ("cross_attn_adaln", "fm_ca"), ("mlp_adaln", "fm_mlp")):
            scale, shift, gate = modulation(w, f"{p}.{src}", cond)
            add(f"{b}.{dst}_mod.weight", 1.0 + scale)
            add(f"{b}.{dst}_mod.bias", shift)
            add(f"{b}.{dst}_gate.weight", gate)
        add(f"{b}.ffn_up.weight", w[f"{p}.mlp.fc1.weight"], True)
        add(f"{b}.ffn_up.bias", w[f"{p}.mlp.fc1.bias"])
        add(f"{b}.ffn_down.weight", w[f"{p}.mlp.fc2.weight"], True)
        add(f"{b}.ffn_down.bias", w[f"{p}.mlp.fc2.bias"])

    # --- GGUF -----------------------------------------------------------------------------
    gw = gguf.GGUFWriter(args.out, ARCH)
    gw.add_name(f"FlashMap drafter for {cfg.get('target_model', 'target')}")
    gw.add_context_length(int(cfg.get("max_position_embeddings", 131072)))
    gw.add_embedding_length(n_embd)
    gw.add_feed_forward_length(cfg["intermediate_size"])
    gw.add_block_count(n_layer)
    gw.add_head_count(n_head)
    gw.add_head_count_kv(n_head)
    gw.add_key_length(head_dim)
    gw.add_value_length(head_dim)
    gw.add_layer_norm_rms_eps(rms_eps)
    gw.add_layer_norm_eps(float(cfg["layer_norm_eps"]))
    gw.add_rope_freq_base(float(cfg["rope_theta"]))
    gw.add_rope_dimension_count(head_dim)
    gw.add_block_size(cfg["block_size"])
    # extract ids follow the fork's DFlash convention: HF hidden_states[i + 1] = input of layer i + 1
    gw.add_target_layers([i + 1 for i in cfg["target_layer_ids"]])
    gw.add_sample_from_anchor(True)
    gw.add_has_confidence_head(False)
    gw.add_uint32(f"{ARCH}.dflash.mask_token_id", cfg["dspark_config"]["mask_token_id"])
    gw.add_uint32(f"{ARCH}.flashmap.target_hidden_size", n_embd_tgt)
    gw.add_float32(f"{ARCH}.flashmap.qk_norm_eps", float(cfg["qk_norm_eps"]))
    gw.add_uint32(f"{ARCH}.flashmap.num_steps", NUM_STEPS)

    # tokenizer: copied from the target GGUF
    reader = gguf.GGUFReader(args.target_gguf)
    for field in reader.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
            else:
                val = [field.parts[i].tolist()[0] for i in field.data]
            gw.add_array(field.name, val)
        elif vtype == gguf.GGUFValueType.STRING:
            gw.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            gw.add_key_value(field.name, field.parts[field.data[0]].tolist()[0], vtype)
    del reader

    n_bytes = 0
    for name, data, mm in tensors:
        if mm and args.outtype == "q8_0" and data.shape[-1] % 32 == 0:
            q = quantize(data, gguf.GGMLQuantizationType.Q8_0)
            gw.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            n_bytes += q.nbytes
        elif mm:
            gw.add_tensor(name, data.astype(np.float16))
            n_bytes += data.size * 2
        else:
            gw.add_tensor(name, data)
            n_bytes += data.size * 4
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"wrote {args.out}: {len(tensors)} tensors, {n_bytes / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()

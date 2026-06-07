#!/usr/bin/env python3
"""Inspect a GGUF file: architecture, modality, and tensor inventory.

Works on both the main model GGUF and an `mmproj` projector GGUF. Answers the
question "what is actually inside this file?" — which is what we used to prove
that the Gemma 4 text GGUF has no vision tensors and that `mmproj-BF16.gguf` is
vision-only (see docs/TECHNICAL.md §14).

Usage:
    python utils/inspect-gguf.py <file.gguf> [--tensors]
    bash   utils/inspect-gguf.sh  <file.gguf> [--tensors]   # runs it in the env

`--tensors` also dumps every tensor name (otherwise just counts + samples).

It needs the `gguf` python module. If it isn't installed it falls back to the
copy vendored at vendor/llama.cpp/gguf-py, so no extra install is required as
long as the llama.cpp source is checked out.
"""
import argparse
import os
import sys


def _import_gguf():
    try:
        import gguf  # noqa: F401
        return gguf
    except ModuleNotFoundError:
        here = os.path.dirname(os.path.abspath(__file__))
        vendored = os.path.join(here, "..", "vendor", "llama.cpp", "gguf-py")
        sys.path.insert(0, vendored)
        try:
            import gguf  # noqa: F811
            return gguf
        except ModuleNotFoundError:
            sys.exit(
                "ERROR: the 'gguf' module is not available.\n"
                "  Install it:  pip install gguf\n"
                f"  Or check out llama.cpp source at: {vendored}"
            )


def field_value(field, gguf):
    """Decode a GGUFReader field to a python scalar/list/str using its declared
    GGUF value type (so numbers come out as numbers, not raw bytes)."""
    T = gguf.GGUFValueType
    types = list(getattr(field, "types", []) or [])
    if not types:
        return None

    def decode_one(idx, vtype):
        part = field.parts[idx]
        if vtype == T.STRING:
            return bytes(part).decode("utf-8", "replace")
        v = part.tolist()
        return v[0] if isinstance(v, list) and len(v) == 1 else v

    if types[0] == T.ARRAY:
        elem = types[1] if len(types) > 1 else T.STRING
        return [decode_one(i, elem) for i in field.data]
    return decode_one(field.data[0], types[0])


# tensor-name fragments that mark each modality
VISION_HINTS = ("vision", "v.blk", "patch_embd", "pos_embd", "mm.", "mm_")
AUDIO_HINTS = ("audio", "conformer", "a.blk", "mel", "whisper")


def main():
    ap = argparse.ArgumentParser(
        description="Inspect a GGUF file's architecture, modality and tensors."
    )
    ap.add_argument("path", help="path to the .gguf file")
    ap.add_argument(
        "--tensors", action="store_true", help="dump every tensor name, not just counts"
    )
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        sys.exit(f"ERROR: not a file: {args.path}")

    gguf = _import_gguf()
    r = gguf.GGUFReader(args.path)

    size_mb = os.path.getsize(args.path) / 1e6
    print(f"file         : {os.path.basename(args.path)}  ({size_mb:,.0f} MB)")

    arch = field_value(r.fields["general.architecture"], gguf) if "general.architecture" in r.fields else "?"
    name = field_value(r.fields["general.name"], gguf) if "general.name" in r.fields else None
    print(f"architecture : {arch}" + (f"   name: {name}" if name else ""))

    # metadata keys that signal a vision/audio/projector component
    modal_keys = [
        k for k in r.fields
        if any(t in k.lower() for t in ("vision", "audio", "clip.", "projector", "mmproj"))
    ]
    if modal_keys:
        print("\nmultimodal metadata keys:")
        for k in sorted(modal_keys):
            print(f"  {k} = {field_value(r.fields[k], gguf)}")
    else:
        print("\nmultimodal metadata keys: NONE")

    # tensor inventory
    names = [t.name for t in r.tensors]
    vis = [n for n in names if any(h in n.lower() for h in VISION_HINTS)]
    aud = [n for n in names if any(h in n.lower() for h in AUDIO_HINTS)]

    print(f"\ntensors      : {len(names)} total")
    print(f"  vision-ish : {len(vis)}" + (f"   e.g. {vis[:4]}" if vis else ""))
    print(f"  audio-ish  : {len(aud)}" + (f"   e.g. {aud[:4]}" if aud else ""))

    # verdict
    has_vis = bool(vis) or any("vision" in k.lower() for k in modal_keys)
    has_aud = bool(aud) or any("audio" in k.lower() for k in modal_keys)
    if arch == "clip":
        mods = [m for m, on in (("vision", has_vis), ("audio", has_aud)) if on]
        verdict = "projector (mmproj) — " + ("+".join(mods) if mods else "unknown modality")
    elif has_vis or has_aud:
        mods = ["text"] + [m for m, on in (("vision", has_vis), ("audio", has_aud)) if on]
        verdict = "multimodal model — " + "+".join(mods)
    else:
        verdict = "text-only (no vision/audio tensors — needs a separate --mmproj for images)"
    print(f"\nverdict      : {verdict}")

    if args.tensors:
        print("\nall tensors:")
        for n in names:
            print(f"  {n}")


if __name__ == "__main__":
    main()

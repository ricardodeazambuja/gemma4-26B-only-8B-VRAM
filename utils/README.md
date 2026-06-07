# utils

Small standalone tools that aren't part of the run/serve flow but are handy for
poking at the model files.

## `inspect-gguf.{sh,py}` — what's inside a GGUF?

Reads a GGUF's header and reports its **architecture**, any **multimodal
metadata**, its **tensor inventory**, and a one-line **verdict** (text-only /
multimodal model / `mmproj` projector). This is how we established that the
Gemma 4 text GGUF carries no vision tensors and that `mmproj-BF16.gguf` is
vision-only — see [`docs/TECHNICAL.md` §14](../docs/TECHNICAL.md).

```bash
# the model (text-only):
bash utils/inspect-gguf.sh models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

# the vision projector:
bash utils/inspect-gguf.sh models/gemma4-26b-a4b-qat/mmproj-BF16.gguf

# also dump every tensor name:
bash utils/inspect-gguf.sh <file.gguf> --tensors
```

Example verdicts:

```
verdict      : text-only (no vision/audio tensors — needs a separate --mmproj for images)
verdict      : projector (mmproj) — vision
```

**Notes**
- The `.sh` wrapper runs the `.py` inside the `llamacpp` conda env (`mamba run`)
  so `numpy` + `gguf` are available; it falls back to plain `python3` if mamba
  isn't on PATH. You can also call the `.py` directly in any env that has `gguf`.
- The `gguf` module is auto-resolved: it uses an installed copy, otherwise the
  one vendored at `vendor/llama.cpp/gguf-py` — no extra install needed when the
  llama.cpp source is checked out.

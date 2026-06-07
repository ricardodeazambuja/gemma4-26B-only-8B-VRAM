#!/usr/bin/env python3
"""Generate docs/speed.svg — a horizontal bar chart of token-generation speed
per backend. Pure-stdlib (no matplotlib); re-run after new measurements:

    python3 scripts/make-speed-chart.py
"""
import os

# Measured generation speed (tok/s), server-side, RTX 2070 8GB, NCMOE=22.
BARS = [
    ("CPU only",      1.9,  "#8b949e"),
    ("Vulkan",        4.9,  "#f0883e"),
    ("CUDA (built)",  23.5, "#2ea043"),
]
NOTE = {"CUDA (built)": "~5x faster"}

# --- layout (all geometry computed here) -----------------------------------
W, H        = 720, 272
PAD_L       = 150          # space for row labels
PAD_R       = 95           # space for value labels
PLOT_X0     = PAD_L
PLOT_X1     = W - PAD_R
PLOT_W      = PLOT_X1 - PLOT_X0
TOP         = 64           # below title
BAR_H       = 34
GAP         = 24
SCALE_MAX   = 25.0         # round number above the max value
TICKS       = [0, 5, 10, 15, 20, 25]

def x_of(v):   return PLOT_X0 + (v / SCALE_MAX) * PLOT_W

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
# card background (readable on light & dark GitHub themes)
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#d0d7de"/>')
# title + subtitle
svg.append(f'<text x="24" y="32" font-size="19" font-weight="700" fill="#1f2328">'
           f'Token generation speed by backend</text>')
svg.append(f'<text x="24" y="52" font-size="12.5" fill="#656d76">'
           f'Gemma 4 26B-A4B QAT (UD-Q4_K_XL) &#183; RTX 2070 8 GB &#183; NCMOE=22 &#183; higher is better</text>')

# gridlines + tick labels
base_y = TOP
plot_h = len(BARS) * BAR_H + (len(BARS) - 1) * GAP
for t in TICKS:
    gx = x_of(t)
    svg.append(f'<line x1="{gx:.1f}" y1="{base_y-6}" x2="{gx:.1f}" y2="{base_y+plot_h+6}" '
               f'stroke="#eaeef2" stroke-width="1"/>')
    svg.append(f'<text x="{gx:.1f}" y="{base_y+plot_h+22}" font-size="10.5" fill="#8c959f" '
               f'text-anchor="middle">{t}</text>')
svg.append(f'<text x="{(PLOT_X0+PLOT_X1)/2:.0f}" y="{base_y+plot_h+38}" font-size="11" '
           f'fill="#656d76" text-anchor="middle">tokens / second</text>')

# bars
y = TOP
for label, val, color in BARS:
    bw = max(2.0, x_of(val) - PLOT_X0)
    cy = y + BAR_H / 2
    svg.append(f'<text x="{PLOT_X0-12}" y="{cy+4:.1f}" font-size="13" fill="#1f2328" '
               f'text-anchor="end" font-weight="600">{label}</text>')
    svg.append(f'<rect x="{PLOT_X0}" y="{y}" width="{bw:.1f}" height="{BAR_H}" rx="5" fill="{color}"/>')
    svg.append(f'<text x="{PLOT_X0+bw+10:.1f}" y="{cy+4:.1f}" font-size="13" fill="#1f2328" '
               f'font-weight="700">{val:g} tok/s</text>')
    if label in NOTE:
        svg.append(f'<text x="{PLOT_X0+bw-10:.1f}" y="{cy+4:.1f}" font-size="11.5" fill="#ffffff" '
                   f'text-anchor="end" font-weight="700">{NOTE[label]}</text>')
    y += BAR_H + GAP

svg.append('</svg>')

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "speed.svg")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("\n".join(svg) + "\n")
print("wrote", out)

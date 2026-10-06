"""Two-GPU layout timelines for the README (docs/img/layout_*.svg), drawn to scale from measured chunk times.

  python tools/layout_svgs.py
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
DIT, DEC, TRACK, INK, MUTED = "#3a6ea5", "#8c2f39", "#ececec", "#222222", "#666666"
W, LABEL_W, ROW_H, GAP, T = 760, 150, 26, 8, 3.0  # px width, label column, row height, row gap, seconds shown


def svg(name, rows):
    """rows: [(label, [(start_s, dur_s, kind, text), ...])]"""
    plot = W - LABEL_W - 16
    x = lambda t: LABEL_W + plot * t / T
    h = len(rows) * (ROW_H + GAP) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
           f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">',
           f'<rect width="{W}" height="{h}" fill="#ffffff"/>']
    for i, (label, bars) in enumerate(rows):
        y = 6 + i * (ROW_H + GAP)
        out.append(f'<text x="{LABEL_W - 10}" y="{y + ROW_H / 2 + 4}" text-anchor="end" fill="{INK}">{label}</text>')
        out.append(f'<rect x="{LABEL_W}" y="{y}" width="{plot}" height="{ROW_H}" rx="3" fill="{TRACK}"/>')
        for s, d, kind, text in bars:
            if s >= T:
                continue
            d = min(d, T - s)
            out.append(f'<rect x="{x(s) + 1:.1f}" y="{y}" width="{max(x(s + d) - x(s) - 2, 1):.1f}" height="{ROW_H}" rx="3" '
                       f'fill="{DIT if kind == "dit" else DEC}"/>')
            if x(s + d) - x(s) > 44:
                out.append(f'<text x="{(x(s) + x(s + d)) / 2:.1f}" y="{y + ROW_H / 2 + 4}" text-anchor="middle" '
                           f'fill="#ffffff" font-weight="600">{text}</text>')
    ya = 6 + len(rows) * (ROW_H + GAP) + 10
    for t in (0, 0.5, 1, 1.5, 2, 2.5, 3):
        out.append(f'<text x="{x(t):.1f}" y="{ya}" text-anchor="middle" fill="{MUTED}">{t:g} s</text>')
    out.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write("\n".join(out))


def chunks(period, dur, kind, label, offset=0.0, n=8):
    return [(offset + i * period, dur, kind, f"{label} {i + 1}") for i in range(n)]


# Classic split 6:6, decoder after the DiT on card 0, with all the split fixes: 20.06 FPS -> 0.798 s per 16 frames;
# decode ~0.32 s (one-card measurement on the same pod), DiT = the rest. Card 1 idles while card 0 decodes.
c, dec = 16 / 20.06, 0.32
svg("layout_classic_6_6.svg", [
    ("card 0", [b for i in range(4) for b in ((i * c, c - dec, "dit", f"DiT {i + 1}"), (i * c + c - dec, dec, "dec", f"dec {i + 1}"))]),
    ("card 1", [(i * c, c - dec, "dit", f"DiT {i + 1}") for i in range(4)]),
])
# B1 (pipeline): 25.3 FPS -> 0.633 s per chunk on card 0; card 1 decodes chunk n-1 (~0.34 s) while card 0 runs chunk n.
c = 16 / 25.3
svg("layout_b1.svg", [
    ("card 0", chunks(c, c, "dit", "DiT")),
    ("card 1", chunks(c, 0.34, "dec", "dec", offset=c)),
])
# Best split 10:2: 29.9 FPS -> 0.535 s per chunk. Card 0: 10 of 12 heads, 5/6 of the tokens; card 1: its DiT share on
# 48 SMs (same chunk, in lockstep with card 0) and the decoder on the other 122 SMs (~0.43 s per chunk, chunk n-1).
c = 16 / 29.9
svg("layout_split_10_2.svg", [
    ("card 0", chunks(c, c, "dit", "DiT")),
    ("card 1 · 48 SMs", chunks(c, c, "dit", "DiT")),
    ("card 1 · 122 SMs", chunks(c, 0.43, "dec", "dec", offset=c)),
])
print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("layout_")))

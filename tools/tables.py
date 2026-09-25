#!/usr/bin/env python3
"""Render the result tables from bench/summary.json into README.md (Markdown) and the write-up (HTML rows).

Each target file carries marker comments; only the text between them is replaced, the styling around it is hand-made:
  Markdown:  <!-- table:engines -->  ...  <!-- /table:engines -->
  HTML:      <!-- table:engines -->  ...  <!-- /table:engines -->   (inside the <tbody>)

  python tools/tables.py                       # rewrite README.md in place
  python tools/tables.py --html path/to/index.html   # also rewrite the write-up's rows
  python tools/tables.py --check [--html ...]  # exit 1 if any target is out of date
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "bench", "summary.json")))


def speedup(fps):
    ours = next(r["fps"] for r in DATA["engines"]["rows"] if r.get("ours"))
    return f"{ours / fps:.1f}×"


# ---------- Markdown ----------

def md_engines():
    out = ["| Engine | s / chunk | FPS | Ours vs it |", "|---|---|---|---|"]
    for r in DATA["engines"]["rows"]:
        name = "**Ours**" if r.get("ours") else r["engine"]
        fps = f"**{r['fps']}**" if r.get("ours") else f"{r['fps']}"
        sp = "—" if r.get("ours") else f"**{speedup(r['fps'])}**"
        out.append(f"| {name} | {r['s_per_chunk']:.2f} | {fps} | {sp} |")
    return "\n".join(out)


def md_baseline():
    out = ["| | Original paper's code | Ours |", "|---|---|---|"]
    for r in DATA["baseline_vs_ours"]["rows"]:
        out.append(f"| {r['metric']} | {r['before']} | **{r['after']}** |")
    return "\n".join(out)


def md_ladder():
    L = DATA["ladder"]
    nb = lambda x: x.replace(" ", "&nbsp;")
    out = ["| Step | Before | After | s/chunk |", "|---|---|---|---|"]
    prev = L["start"]
    for r in L["rows"]:
        after = r["after"]
        if r.get("after_url"):
            after = after.replace(r["after_link_text"], f"[{r['after_link_text']}]({r['after_url']})")
        before = r["before"]
        if r.get("before_url"):
            t = r.get("before_link_text", before)
            before = before.replace(t, f"[{t}]({r['before_url']})")
        out.append(f"| {nb(r['step'])} | {before} | {after} | {prev:.2f}&nbsp;→&nbsp;{r['after_s']:.2f} |")
        prev = r["after_s"]
    t = L["total"]
    out.append(f"| **Total** | {nb(t['before'])} | **{nb(t['after'])}** | **{t['before_s']:.2f}&nbsp;→&nbsp;{t['after_s']:.2f}** |")
    return "\n".join(out)


def md_peaks():
    out = ["| Kernel | Reached | Peak on RTX 5090 | of peak |", "|---|---|---|---|"]
    for r in DATA["peaks"]["rows"]:
        out.append(f"| {r['kernel']} | {r['reached']} | {r['peak']} | **{r['pct']}** |")
    return "\n".join(out)


def md_sol():
    out = ["| Precision | Peak on RTX 5090 | Original paper's code | Ours |", "|---|---|---|---|"]
    for r in DATA["sol"]["rows"]:
        out.append(f"| {r['precision']} | {r['peak']} | {r['paper']} | {r['ours']} |")
    return "\n".join(out)


def md_sol_chunk():
    S = DATA["sol_chunk"]
    out = ["| Component | Work per chunk | Original paper's code | Ours |", "|---|---|---|---|"]
    for r in S["rows"]:
        out.append(f"| {r['component']} | {r['work']} | {r['paper_peak']} → {r['paper_floor']:.2f} s | {r['ours_peak']} → {r['ours_floor']:.2f} s |")
    P, O = S["paper"], S["ours"]
    out.append(f"| **Speed of light** | | **{P['floor_s']:.2f} s, {P['ceiling_fps']} FPS** | **{O['floor_s']:.2f} s, {O['ceiling_fps']} FPS** |")
    out.append(f"| Measured | | {P['measured_s']:.2f} s, {P['measured_fps']} FPS ({P['pct']} %) | {O['measured_s']:.2f} s, {O['measured_fps']} FPS ({O['pct']} %) |")
    return "\n".join(out)


def _rf(r):
    f = f"{r['flops_t']:.0f} T" if r.get("flops_t") else "—"
    b = f"{'~' if r.get('est') else ''}{r['bytes_gb']:.0f} GB"
    i = f"{r['intensity']:,}" if r.get("intensity") else "—"
    pct = f"**{r['pct']} %**" if r.get("pct") is not None else "—"
    return f, b, i, pct


def md_roofline(who="ours"):
    R = DATA["roofline"][who]
    out = ["| Operation | Precision | FLOP / chunk | Bytes / chunk | FLOP / B | Ridge | Bound | Floor | Measured | Of speed of light |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["rows"]:
        f, b, i, pct = _rf(r)
        out.append(f"| {r['op']} | {r['precision']} | {f} | {b} | {i} | {r['ridge']} | {r['bound']} | {r['floor_s']:.3f} s | {r['measured_s']:.3f} s | {pct} |")
    out.append(f"| **Chunk** | | | | | | | **{R['floor_s']:.2f} s, {R['ceiling_fps']} FPS** | **{R['chunk_s']:.2f} s, {R['chunk_fps']} FPS** | **{R['pct']} %** |")
    return "\n".join(out)


def html_roofline(who="ours"):
    R = DATA["roofline"][who]
    out = []
    for r in R["rows"]:
        f, b, i, pct = _rf(r)
        pct = pct.replace("**", "")
        out.append(f'  <tr><td>{r["op"]}</td><td class="what">{r["precision"]}</td><td class="what">{f}</td><td class="what">{b}</td><td class="what">{i}</td><td class="what">{r["ridge"]}</td><td class="what">{r["bound"]}</td><td class="what">{r["floor_s"]:.3f} s</td><td class="what">{r["measured_s"]:.3f} s</td><td class="num hi">{pct}</td></tr>')
    out.append(f'  <tr class="sum"><td>Chunk</td><td></td><td></td><td></td><td></td><td></td><td></td><td class="what">{R["floor_s"]:.2f} s, {R["ceiling_fps"]} FPS</td><td class="what">{R["chunk_s"]:.2f} s, {R["chunk_fps"]} FPS</td><td class="num hi">{R["pct"]} %</td></tr>')
    return "\n".join(out)


def md_quality():
    # No links here on purpose: the README's "What those metrics mean" table right below this one
    # carries them, and nothing should appear twice. The blog still links via html_quality's cites.
    out = ["| | Original paper's code | Ours |", "|---|---|---|"]
    for r in DATA["quality"]["rows"]:
        out.append(f"| {r['metric']} | {r['before']} | **{r['after']}** |")
    return "\n".join(out)


# ---------- HTML rows (the write-up) ----------

def html_engines():
    out = []
    for r in DATA["engines"]["rows"]:
        if r.get("ours"):
            out.append(f'  <tr class="ours"><td>Ours</td><td class="num">{r["s_per_chunk"]:.2f}</td><td class="num">{r["fps"]}</td><td class="num">—</td></tr>')
        else:
            cite = f' <span class="cite">[<a href="{r["url"]}">{r["cite"]}</a>]</span>' if r.get("cite") else ""
            out.append(f'  <tr><td>{r["engine"]}{cite}</td><td class="num">{r["s_per_chunk"]:.2f}</td><td class="num">{r["fps"]}</td><td class="num">{speedup(r["fps"])}</td></tr>')
    return "\n".join(out)


def html_baseline_small():
    out = []
    for r in DATA["baseline_vs_ours"]["rows"]:
        if r.get("big"):
            continue
        out.append(f'    <tr><td>{r["metric"]}</td><td class="num">{r["before"]}</td><td class="arr">→</td><td class="num hi">{r["after"]}</td></tr>')
    return "\n".join(out)


def html_ladder():
    L = DATA["ladder"]
    out = []
    prev = L["start"]
    for r in L["rows"]:
        after = r["after"]
        if r.get("after_url"):
            after = after.replace(r["after_link_text"], f'<a href="{r["after_url"]}">{r["after_link_text"]}</a>')
        before = r["before"]
        if r.get("before_url"):
            t = r.get("before_link_text", before)
            before = before.replace(t, f'<a href="{r["before_url"]}">{t}</a>')
        w = round(r["after_s"] / L["start"] * 100)
        out.append(f'  <tr><td>{r["step"]}</td><td class="before">{before}</td><td class="after">{after}</td><td class="num"><span class="bar"><i style="width:{w}%"></i></span><span class="b">{prev:.2f}</span> → {r["after_s"]:.2f}</td></tr>')
        prev = r["after_s"]
    t = L["total"]
    w = round(t["after_s"] / L["start"] * 100)
    out.append(f'  <tr class="ours"><td>Total</td><td class="before">{t["before"]}</td><td class="after">{t["after"]}</td><td class="num"><span class="bar"><i style="width:{w}%"></i></span><span class="b">{t["before_s"]:.2f}</span> → {t["after_s"]:.2f}</td></tr>')
    return "\n".join(out)


def html_peaks():
    return "\n".join(f'  <tr><td>{r["kernel"]}</td><td class="what">{r["reached"]}</td><td class="what">{r["peak"]}</td><td class="num hi">{r["pct"]}</td></tr>' for r in DATA["peaks"]["rows"])


def html_sol(who="both"):
    """who: "both", "paper" or "ours"; a single-stack table keeps only the rows that stack uses."""
    def cell(x, hi):
        return f'<td class="{"hi" if hi else "dim"}">{x}</td>'
    rows = []
    for r in DATA["sol"]["rows"]:
        if who != "both" and r[who] == "—":
            continue
        use = "".join(cell(r[k], r[k] != "—") for k in (["paper", "ours"] if who == "both" else [who]))
        rows.append(f'  <tr><td>{r["precision"]}</td><td class="what">{r["peak"]}</td>{use}</tr>')
    return "\n".join(rows)


def html_sol_chunk(who="both"):
    S = DATA["sol_chunk"]
    stacks = ["paper", "ours"] if who == "both" else [who]
    out = []
    for r in S["rows"]:
        use = "".join(f'<td><span class="b">{r[k + "_peak"]}</span> → {r[k + "_floor"]:.2f} s</td>' for k in stacks)
        out.append(f'  <tr><td>{r["component"]}</td><td class="what">{r["work"]}</td>{use}</tr>')
    use = "".join(f'<td class="hi">{S[k]["floor_s"]:.2f} s, {S[k]["ceiling_fps"]} FPS</td>' for k in stacks)
    out.append(f'  <tr class="sum"><td>Speed of light</td><td></td>{use}</tr>')
    use = "".join(f'<td>{S[k]["measured_s"]:.2f} s, {S[k]["measured_fps"]} FPS, {S[k]["pct"]} % of speed of light</td>' for k in stacks)
    out.append(f'  <tr><td>Measured</td><td></td>{use}</tr>')
    return "\n".join(out)


def html_quality():
    out = []
    for r in DATA["quality"]["rows"]:
        m = r["metric"]
        if r.get("cite"):
            m = f'{m} <span class="cite">[<a href="{r["url"]}">{r["cite"]}</a>]</span>'
        out.append(f'  <tr><td>{m}</td><td class="num">{r["before"]}</td><td class="arr">→</td><td class="num hi">{r["after"]}</td></tr>')
    return "\n".join(out)


MD = {"engines": md_engines, "baseline": md_baseline, "ladder": md_ladder, "sol": md_sol, "sol_chunk": md_sol_chunk, "roofline": md_roofline, "roofline_paper": lambda: md_roofline("paper"), "peaks": md_peaks, "quality": md_quality}
HTML = {"engines": html_engines, "baseline": html_baseline_small, "ladder": html_ladder, "sol_paper": lambda: html_sol("paper"), "sol_chunk_paper": lambda: html_sol_chunk("paper"), "sol_ours": lambda: html_sol("ours"), "sol_chunk_ours": lambda: html_sol_chunk("ours"), "roofline": html_roofline, "roofline_paper": lambda: html_roofline("paper"), "peaks": html_peaks, "quality": html_quality}


def render(path, renderers, check):
    text = open(path).read()
    new = text
    # inline numbers: <!-- n:key -->value<!-- /n -->
    for key, val in DATA["numbers"].items():
        if key.startswith("_"):
            continue
        new = re.sub(rf"(<!-- n:{key} -->)(.*?)(<!-- /n -->)", lambda m: m.group(1) + val + m.group(3), new)
    for name, fn in renderers.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n)(.*?)(<!-- /table:{name} -->)", re.S)
        if not pat.search(new):
            continue
        new = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(3), new)
    if new == text:
        print(f"{path}: up to date")
        return True
    if check:
        print(f"{path}: OUT OF DATE (run tools/tables.py)")
        return False
    open(path, "w").write(new)
    print(f"{path}: rewritten")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="the write-up's index.html")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok = render(os.path.join(ROOT, "README.md"), MD, a.check)
    if a.html:
        ok = render(a.html, HTML, a.check) and ok
    sys.exit(0 if ok else 1)

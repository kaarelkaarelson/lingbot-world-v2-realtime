"""Where does each GPU stream wait? Gaps between consecutive kernels on a stream, grouped by (kernel before -> kernel
after, NVTX range open on the CPU when the next kernel was launched), plus GPU time per NVTX range of a chosen name.

  python gap_report.py capture.sqlite [--min-us 50] [--per 4 --per-label chunk] [--span fwd] [--stages pre,attn,post]

--per N divides totals by N (e.g. captured iterations). --span NAME prints the GPU time (first to last kernel launched
inside each NVTX range NAME) per stream, in order: an outlier iteration shows up there. Standard library only.
"""
import argparse
import bisect
import collections
import re
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--min-us", type=float, default=50)
ap.add_argument("--per", type=float, default=1, help="divide totals by this (captured iterations)")
ap.add_argument("--span", default="fwd", help="NVTX range name to report per-range GPU time for")
ap.add_argument("--stages", default="", help="comma list of NVTX range names that count as stages (default: innermost)")
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()

c = sqlite3.connect(a.db)
S = dict(c.execute("select id, value from StringIds"))
cls = lambda n: re.sub(r"[_<(].*$", "", n.split("::")[-1])[:22] or "kernel"
R = {cid: (g, t) for g, t, cid in c.execute("select globalTid, start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
main = collections.Counter(g for g, _ in R.values()).most_common(1)[0][0]
cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
N = sorted((s, e, t) for s, e, t, g in c.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS "
                                                 "where end is not null and eventType in (59, 60)") if g == main and t)
want = set(a.stages.split(",")) if a.stages else None
stages = [x for x in N if (x[2] in want if want else True)]
spans = [x for x in N if x[2] == a.span]


def inside(lst, t):
    """Innermost range of lst (sorted by start) containing t."""
    best = None
    for s, e, n in reversed(lst[:bisect.bisect_right([x[0] for x in lst], t)]):
        if s <= t <= e and (best is None or s > best[0]):
            best = (s, e, n)
            if want is not None:
                break
        if best is not None and s < best[0] - 50_000_000:
            break
    return best


streams = c.execute("select deviceId, streamId, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by 1, 2 having count(*) > 100").fetchall()
for d, s, _ in streams:
    K = list(c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
                       "where deviceId = ? and streamId = ? order by start", (d, s)))
    busy = sum(k[1] - k[0] for k in K)
    tot, sizes = collections.defaultdict(lambda: [0, 0]), []
    for i in range(len(K) - 1):
        g = K[i + 1][0] - K[i][1]
        sizes.append(g)
        if g < a.min_us * 1000:
            continue
        t = R.get(K[i + 1][3], (None, None))[1]
        st = inside(stages, t) if t is not None else None
        key = (cls(S.get(K[i][2], "")), cls(S.get(K[i + 1][2], "")), st[2] if st else "no range")
        tot[key][0] += g
        tot[key][1] += 1
    span = K[-1][1] - K[0][0]
    print(f"\n== GPU {d} stream {s}: busy {100 * busy / span:.1f} %, gaps > {a.min_us:g} us: "
          f"{sum(v[0] for v in tot.values()) / 1e6 / a.per:.1f} ms per iteration")
    for (x, y, w), (g, n) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {g / 1e6 / a.per:7.2f} ms  n={n:5d}  avg {g / n / 1e3:6.0f} us  {x} -> {y}   (next launched in '{w}')")
    print("  largest gaps (us):", sorted((round(g / 1e3) for g in sizes), reverse=True)[:8])
    if spans:
        per = collections.defaultdict(lambda: [float("inf"), 0])
        starts = [x[0] for x in spans]
        for st_, en, _, cid in K:
            t = R.get(cid, (None, None))[1]
            j = bisect.bisect_right(starts, t) - 1 if t is not None else -1
            if j >= 0 and t <= spans[j][1]:
                per[j][0] = min(per[j][0], st_)
                per[j][1] = max(per[j][1], en)
        print(f"  GPU ms per '{a.span}':", [round((v[1] - v[0]) / 1e6, 1) for _, v in sorted(per.items())])

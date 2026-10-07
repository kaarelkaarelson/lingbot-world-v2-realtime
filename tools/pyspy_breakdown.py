"""Aggregate a py-spy `--format raw` (collapsed stacks) profile: where does the Python thread spend CPU?

  python pyspy_breakdown.py profile.txt [--rate 250] [--top 25] [--under "fn (file.py"] [--depth 2]

Prints inclusive time of the busiest functions, and with --under the children of one function
(e.g. --under "_layer (lingbot/parallel/split_dit.py") so you can see what a hot loop spends CPU on.
Standard library only.
"""
import argparse
import collections
import re

ap = argparse.ArgumentParser()
ap.add_argument("profile")
ap.add_argument("--rate", type=float, default=250.0, help="the --rate py-spy recorded with (samples/s)")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--under", action="append", default=[], help="substring of a frame; show its children")
ap.add_argument("--depth", type=int, default=1)
a = ap.parse_args()

rows = []
for line in open(a.profile, errors="replace"):
    line = line.rstrip()
    if line:
        stack, n = line.rsplit(" ", 1)
        rows.append((stack.split(";"), int(n)))
total = sum(n for _, n in rows)
norm = lambda f: re.sub(r":\d+\)$", ")", f)

incl = collections.Counter()
for frames, n in rows:
    for f in {norm(f) for f in frames}:
        incl[f] += n
print(f"{total} samples = {total / a.rate:.1f} s of CPU on all threads\n\ninclusive (top {a.top}):")
for f, n in incl.most_common(a.top):
    print(f"  {n / a.rate:7.2f} s  {n / total * 100:5.1f}%  {f[:140]}")

for target in a.under:
    tot, ch = 0, collections.Counter()
    for frames, n in rows:
        idx = [i for i, f in enumerate(frames) if target in f]
        if not idx:
            continue
        i = idx[0]
        tot += n
        ch[" > ".join(norm(x)[:70] for x in frames[i + 1:i + 1 + a.depth]) or "(self)"] += n
    print(f"\nunder {target!r}: {tot / a.rate:.2f} s")
    for k, n in ch.most_common(a.top):
        print(f"  {n / a.rate:7.2f} s  {n / max(tot, 1) * 100:5.1f}%  {k}")

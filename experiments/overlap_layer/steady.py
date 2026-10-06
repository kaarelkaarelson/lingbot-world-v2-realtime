"""Steady-state ms per layer for every layout (real_layer.steady_period), 12 layers, median from layer 4 on."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import real_layer as R  # noqa: E402

torch.set_grad_enabled(False)
T, D, H, HD, WIN, NL = R.T, R.D, R.H, R.HD, R.WIN, 12

x0 = torch.randn(T, D, device=R.dev[0], dtype=torch.bfloat16)

# one card
ws1 = [R.Weights(R.dev[0]) for _ in range(NL)]
kv1 = [(torch.randn(1, H, WIN, HD, device=R.dev[0], dtype=torch.bfloat16),
        torch.randn(1, H, WIN, HD, device=R.dev[0], dtype=torch.bfloat16)) for _ in range(NL)]


def one(state, l):
    return x0.clone() if l < 0 else R.one_card_layer(state, ws1[l], *kv1[l])


one(one(None, -1), 0)
res = {"one card": [R.steady_period(one, [torch.cuda.default_stream(R.dev[0])], NL) for _ in range(3)]}
del ws1, kv1

for name, t0, h0, g in (("split 6:6", T // 2, 6, 1), ("split 9:3", T * 3 // 4, 9, 1),
                        ("split 9:3, overlap 3 groups", T * 3 // 4, 9, 3)):
    sp = R.Split(t0, h0, g)
    ws = [[R.Weights(d) for d in R.dev] for _ in range(NL)]
    kv = [[(torch.randn(1, len(sp.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16),
            torch.randn(1, len(sp.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16)) for c in (0, 1)]
          for _ in range(NL)]
    x_in = [x0[:t0].clone(), x0[t0:].to(R.dev[1])]

    def fn(state, l):
        if l < 0:
            return [x_in[0].clone(), x_in[1].clone()]
        return sp.layer(state, ws[l], [kv[l][0][0], kv[l][1][0]], [kv[l][0][1], kv[l][1][1]])

    fn(fn(None, -1), 0)
    res[name + f", pieces {R.PIECES}"] = [R.steady_period(fn, sp.comp, NL) for _ in range(3)]
    del ws, kv

for k, v in res.items():
    print(f"{k}\t" + " / ".join(f"{x:.2f}" for x in v) + f"\tms per layer -> DiT {sorted(v)[1] * 150 / 1e3:.3f} s per chunk")

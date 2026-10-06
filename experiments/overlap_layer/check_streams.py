"""Does SageAttention give the default-stream result on a high-priority custom stream? (the split layer, 4 runs)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import real_layer as R  # noqa: E402

torch.set_grad_enabled(False)
T, H, D, HD, WIN = R.T, R.H, R.D, R.HD, R.WIN
w0, w1 = R.Weights(R.dev[0]), R.Weights(R.dev[1])
kw = torch.randn(1, H, WIN, HD, device=R.dev[0], dtype=torch.bfloat16)
vw = torch.randn_like(kw)
x = torch.randn(T, D, device=R.dev[0], dtype=torch.bfloat16)


def once(hiprio):
    s = R.Split(T * 3 // 4, 9, 1)
    if hiprio:
        s.comp = [torch.cuda.Stream(device=d, priority=-5) for d in R.dev]
    xs = s.layer([x[:T * 3 // 4].clone(), x[T * 3 // 4:].to(R.dev[1])], [w0, w1],
                 [kw[:, :9].clone(), kw[:, 9:].to(R.dev[1])], [vw[:, :9].clone(), vw[:, 9:].to(R.dev[1])])
    for d in R.dev:
        torch.cuda.synchronize(d)
    return [t.float().cpu() for t in xs]


ref = once(False)
diffs = [max((a - b).abs().max().item() for a, b in zip(once(True), ref)) for _ in range(4)]
print("high-priority streams vs default stream, max |diff| over 4 runs:", diffs)
print("PASS" if all(d < 0.05 for d in diffs) else "FAIL")

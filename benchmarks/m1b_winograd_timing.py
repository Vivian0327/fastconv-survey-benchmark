# -*- coding: utf-8 -*-
"""M1b: explicit Winograd timing, output-normalized for a fair comparison.

The explicit Winograd paths emit only the whole-tile valid region, whose
output count differs from the 'same' paths of M1. We therefore report
throughput as ns per output element (ns/out), which is invariant to the
valid-region size, alongside raw median ms. Multi-seed, median + IQR.
"""
import csv, math, statistics
import torch
import torch.nn.functional as F
from bench_common import rel_rms, time_gpu

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"device: {torch.cuda.get_device_name(0)}")
SEEDS = 3

def mats_f23(dt):
    Bt = torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]], dtype=dt, device=dev)
    G  = torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]], dtype=dt, device=dev)
    At = torch.tensor([[1,1,1,0],[0,1,-1,-1]], dtype=dt, device=dev)
    return Bt, G, At

def mats_f43(dt):
    Bt = torch.tensor([[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
                       [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]], dtype=dt, device=dev)
    G = torch.tensor([[1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],
                      [1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]], dtype=dt, device=dev)
    At = torch.tensor([[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]], dtype=dt, device=dev)
    return Bt, G, At

def winograd_conv(x, w, mats, m):
    Bt, G, At = mats
    t = m + 2
    N, C, H, _ = x.shape
    K = w.shape[0]
    P = (H - 2) // m
    U = torch.einsum("ij,kcjl,ml->kcim", G, w, G)
    tiles = x.unfold(2, t, m).unfold(3, t, m)[:, :, :P, :P]
    V = torch.einsum("ij,ncpqjl,ml->ncpqim", Bt, tiles, Bt)
    M = torch.einsum("kcim,ncpqim->nkpqim", U, V)
    Y = torch.einsum("ij,nkpqjl,ml->nkpqim", At, M, At)
    return Y.permute(0, 1, 2, 4, 3, 5).reshape(N, K, P * m, P * m)

def conv_im2col(x, w):
    N, C, H, W = x.shape
    K, _, r, _ = w.shape
    cols = F.unfold(x, r, padding=r // 2)
    return (w.view(K, -1) @ cols).view(N, K, H, W)

rows = []
print(f"\n{'layer':16s} {'path':12s} {'ms':>8s} {'ns/out':>9s}  valid")
for (name, N, C, K, H) in [("resnet50-3x3-56", 8, 64, 64, 56),
                           ("resnet50-3x3-14", 8, 256, 256, 14)]:
    n_same = N * K * H * H
    variants = []  # (label, fn, n_out)
    def add(label, fn, n_out):
        variants.append((label, fn, n_out))
    # build per-seed timing
    perf = {}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(K, C, 3, 3, device=dev) / (3 * math.sqrt(C))
        specs = [("cudnn_best", lambda x=x, w=w: F.conv2d(x, w, padding="same"), n_same),
                 ("im2col",     lambda x=x, w=w: conv_im2col(x, w), n_same)]
        for lbl, mfn, m in [("Wino F(2,3)", mats_f23, 2), ("Wino F(4,3)", mats_f43, 4)]:
            P = (H - 2) // m; n_out = N * K * (P * m) ** 2
            mats = mfn(torch.float32)
            specs.append((lbl, lambda x=x, w=w, mats=mats, m=m: winograd_conv(x, w, mats, m), n_out))
        for lbl, fn, n_out in specs:
            ms, iqr = time_gpu(fn, warmup=10, iters=50)
            perf.setdefault(lbl, {"ms": [], "nsout": [], "nout": n_out})
            perf[lbl]["ms"].append(ms)
            perf[lbl]["nsout"].append(ms * 1e6 / n_out)  # ns per output
    print(f"\n[{name}]")
    for lbl, d in perf.items():
        ms = statistics.median(d["ms"]); nsout = statistics.median(d["nsout"])
        frac = 100 * d["nout"] / n_same
        rows.append([name, lbl, f"{ms:.3f}", f"{nsout:.2f}", f"{frac:.0f}"])
        print(f"  {lbl:12s} {ms:8.3f} {nsout:9.2f}  {frac:.0f}% of same-conv outputs")

with open("m1b_results.csv", "w", newline="") as f:
    wc = csv.writer(f)
    wc.writerow(["layer", "path", "median_ms", "ns_per_output", "pct_of_same_outputs"])
    wc.writerows(rows)
print("\nresults -> m1b_results.csv")
print("Fair comparison uses ns/out (invariant to valid-region size).")

# -*- coding: utf-8 -*-
"""M4: depthwise stressor -- illustrates the amortization gap (paper Sec. VIII-A).

cuDNN-best vs ATen direct (cudnn disabled) vs explicit per-channel FFT on
depthwise layers. Multi-seed median + IQR; error vs fp64 direct.
"""
import csv, math, statistics
import torch
import torch.nn.functional as F
from bench_common import rel_rms, rel_linf, time_gpu

dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
SEEDS = 3

LAYERS = [  # (name, N, C, H, r)  depthwise, groups=C
    ("mobilenet-dw3x3-56", 8, 144, 56,  3),
    ("convnext-dw7x7-14",  8, 384, 14,  7),
    ("convnext-dw7x7-28",  8, 192, 28,  7),
    ("replknet-dw31x31-28",8, 128, 28, 31),
]

def conv_cudnn(x, w):
    torch.backends.cudnn.enabled = True
    return F.conv2d(x, w, padding="same", groups=x.shape[1])

def conv_aten_direct(x, w):
    torch.backends.cudnn.enabled = False
    try:
        return F.conv2d(x, w, padding="same", groups=x.shape[1])
    finally:
        torch.backends.cudnn.enabled = True

def conv_fft_dw(x, w):
    N, C, H, W = x.shape
    r = w.shape[-1]
    s = H + r - 1
    wf = torch.flip(w, (-2, -1)).squeeze(1)
    Xf = torch.fft.rfft2(x, s=(s, s))
    Wf = torch.fft.rfft2(wf, s=(s, s))
    y = torch.fft.irfft2(Xf * Wf.unsqueeze(0), s=(s, s))
    lo = r - 1 - r // 2
    return y[..., lo:lo + H, lo:lo + W]

PATHS = {"cudnn_best": conv_cudnn, "aten_direct": conv_aten_direct, "fft_dw": conv_fft_dw}

rows = []
print(f"\n{'layer':22s} {'path':12s} {'ms':>8s} {'IQR':>6s} {'vs cudnn':>9s} {'relRMS':>9s}")
for (name, N, C, H, r) in LAYERS:
    agg = {p: {"ms": [], "iqr": [], "rms": [], "linf": []} for p in PATHS}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(C, 1, r, r, device=dev) / r
        ref = F.conv2d(x.double(), w.double(), padding="same", groups=C)
        for pname, fn in PATHS.items():
            ms, iqr = time_gpu(lambda: fn(x, w), warmup=10, iters=50)
            agg[pname]["ms"].append(ms); agg[pname]["iqr"].append(iqr)
            agg[pname]["rms"].append(rel_rms(fn(x, w), ref))
            agg[pname]["linf"].append(rel_linf(fn(x, w), ref))
    base = statistics.median(agg["cudnn_best"]["ms"])
    for pname in PATHS:
        a = agg[pname]
        ms = statistics.median(a["ms"]); iqr = statistics.median(a["iqr"])
        rms = statistics.median(a["rms"]); linf = statistics.median(a["linf"])
        rows.append([name, r, pname, f"{ms:.3f}", f"{iqr:.3f}", f"{rms:.2e}", f"{linf:.2e}"])
        print(f"{name:22s} {pname:12s} {ms:8.3f} {iqr:6.3f} {base/ms:8.2f}x {rms:9.2e}")
    print()

with open("m4_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "r", "path", "median_ms", "iqr_ms",
                "rel_rms_vs_fp64", "rel_linf_peaknorm_vs_fp64"])
    w.writerows(rows)
print("results -> m4_results.csv")

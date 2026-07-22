# -*- coding: utf-8 -*-
"""M3: CPU regime sweep — oneDNN autotuned vs native (oneDNN off) vs
explicit im2col vs explicit FFT, same layer set as M1.

CPU: Intel i7-10700 (8C/16T, AVX2). fp32. Median over warmed-up runs.
"""
import math, statistics, time, torch
import torch.nn.functional as F

torch.manual_seed(0)
print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")

LAYERS = [  # same set as M1
    ("resnet50-3x3-56",  8,  64,  64, 56,  3),
    ("resnet50-3x3-14",  8, 256, 256, 14,  3),
    ("dense-5x5-56",     8,  64,  64, 56,  5),
    ("dense-7x7-56",     8,  64,  64, 56,  7),
    ("dense-13x13-28",   8,  64,  64, 28, 13),
    ("dense-31x31-28",   8,  32,  32, 28, 31),
]

def conv_onednn(x, w):
    return F.conv2d(x, w, padding="same")

def conv_native(x, w):
    with torch.backends.mkldnn.flags(enabled=False):
        return F.conv2d(x, w, padding="same")

def conv_im2col(x, w):
    N, C, H, W = x.shape
    K, _, r, _ = w.shape
    cols = F.unfold(x, r, padding=r // 2)
    return (w.view(K, -1) @ cols).view(N, K, H, W)

def conv_fft(x, w):
    N, C, H, W = x.shape
    K, _, r, _ = w.shape
    s = H + r - 1
    Xf = torch.fft.rfft2(x, s=(s, s))
    Wf = torch.fft.rfft2(torch.flip(w, (-2, -1)), s=(s, s))
    y = torch.fft.irfft2(torch.einsum("nchw,kchw->nkhw", Xf, Wf), s=(s, s))
    lo = r - 1 - r // 2
    return y[..., lo:lo + H, lo:lo + W]

PATHS = {"onednn": conv_onednn, "native": conv_native,
         "im2col": conv_im2col, "fft": conv_fft}

def time_cpu(fn, x, w, warmup=3, iters=15):
    for _ in range(warmup):
        fn(x, w)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x, w)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

print(f"\n{'layer':18s} {'path':8s} {'ms':>9s} {'vs onednn':>10s} {'err':>10s}")
import csv
rows = []
for (name, N, C, K, H, r) in LAYERS:
    x = torch.randn(N, C, H, H)
    w = torch.randn(K, C, r, r) / (r * math.sqrt(C))
    ref = F.conv2d(x.double(), w.double(), padding="same")
    base = None
    for pname, fn in PATHS.items():
        ms = time_cpu(fn, x, w)
        err = ((fn(x, w).double() - ref).abs().max() / ref.abs().max()).item()
        if pname == "onednn":
            base = ms
        rows.append([name, r, pname, f"{ms:.2f}", f"{err:.2e}"])
        print(f"{name:18s} {pname:8s} {ms:9.2f} {base/ms:9.2f}x {err:10.2e}")
    print()

with open("m3_results.csv", "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["layer", "r", "path", "median_ms", "max_rel_err_vs_fp64"])
    wcsv.writerows(rows)
print("results -> m3_results.csv")

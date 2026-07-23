# -*- coding: utf-8 -*-
"""M3: CPU regime sweep -- oneDNN vs native (oneDNN off) vs explicit
im2col vs explicit FFT, same layer set and protocol as M1.

Illustrative measurements on one CPU (Intel i7-10700, AVX2), fp32.
Median + IQR over 50 warmed-up runs, 3 seeds; error vs fp64 direct.
"""
import csv, math, statistics
import torch
import torch.nn.functional as F
from bench_common import rel_rms, rel_linf, time_cpu

torch.manual_seed(0)
print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")

LAYERS = [
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
SEEDS = 3

rows = []
print(f"\n{'layer':18s} {'path':8s} {'ms':>9s} {'IQR':>7s} {'vs onednn':>10s} {'relRMS':>9s}")
for (name, N, C, K, H, r) in LAYERS:
    agg = {p: {"ms": [], "iqr": [], "rms": [], "linf": []} for p in PATHS}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(N, C, H, H)
        w = torch.randn(K, C, r, r) / (r * math.sqrt(C))
        ref = F.conv2d(x.double(), w.double(), padding="same")
        for pname, fn in PATHS.items():
            ms, iqr = time_cpu(lambda: fn(x, w), warmup=10, iters=50)
            agg[pname]["ms"].append(ms); agg[pname]["iqr"].append(iqr)
            agg[pname]["rms"].append(rel_rms(fn(x, w), ref))
            agg[pname]["linf"].append(rel_linf(fn(x, w), ref))
    base = statistics.median(agg["onednn"]["ms"])
    for pname in PATHS:
        a = agg[pname]
        ms = statistics.median(a["ms"]); iqr = statistics.median(a["iqr"])
        rms = statistics.median(a["rms"]); linf = statistics.median(a["linf"])
        rows.append([name, r, pname, f"{ms:.2f}", f"{iqr:.2f}",
                     f"{rms:.2e}", f"{linf:.2e}"])
        print(f"{name:18s} {pname:8s} {ms:9.2f} {iqr:7.2f} {base/ms:9.2f}x {rms:9.2e}")
    print()

with open("m3_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "r", "path", "median_ms", "iqr_ms",
                "rel_rms_vs_fp64", "rel_linf_peaknorm_vs_fp64"])
    w.writerows(rows)
print("results -> m3_results.csv")

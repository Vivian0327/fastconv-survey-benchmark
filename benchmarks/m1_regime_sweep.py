# -*- coding: utf-8 -*-
"""M1: GPU regime sweep -- cuDNN-best vs explicit im2col vs explicit FFT.

Illustrative measurements (NOT a universal validation): one Turing GPU,
fp32, batch 8. See the paper's Threats-to-Validity for scope limits.
Protocol: median + IQR over `iters` warmed-up runs, across `--seeds` seeds;
error vs an fp64 direct reference (rel_rms and peak-normalized rel_linf).

Usage:  python m1_regime_sweep.py [--quick] [--seeds 3] [--csv out.csv]
"""
import argparse, csv, math, statistics
import torch
import torch.nn.functional as F
from bench_common import rel_rms, rel_linf, time_gpu

LAYERS = [  # (name, N, C, K, H, r)  stride 1, square, same-channel
    ("resnet50-3x3-56",  8,  64,  64, 56,  3),
    ("resnet50-3x3-14",  8, 256, 256, 14,  3),
    ("dense-5x5-56",     8,  64,  64, 56,  5),
    ("dense-7x7-56",     8,  64,  64, 56,  7),
    ("dense-13x13-28",   8,  64,  64, 28, 13),
    ("dense-31x31-28",   8,  32,  32, 28, 31),
]

def conv_cudnn(x, w):
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

PATHS = {"cudnn_best": conv_cudnn, "im2col": conv_im2col, "fft": conv_fft}

def peak_mem(fn):
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--csv", default="m1_results.csv")
    args = ap.parse_args()
    warmup, iters = (3, 5) if args.quick else (10, 50)

    assert torch.cuda.is_available(), "CUDA build of PyTorch required"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"seeds={args.seeds}, iters={iters}")

    rows = []
    for (name, N, C, K, H, r) in LAYERS:
        # per-path lists across seeds
        agg = {p: {"ms": [], "iqr": [], "rms": [], "linf": [], "mem": []} for p in PATHS}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            x = torch.randn(N, C, H, H, device=dev)
            w = torch.randn(K, C, r, r, device=dev) / (r * math.sqrt(C))
            ref = F.conv2d(x.double(), w.double(), padding="same")
            for pname, fn in PATHS.items():
                f = lambda: fn(x, w)
                try:
                    ms, iqr = time_gpu(f, warmup, iters)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); continue
                agg[pname]["ms"].append(ms); agg[pname]["iqr"].append(iqr)
                agg[pname]["rms"].append(rel_rms(fn(x, w), ref))
                agg[pname]["linf"].append(rel_linf(fn(x, w), ref))
                agg[pname]["mem"].append(peak_mem(f))
        base = statistics.median(agg["cudnn_best"]["ms"]) if agg["cudnn_best"]["ms"] else None
        print(f"\n[{name}]  r={r}")
        for pname in PATHS:
            a = agg[pname]
            if not a["ms"]:
                rows.append([name, r, pname, "OOM", "", "", "", ""]); continue
            ms = statistics.median(a["ms"]); iqr = statistics.median(a["iqr"])
            rms = statistics.median(a["rms"]); linf = statistics.median(a["linf"])
            mem = statistics.median(a["mem"])
            sp = base / ms if base else float("nan")
            rows.append([name, r, pname, f"{ms:.3f}", f"{iqr:.3f}", f"{mem:.1f}",
                         f"{rms:.2e}", f"{linf:.2e}"])
            print(f"  {pname:10s} {ms:7.3f} ms (IQR {iqr:5.3f})  x{sp:5.2f}  "
                  f"mem {mem:7.1f}  relRMS {rms:.2e}  relLinf {linf:.2e}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "r", "path", "median_ms", "iqr_ms", "peak_mem_MiB",
                    "rel_rms_vs_fp64", "rel_linf_peaknorm_vs_fp64"])
        w.writerows(rows)
    print("\nresults ->", args.csv)

if __name__ == "__main__":
    main()

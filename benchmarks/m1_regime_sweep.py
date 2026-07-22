# -*- coding: utf-8 -*-
"""M1: GPU regime sweep — cuDNN-best vs explicit im2col vs explicit FFT.

Validates the phase-diagram crossovers (Fig. 1) on real hardware.
Protocol: Sec. 7.1 of the survey draft — median wall-clock latency over
warmed-up runs, peak workspace memory, elementwise error vs fp64 reference.

Usage:  python m1_regime_sweep.py [--quick] [--csv out.csv]
"""
import argparse, csv, math, statistics, sys
import torch
import torch.nn.functional as F

# ---------------- layer set (Sec. 7.1): dense convs across regimes ----------
# (name, N, C, K, H, r)  stride=1, same-channel square layers
LAYERS = [
    ("resnet50-3x3-56",  8,  64,  64, 56,  3),
    ("resnet50-3x3-14",  8, 256, 256, 14,  3),
    ("dense-5x5-56",     8,  64,  64, 56,  5),
    ("dense-7x7-56",     8,  64,  64, 56,  7),
    ("dense-13x13-28",   8,  64,  64, 28, 13),
    ("dense-31x31-28",   8,  32,  32, 28, 31),
]

# ---------------- convolution paths ----------------------------------------

def conv_cudnn(x, w):
    """Vendor-autotuned best path."""
    return F.conv2d(x, w, padding="same")

def conv_im2col(x, w):
    """Explicit im2col lowering: unfold -> GEMM -> fold."""
    N, C, H, W = x.shape
    K, _, r, _ = w.shape
    pad = r // 2
    cols = F.unfold(x, r, padding=pad)                 # N x C*r*r x H*W
    out = w.view(K, -1) @ cols                         # N x K x H*W
    return out.view(N, K, H, W)

def conv_fft(x, w):
    """Explicit frequency-domain convolution (full-map rFFT, pointwise
    channel contraction in frequency domain, crop). Linear conv via
    zero-padding to H+r-1; cross-correlation semantics matched to conv2d
    by kernel flip."""
    N, C, H, W = x.shape
    K, _, r, _ = w.shape
    s = H + r - 1
    wf = torch.flip(w, (-2, -1))
    Xf = torch.fft.rfft2(x, s=(s, s))                  # N x C x s x s/2+1
    Wf = torch.fft.rfft2(wf, s=(s, s))                 # K x C x s x s/2+1
    Yf = torch.einsum("nchw,kchw->nkhw", Xf, Wf)       # channel contraction
    y = torch.fft.irfft2(Yf, s=(s, s))
    lo = r - 1 - r // 2                                # 'same' crop
    return y[..., lo:lo + H, lo:lo + W]

PATHS = {"cudnn_best": conv_cudnn, "im2col": conv_im2col, "fft": conv_fft}

# ---------------- measurement ----------------------------------------------

def time_gpu(fn, x, w, warmup, iters):
    for _ in range(warmup):
        fn(x, w)
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn(x, w)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))          # ms
    return statistics.median(times)

def peak_mem(fn, x, w):
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(x, w)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20   # MiB

def max_rel_err(y, ref):
    scale = ref.abs().max().clamp_min(1e-30)
    return ((y.double() - ref).abs().max() / scale).item()

# ---------------- main ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer iters, smoke test")
    ap.add_argument("--csv", default="m1_results.csv")
    args = ap.parse_args()
    warmup, iters = (3, 5) if args.quick else (10, 50)

    assert torch.cuda.is_available(), "CUDA build of PyTorch required"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True             # let cuDNN autotune
    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    rows = []
    for (name, N, C, K, H, r) in LAYERS:
        torch.manual_seed(0)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(K, C, r, r, device=dev) / (r * math.sqrt(C))
        ref = F.conv2d(x.double(), w.double(), padding="same")  # fp64 reference

        base_ms = None
        for pname, fn in PATHS.items():
            try:
                ms = time_gpu(fn, x, w, warmup, iters)
                mem = peak_mem(fn, x, w)
                err = max_rel_err(fn(x, w), ref)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append([name, r, pname, "OOM", "", ""])
                print(f"{name:18s} {pname:10s} OOM")
                continue
            if pname == "cudnn_best":
                base_ms = ms
            speedup = base_ms / ms if base_ms else float("nan")
            rows.append([name, r, pname, f"{ms:.3f}", f"{mem:.1f}", f"{err:.2e}"])
            print(f"{name:18s} {pname:10s} {ms:8.3f} ms  x{speedup:5.2f} vs cudnn"
                  f"  mem {mem:8.1f} MiB  err {err:.2e}")
        print()

    with open(args.csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["layer", "r", "path", "median_ms", "peak_mem_MiB", "max_rel_err_vs_fp64"])
        wcsv.writerows(rows)
    print("results ->", args.csv)

if __name__ == "__main__":
    main()

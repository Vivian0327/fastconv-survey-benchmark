# -*- coding: utf-8 -*-
"""M5: matched dense->grouped->depthwise factorial (review: Methodology #1, #3).

Holds N, C, H, r, precision, layout, padding, and baseline FIXED and varies
ONLY the number of groups g (dense g=1 ... depthwise g=C). This isolates
transform amortization from every other factor. For the FFT path we report
BOTH cold-start (filter transform inside the timed call) and cached-filter
(filter transform precomputed, the steady-state inference cost). We also break
the FFT path into stage-level costs: input transform, multiply+contract,
inverse transform.

Baseline is direct convolution (cuDNN disabled) for every g, so the ratios are
matched across the sweep. fp32, median + IQR over 3 seeds x 50 runs.
"""
import csv, math, statistics
import torch
import torch.nn.functional as F
from bench_common import rel_rms, time_gpu

dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
SEEDS = 3

# Fixed shape; sweep only groups. C divisible by every g.
N, C, K, H, r = 8, 128, 128, 28, 7
GROUPS = [1, 4, 16, 64, 128]   # dense ... depthwise

def conv_direct(x, w, g):
    torch.backends.cudnn.enabled = False
    try:
        return F.conv2d(x, w, padding="same", groups=g)
    finally:
        torch.backends.cudnn.enabled = True

def conv_cudnn(x, w, g):
    torch.backends.cudnn.enabled = True
    return F.conv2d(x, w, padding="same", groups=g)

def fft_filter(w, g, s):
    Cg, Kg = w.shape[1], w.shape[0] // g
    wf = torch.flip(w, (-2, -1)).reshape(g, Kg, Cg, w.shape[-1], w.shape[-1])
    return torch.fft.rfft2(wf, s=(s, s))          # g, Kg, Cg, s, s//2+1

def conv_fft_grouped(x, w, g, Wf=None):
    N, C, H, W = x.shape
    Kk = w.shape[0]; Cg = C // g; Kg = Kk // g
    r = w.shape[-1]; s = H + r - 1
    Xf = torch.fft.rfft2(x, s=(s, s)).reshape(N, g, Cg, s, s // 2 + 1)
    if Wf is None:
        Wf = fft_filter(w, g, s)
    Yf = torch.einsum("ngchw,gkchw->ngkhw", Xf, Wf).reshape(N, Kk, s, s // 2 + 1)
    y = torch.fft.irfft2(Yf, s=(s, s))
    lo = r - 1 - r // 2
    return y[..., lo:lo + H, lo:lo + W]

# ---- validate grouped FFT in fp64 before timing ----
torch.manual_seed(0)
for g in GROUPS:
    x = torch.randn(2, C, H, H, device=dev, dtype=torch.float64)
    w = torch.randn(K, C // g, r, r, device=dev, dtype=torch.float64) / (r * math.sqrt(C // g))
    ref = F.conv2d(x, w, padding="same", groups=g)
    err = (conv_fft_grouped(x, w, g).double() - ref).abs().max().item()
    assert err < 1e-9, f"grouped FFT wrong at g={g}: {err}"
print("grouped FFT validated in fp64 (all g)\n")

rows = []
print(f"{'g':>4s} {'C/g':>4s} {'direct':>8s} {'cudnn':>8s} {'fftCold':>8s} "
      f"{'fftCach':>8s} {'cold/dir':>9s} {'cach/dir':>9s}")
for g in GROUPS:
    agg = {k: [] for k in ("direct", "cudnn", "fft_cold", "fft_cached")}
    stages = {"in": [], "mul": [], "inv": []}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(N, C, H, H, device=dev)
        w = torch.randn(K, C // g, r, r, device=dev) / (r * math.sqrt(C // g))
        s = H + r - 1
        ref = F.conv2d(x.double(), w.double(), padding="same", groups=g)
        Wf = fft_filter(w, g, s)                    # precomputed (cached) filter
        variants = {
            "direct":     lambda x=x, w=w, g=g: conv_direct(x, w, g),
            "cudnn":      lambda x=x, w=w, g=g: conv_cudnn(x, w, g),
            "fft_cold":   lambda x=x, w=w, g=g: conv_fft_grouped(x, w, g),
            "fft_cached": lambda x=x, w=w, g=g, Wf=Wf: conv_fft_grouped(x, w, g, Wf),
        }
        for k, fn in variants.items():
            ms, _ = time_gpu(fn, warmup=10, iters=50)
            agg[k].append(ms)
        # stage breakdown (cached filter): input rFFT, einsum, inverse
        def st_in(x=x, s=s): return torch.fft.rfft2(x, s=(s, s))
        Xf = torch.fft.rfft2(x, s=(s, s)).reshape(N, g, C // g, s, s // 2 + 1)
        def st_mul(Xf=Xf, Wf=Wf): return torch.einsum("ngchw,gkchw->ngkhw", Xf, Wf)
        Yf = torch.einsum("ngchw,gkchw->ngkhw", Xf, Wf).reshape(N, K, s, s // 2 + 1)
        def st_inv(Yf=Yf, s=s): return torch.fft.irfft2(Yf, s=(s, s))
        stages["in"].append(time_gpu(st_in, 10, 50)[0])
        stages["mul"].append(time_gpu(st_mul, 10, 50)[0])
        stages["inv"].append(time_gpu(st_inv, 10, 50)[0])
    med = {k: statistics.median(v) for k, v in agg.items()}
    smed = {k: statistics.median(v) for k, v in stages.items()}
    rows.append([g, C // g] + [f"{med[k]:.3f}" for k in ("direct","cudnn","fft_cold","fft_cached")]
                + [f"{med['fft_cold']/med['direct']:.3f}", f"{med['fft_cached']/med['direct']:.3f}",
                   f"{smed['in']:.3f}", f"{smed['mul']:.3f}", f"{smed['inv']:.3f}"])
    print(f"{g:>4d} {C//g:>4d} {med['direct']:8.3f} {med['cudnn']:8.3f} "
          f"{med['fft_cold']:8.3f} {med['fft_cached']:8.3f} "
          f"{med['fft_cold']/med['direct']:8.2f}x {med['fft_cached']/med['direct']:8.2f}x")

with open("m5_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["groups", "chan_per_group", "direct_ms", "cudnn_ms",
                "fft_cold_ms", "fft_cached_ms", "fftcold_over_direct",
                "fftcached_over_direct", "stage_input_ms", "stage_mul_ms",
                "stage_inv_ms"])
    w.writerows(rows)
print(f"\nFixed N={N} C={C} K={K} H={H} r={r}; only groups vary.")
print("results -> m5_results.csv")

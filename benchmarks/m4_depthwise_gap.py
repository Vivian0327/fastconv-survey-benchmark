# -*- coding: utf-8 -*-
"""M4: depthwise stressor — quantify the amortization gap (survey Sec. 8.1).

Compares cuDNN-best vs ATen direct (cudnn disabled) vs explicit per-channel
FFT on real depthwise layers, and contrasts the FFT advantage with the dense
case measured in M1.
"""
import math, statistics, torch
import torch.nn.functional as F

dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

# (name, N, C, H, r)  depthwise: groups = C, one filter per channel
LAYERS = [
    ("mobilenet-dw3x3-56", 8, 144, 56,  3),
    ("convnext-dw7x7-14",  8, 384, 14,  7),
    ("convnext-dw7x7-28",  8, 192, 28,  7),
    ("replknet-dw31x31-28",8, 128, 28, 31),
]

def conv_cudnn(x, w):
    torch.backends.cudnn.enabled = True
    return F.conv2d(x, w, padding="same", groups=x.shape[1])

def conv_aten_direct(x, w):
    torch.backends.cudnn.enabled = False        # ATen depthwise direct kernel
    try:
        return F.conv2d(x, w, padding="same", groups=x.shape[1])
    finally:
        torch.backends.cudnn.enabled = True

def conv_fft_dw(x, w):
    """Per-channel frequency-domain conv: NO cross-channel contraction, so
    每个通道的变换只被一个滤波器复用一次 — the amortization-free case."""
    N, C, H, W = x.shape
    r = w.shape[-1]
    s = H + r - 1
    wf = torch.flip(w, (-2, -1)).squeeze(1)     # C x r x r
    Xf = torch.fft.rfft2(x, s=(s, s))           # N C s s/2+1
    Wf = torch.fft.rfft2(wf, s=(s, s))          # C s s/2+1
    y = torch.fft.irfft2(Xf * Wf.unsqueeze(0), s=(s, s))
    lo = r - 1 - r // 2
    return y[..., lo:lo + H, lo:lo + W]

PATHS = {"cudnn_best": conv_cudnn, "aten_direct": conv_aten_direct,
         "fft_dw": conv_fft_dw}

def time_gpu(fn, x, w, warmup=10, iters=50):
    for _ in range(warmup):
        fn(x, w)
    torch.cuda.synchronize()
    ts, (s0, s1) = [], (torch.cuda.Event(True), torch.cuda.Event(True))
    for _ in range(iters):
        s0.record(); fn(x, w); s1.record()
        torch.cuda.synchronize()
        ts.append(s0.elapsed_time(s1))
    return statistics.median(ts)

print(f"\n{'layer':22s} {'path':12s} {'ms':>8s} {'vs cudnn':>9s} {'err':>10s}")
for (name, N, C, H, r) in LAYERS:
    torch.manual_seed(0)
    x = torch.randn(N, C, H, H, device=dev)
    w = torch.randn(C, 1, r, r, device=dev) / r
    ref = F.conv2d(x.double(), w.double(), padding="same", groups=C)
    base = None
    for pname, fn in PATHS.items():
        ms = time_gpu(fn, x, w)
        err = ((fn(x, w).double() - ref).abs().max() / ref.abs().max()).item()
        if pname == "cudnn_best":
            base = ms
        print(f"{name:22s} {pname:12s} {ms:8.3f} {base/ms:8.2f}x {err:10.2e}")
    print()

print("Context from M1 (dense, r=31): explicit FFT was 8.7x faster than "
      "explicit im2col;\nif the depthwise FFT advantage above is far smaller, "
      "the amortization gap is measured.")

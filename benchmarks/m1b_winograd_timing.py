# -*- coding: utf-8 -*-
"""M1b: explicit Winograd timing (fills review gap M3).
Times explicit F(2,3)/F(4,3) on the two r=3 layers of M1, same protocol.
Note: whole-tile valid region (>=93% of outputs); also reports us/Moutput.
"""
import math, statistics, torch
import torch.nn.functional as F

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"device: {torch.cuda.get_device_name(0)}")

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

def time_gpu(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts, (a, b) = [], (torch.cuda.Event(True), torch.cuda.Event(True))
    for _ in range(iters):
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)

for (name, N, C, K, H) in [("resnet50-3x3-56", 8, 64, 64, 56),
                            ("resnet50-3x3-14", 8, 256, 256, 14)]:
    torch.manual_seed(0)
    x = torch.randn(N, C, H, H, device=dev)
    w = torch.randn(K, C, 3, 3, device=dev) / (3 * math.sqrt(C))
    n_same = N * K * H * H            # outputs of the 'same' paths in M1
    print(f"\n[{name}]  (same-conv outputs = {n_same/1e6:.2f} M)")
    ms = time_gpu(lambda: F.conv2d(x, w, padding="same"))
    print(f"  cudnn_best    {ms:7.3f} ms   {ms*1e3/ (n_same/1e6):7.1f} us/Mout")
    for label, mfn, m in [("Wino F(2,3)", mats_f23, 2), ("Wino F(4,3)", mats_f43, 4)]:
        mats = mfn(torch.float32)
        P = (H - 2) // m
        n_out = N * K * (P * m) ** 2
        ms = time_gpu(lambda: winograd_conv(x, w, mats, m))
        print(f"  {label:12s}  {ms:7.3f} ms   {ms*1e3/(n_out/1e6):7.1f} us/Mout"
              f"   (valid {P*m}x{P*m}, {100*n_out/n_same:.0f}% of outputs)")

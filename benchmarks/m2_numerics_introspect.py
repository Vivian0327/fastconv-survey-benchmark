# -*- coding: utf-8 -*-
"""M2: (a) cuDNN algorithm introspection via profiler kernel names;
       (b) empirical Winograd stability tax: explicit F(2,3)/F(4,3)
           error growth across fp32/fp16 vs fp64 direct reference.
"""
import math, os, re, subprocess, sys, torch
import torch.nn.functional as F

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# ---------------- (a) what numerical notes does cuDNN's selected engine carry? --
# We read the cuDNN info log and collect the CUDNN_NUMERICAL_NOTE_* flags of the
# selected engine. IMPORTANT: these notes advertise engine PROPERTIES (uses FFT,
# uses Winograd, tensor-core, ...), NOT a verified algorithm identifier. Absence
# of a Winograd/FFT note is *consistent with* a GEMM/direct engine, not proof of
# one. Recovering exact engine IDs would require the cuDNN v9 backend API.

INTROSPECT = [  # (name, N, C, K, H, r, groups, dtype-str)
    ("3x3-56 fp32",   8,  64,  64, 56,  3, 1,   "fp32"),
    ("3x3-56 fp16",   8,  64,  64, 56,  3, 1,   "fp16"),
    ("7x7-56 fp32",   8,  64,  64, 56,  7, 1,   "fp32"),
    ("13x13-28 fp32", 8,  64,  64, 28, 13, 1,   "fp32"),
    ("31x31-28 fp32", 8,  32,  32, 28, 31, 1,   "fp32"),
    ("dw31x31 fp32",  8, 128, 128, 28, 31, 128, "fp32"),
]

def run_one_config(argv):
    """Child mode: single heuristic-mode conv so the log shows the one
    engine cuDNN picks (benchmark=False avoids autotune candidate noise)."""
    torch.backends.cudnn.benchmark = False
    N, C, K, H, r, g = map(int, argv[:6])
    dt = torch.float16 if argv[6] == "fp16" else torch.float32
    x = torch.randn(N, C, H, H, device=dev, dtype=dt)
    w = torch.randn(K, C // g, r, r, device=dev, dtype=dt)
    F.conv2d(x, w, padding="same", groups=g)
    torch.cuda.synchronize()

if len(sys.argv) > 2 and sys.argv[1] == "--child":
    run_one_config(sys.argv[2:])
    sys.exit(0)

print("=== (a) cuDNN engine numerical-note observations (not verified engine IDs) ===")
for (name, N, C, K, H, r, g, dts) in INTROSPECT:
    env = dict(os.environ, TORCH_CUDNN_V8_API_DISABLED="1",
               CUDNN_LOGINFO_DBG="1", CUDNN_LOGDEST_DBG="stdout")
    out = subprocess.run(
        [sys.executable, "-X", "utf8", __file__, "--child",
         str(N), str(C), str(K), str(H), str(r), str(g), dts],
        env=env, capture_output=True, text=True, timeout=300).stdout
    notes = {}
    for mm in re.finditer(r"CUDNN_NUMERICAL_NOTE_(\w+): type=bool; val=true", out):
        notes[mm.group(1)] = notes.get(mm.group(1), 0) + 1
    notes.pop("STRICT_NAN_PROP", None)      # uninformative
    tag = ", ".join(f"{k}:{v}" for k, v in
                    sorted(notes.items(), key=lambda kv: -kv[1])) \
          or "no Winograd/FFT note (consistent with GEMM/direct, not verified)"
    print(f"[{name:14s}] engine numerical notes: {tag}")

# ---------------- (b) explicit Winograd tile-size error growth --------------
# Standard matrices: F(2,3) points {0,1,-1}; F(4,3) points {0,1,-1,2,-2}.

def mats_f23(dt):
    Bt = torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]], dtype=dt)
    G  = torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]], dtype=dt)
    At = torch.tensor([[1,1,1,0],[0,1,-1,-1]], dtype=dt)
    return Bt, G, At

def mats_f43(dt):
    Bt = torch.tensor([
        [4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
        [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]], dtype=dt)
    G = torch.tensor([
        [1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],
        [1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]], dtype=dt)
    At = torch.tensor([
        [1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]], dtype=dt)
    return Bt, G, At

def winograd_conv(x, w, mats, m):
    """Explicit Winograd F(mxm,3x3), stride 1, VALID output, tiled."""
    Bt, G, At = (M.to(device=x.device, dtype=x.dtype) for M in mats)
    t = m + 2
    N, C, H, _ = x.shape
    K = w.shape[0]
    P = (H - 2) // m                                     # whole tiles per dim
    # filter transform: K x C x t x t
    U = torch.einsum("ij,kcjl,ml->kcim", G, w, G)
    # extract overlapping t x t tiles, stride m
    tiles = x.unfold(2, t, m).unfold(3, t, m)            # N C P P t t
    tiles = tiles[:, :, :P, :P]
    V = torch.einsum("ij,ncpqjl,ml->ncpqim", Bt, tiles, Bt)
    M = torch.einsum("kcim,ncpqim->nkpqim", U, V)        # elementwise + C-sum
    Y = torch.einsum("ij,nkpqjl,ml->nkpqim", At, M, At)  # N K P P m m
    return Y.permute(0, 1, 2, 4, 3, 5).reshape(N, K, P * m, P * m)

print("\n=== (b) Winograd stability tax across shapes/seeds (r=3, valid, vs fp64) ===")
print("    metric: peak-normalized max error (relLinf) = max|y-ref|/max|ref|")

# Sweep several (C, H) shapes and seeds so the F(2,3)->F(4,3) error ratio
# is reported as a measured range, not a single instance (review item).
SHAPES = [(64, 50), (128, 50), (32, 62), (256, 38)]   # (C=K, H)
SEEDS = 3

def rel_linf_ref(y, ref):
    r0 = ref[..., :y.shape[-2], :y.shape[-1]]
    return ((y.double() - r0).abs().max() / r0.abs().max().clamp_min(1e-30)).item()

# fp64 construction sanity (once)
torch.manual_seed(0)
xs = torch.randn(4, 16, 26, 26, device=dev, dtype=torch.float64)
ws = torch.randn(16, 16, 3, 3, device=dev, dtype=torch.float64)
rs = F.conv2d(xs, ws)
for nm, mf, m in [("F(2,3)", mats_f23, 2), ("F(4,3)", mats_f43, 4)]:
    print(f"  sanity fp64 {nm}: "
          f"{rel_linf_ref(winograd_conv(xs, ws, mf(torch.float64), m), rs):.2e}")

import statistics as st
tax32, tax16 = [], []
agg = {("im2col", torch.float32): [], ("im2col", torch.float16): [],
       ("F(2,3)", torch.float32): [], ("F(2,3)", torch.float16): [],
       ("F(4,3)", torch.float32): [], ("F(4,3)", torch.float16): []}
for (C, H) in SHAPES:
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x64 = torch.randn(8, C, H, H, device=dev, dtype=torch.float64)
        w64 = torch.randn(C, C, 3, 3, device=dev, dtype=torch.float64) / (3 * math.sqrt(C))
        ref = F.conv2d(x64, w64)
        for dt in (torch.float32, torch.float16):
            x, w = x64.to(dt), w64.to(dt)
            e_im = rel_linf_ref(F.conv2d(x, w), ref)
            e2 = rel_linf_ref(winograd_conv(x, w, mats_f23(dt), 2), ref)
            e4 = rel_linf_ref(winograd_conv(x, w, mats_f43(dt), 4), ref)
            agg[("im2col", dt)].append(e_im)
            agg[("F(2,3)", dt)].append(e2)
            agg[("F(4,3)", dt)].append(e4)
            (tax32 if dt == torch.float32 else tax16).append(e4 / e2)

print(f"\n  {'path':10s} {'fp32 (median)':>16s} {'fp16 (median)':>16s}")
for p in ("im2col", "F(2,3)", "F(4,3)"):
    m32 = st.median(agg[(p, torch.float32)]); m16 = st.median(agg[(p, torch.float16)])
    print(f"  {p:10s} {m32:16.2e} {m16:16.2e}")
print(f"\n  F(2,3)->F(4,3) per-tile-step error ratio:")
print(f"    fp32: median {st.median(tax32):.1f}x  range [{min(tax32):.1f}, {max(tax32):.1f}]  (n={len(tax32)})")
print(f"    fp16: median {st.median(tax16):.1f}x  range [{min(tax16):.1f}, {max(tax16):.1f}]  (n={len(tax16)})")

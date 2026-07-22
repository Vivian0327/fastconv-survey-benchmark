# -*- coding: utf-8 -*-
"""M2: (a) cuDNN algorithm introspection via profiler kernel names;
       (b) empirical Winograd stability tax: explicit F(2,3)/F(4,3)
           error growth across fp32/fp16 vs fp64 direct reference.
"""
import math, os, re, subprocess, sys, torch
import torch.nn.functional as F

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# ---------------- (a) which algorithm does cuDNN actually pick? -------------
# Strategy: re-exec this script per config with the legacy v7 cuDNN API forced
# (TORCH_CUDNN_V8_API_DISABLED=1) and CUDNN_LOGINFO_DBG=1; the v7 log prints
# the chosen algo enum, which is human-readable:
ALGO_FWD = {0: "IMPLICIT_GEMM", 1: "IMPLICIT_PRECOMP_GEMM", 2: "GEMM",
            3: "DIRECT", 4: "FFT", 5: "FFT_TILING", 6: "WINOGRAD",
            7: "WINOGRAD_NONFUSED"}

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

print("=== (a) cuDNN algorithm introspection (v7 API log) ===")
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
                    sorted(notes.items(), key=lambda kv: -kv[1])) or "none (GEMM-family)"
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

print("\n=== (b) Winograd stability tax (r=3, valid conv, vs fp64 direct) ===")
torch.manual_seed(0)
N, C, K, H = 8, 64, 64, 50                               # 50 -> divisible tiles
x64 = torch.randn(N, C, H, H, device=dev, dtype=torch.float64)
w64 = torch.randn(K, C, 3, 3, device=dev, dtype=torch.float64) / (3 * math.sqrt(C))
ref = F.conv2d(x64, w64)                                 # valid, fp64

def rel_err(y, m):
    r0 = ref[..., :y.shape[-2], :y.shape[-1]]
    return ((y.double() - r0).abs().max() / r0.abs().max()).item()

# sanity: fp64 winograd must match to ~1e-12 (construction check)
for name, mats_fn, m in [("F(2,3)", mats_f23, 2), ("F(4,3)", mats_f43, 4)]:
    y = winograd_conv(x64, w64, mats_fn(torch.float64), m)
    print(f"  sanity fp64 {name}: err={rel_err(y, m):.2e}")

print(f"\n  {'path':14s} {'fp32':>10s} {'fp16':>10s}")
for dt, col in [(torch.float32, 0), (torch.float16, 1)]:
    pass
rows = {}
for name, fn in [
    ("im2col", lambda xx, ww: conv_ref(xx, ww)),
    ("Wino F(2,3)", lambda xx, ww: winograd_conv(xx, ww, mats_f23(xx.dtype), 2)),
    ("Wino F(4,3)", lambda xx, ww: winograd_conv(xx, ww, mats_f43(xx.dtype), 4)),
]:
    rows[name] = []

def conv_ref(xx, ww):                                    # plain conv path
    return F.conv2d(xx, ww)

for dt in (torch.float32, torch.float16):
    x, w = x64.to(dt), w64.to(dt)
    rows["im2col"].append(rel_err(F.conv2d(x, w), 0))
    rows["Wino F(2,3)"].append(rel_err(winograd_conv(x, w, mats_f23(dt), 2), 2))
    rows["Wino F(4,3)"].append(rel_err(winograd_conv(x, w, mats_f43(dt), 4), 4))

for name, (e32, e16) in rows.items():
    print(f"  {name:14s} {e32:10.2e} {e16:10.2e}")

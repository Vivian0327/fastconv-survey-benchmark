# -*- coding: utf-8 -*-
"""Second-order amortization model (review item M2): FFT speedup over
direct as a function of kernel size r, for amortization factors A.
mu_eff = pointwise + transform/(m^2 * A), transform = gamma * t^2 * log2(t)
(operation-count model, gamma=5). Measured dots from M1/M4 overlay.
"""
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.7,
                     "axes.edgecolor": "#666", "pdf.fonttype": 42})
C_F, C_D = "#D55E00", "#0072B2"
GAMMA, TILES = 5.0, [8, 16, 32, 64, 128]

def speedup(r, A):
    best = 0.0
    for t in TILES:
        m = t - r + 1
        if m <= 0 or t > 4 * r:
            continue
        mu = 4 * t * (t // 2 + 1) / m**2 + GAMMA * t * t * np.log2(t) / (m * m * A)
        best = max(best, r * r / mu)
    return best

R = np.arange(3, 32, 2)
fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
for A, ls, lab in [(np.inf, "-", r"$A=\infty$ (ideal)"), (64, "--", "$A=64$"),
                   (8, "-.", "$A=8$"), (1, ":", "$A=1$ (depthwise)")]:
    ax.plot(R, [speedup(r, A) for r in R], ls, color=C_F, lw=1.7, label=lab)
# measured: dense explicit FFT / explicit im2col (M1); depthwise FFT / direct (M4)
ax.plot([3, 7, 13, 31], [0.18, 0.75, 2.6, 8.7], "o", color=C_D, ms=6,
        mec="white", mew=0.8, label="measured dense", zorder=5)
ax.plot([3, 7, 31], [0.18, 0.40, 1.16], "s", color="#333", ms=6,
        mec="white", mew=0.8, label="measured depthwise", zorder=5)
ax.axhline(1, color="#999", lw=1, ls=(0, (4, 3)))
ax.set_yscale("log"); ax.set_ylim(0.08, 400); ax.set_xlim(2, 32.5)
ax.set_xticks([3, 7, 13, 21, 31])
ax.set_yticks([0.1, 1, 10, 100])
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:g}$\\times$"))
ax.set_xlabel("kernel size $r$")
ax.set_ylabel("FFT speedup over direct")
ax.grid(True, axis="y", color="#ddd", lw=0.5); ax.set_axisbelow(True)
# compact 2-column legend in the empty lower-right region
ax.legend(fontsize=6.8, loc="lower right", ncol=2, framealpha=0.95,
          edgecolor="#ccc", borderpad=0.4, columnspacing=0.9,
          handlelength=1.9, handletextpad=0.5)
out = Path(__file__).parent
fig.savefig(out / "fig_amortization.pdf"); fig.savefig(out / "fig_amortization.png", dpi=200)
print("saved fig_amortization.{pdf,png}")
for r in (7, 31):
    print(f"r={r}: A=inf {speedup(r,np.inf):.1f}  A=64 {speedup(r,64):.1f}  A=8 {speedup(r,8):.1f}  A=1 {speedup(r,1):.2f}")

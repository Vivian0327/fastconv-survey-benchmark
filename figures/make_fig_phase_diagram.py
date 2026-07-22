# -*- coding: utf-8 -*-
"""Fig. phase-diagram: kernel size vs best fast-convolution algorithm.
Amortized model (see S2_Complexity_Anchor_Table.md). Reproducible: python make_fig_phase_diagram.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.linewidth": 0.7, "axes.edgecolor": "#666666",
    "xtick.color": "#666666", "ytick.color": "#666666",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "pdf.fonttype": 42,  # editable text in PDF
})

# Okabe-Ito (CVD-safe)
C_WINO, C_FFT, C_GRAY = "#0072B2", "#D55E00", "#8a8a8a"

# ---------------- model ----------------
def wino_speedup(m, r):            # F(mxm, rxr), t = m+r-1
    t = m + r - 1
    return (r * r) * (m * m) / (t * t)

def fft_speedup(t, r, c=4):        # overlap-add, rFFT, c real mults per cplx product
    m = t - r + 1
    return (r * r) * (m * m) / (c * t * (t // 2 + 1)) if m > 0 else 0.0

R = np.arange(3, 32, 2)

# Winograd, numerically safe: best m>=2 with tile t<=8 (fp32 practical limit)
wino_safe_r, wino_safe_y = [], []
for r in R:
    cand = [wino_speedup(m, r) for m in range(2, 9) if m + r - 1 <= 8]
    if cand:
        wino_safe_r.append(r); wino_safe_y.append(max(cand))

# Winograd F(4,r) unconstrained (numerically unstable beyond t=8)
wino_unst_r = [r for r in R if r >= 7]
wino_unst_y = [wino_speedup(4, r) for r in wino_unst_r]

# FFT with practical tile constraint t <= 4r (power-of-two tiles, consistent with Table 2)
TILES = [8, 16, 32, 64, 128]
fft_y = [max(fft_speedup(t, r) for t in TILES if t <= 4 * r) for r in R]

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")

# regime bands (very light tints; labels at top)
ax.axvspan(2.0, 4.5, color=C_WINO, alpha=0.07, lw=0)
ax.axvspan(4.5, 7.5, color="#000000", alpha=0.045, lw=0)
ax.axvspan(7.5, 32.5, color=C_FFT, alpha=0.06, lw=0)
for x, s, c in [(3.2, "Winograd\nregime", C_WINO), (6.0, "transi-\ntion", "#555555"),
                (19, "FFT / spectral regime", C_FFT)]:
    ax.text(x, 240, s, ha="center", va="top", fontsize=8, color=c, fontweight="bold")

# curves
ax.axhline(1.0, color=C_GRAY, lw=1.4, ls=(0, (4, 3)))
ax.text(31.6, 1.0, "direct / im2col", color="#666666", fontsize=8.5, va="center", ha="right",
        bbox=dict(fc="white", ec="none", pad=1.2))

ax.plot(wino_safe_r, wino_safe_y, "-", color=C_WINO, lw=2.0, marker="^",
        ms=6.5, mfc=C_WINO, mec="white", mew=1.0, zorder=5,
        label="Winograd, numerically safe (tile $t\\leq 8$)")
ax.plot(wino_unst_r, wino_unst_y, ":", color=C_WINO, lw=1.8, marker="^",
        ms=5.5, mfc="white", mec=C_WINO, mew=1.2, zorder=4,
        label="Winograd $F(4{\\times}4,\\,r{\\times}r)$, unstable ($t>8$)")
ax.plot(R, fft_y, "-", color=C_FFT, lw=2.0, marker="o",
        ms=5.5, mfc=C_FFT, mec="white", mew=1.0, zorder=5,
        label="FFT overlap-add (practical tile $t\\leq 4r$)")

# annotations
ax.annotate("crossover $r{=}5$:\nboth $6.25\\times$", xy=(5, 6.25), xytext=(3.5, 2.2),
            fontsize=8.5, ha="center", color="#333333",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
ax.annotate("", xy=(7, 3.06), xytext=(7, 7.84),
            arrowprops=dict(arrowstyle="->", color=C_WINO, lw=1.1))
ax.text(7.35, 4.7, "numerical-\nstability tax", fontsize=8, color=C_WINO, va="center")
ax.text(29.6, 175, f"${int(fft_y[-1])}\\times$", fontsize=9.5, color=C_FFT,
        fontweight="bold", ha="center")

# network-era markers
for r, name in [(3, "ResNet/VGG"), (7, "ConvNeXt"), (13, "UniRepLKNet"), (31, "RepLKNet")]:
    ax.axvline(r, color="#999999", lw=0.6, ls=(0, (1, 3)), zorder=1)
    ax.text(r, 0.60, name, rotation=0, ha="center", va="top", fontsize=7.8, color="#777777")

# axes
ax.set_yscale("log")
ax.set_ylim(0.5, 300)
ax.set_xlim(2, 32.5)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:g}$\\times$"))
ax.set_xticks([3, 5, 7, 9, 13, 17, 21, 25, 31])
ax.set_xlabel("kernel size  $r$  (kernel $r\\times r$, stride 1)")
ax.set_ylabel("theoretical speedup over direct convolution")
ax.grid(True, which="major", axis="y", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8.3, framealpha=0.95, edgecolor="#cccccc",
          bbox_to_anchor=(0.012, 0.845))

from pathlib import Path
out = Path(__file__).parent
fig.savefig(out / "fig_phase_diagram.pdf")
fig.savefig(out / "fig_phase_diagram.png", dpi=220)
print("saved:", out / "fig_phase_diagram.pdf", "and .png")

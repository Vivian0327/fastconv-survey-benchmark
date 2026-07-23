# Fast Convolution Survey — Benchmark Suite

Companion code for the survey paper:

> **Fast Convolution Algorithms for Deep Neural Networks: A Survey of
> Methods, Numerical Behavior, and Hardware Mappings**
> Weiwei Wang(under review, 2026)

This repository contains every measurement script and raw result behind the
paper's Section VII (cross-cutting comparison), plus the scripts that generate
the paper's model figures. All experiments run on commodity hardware with
stock PyTorch — no custom CUDA kernels required.

## What is measured

| Script | Experiment | Validates |
|---|---|---|
| `benchmarks/m1_regime_sweep.py` | GPU latency/memory/error sweep: cuDNN-best vs explicit im2col vs explicit FFT over kernel sizes 3–31 | The kernel-size "phase diagram" (paper Fig. 1) |
| `benchmarks/m1b_winograd_timing.py` | Explicit Winograd F(2,3)/F(4,3) timing, output-normalized (ns/output) | Within-family tile scaling (measured 1.66× vs theoretical 1.78×); explicit Winograd is slower per output than im2col → fusion necessity |
| `benchmarks/m2_numerics_introspect.py` | (a) cuDNN engine introspection via `CUDNN_LOGINFO_DBG` numerical notes; (b) Winograd tile-size error growth across 12 shape/seed configs | Vendor selections align with the regime structure; the "numerical-stability tax" (fp32 median 19.7×/tile step, range 17–27×; ~1.3% error for fp16 F(4,3)) |
| `benchmarks/m3_cpu_sweep.py` | CPU sweep: oneDNN vs native vs explicit im2col vs explicit FFT | Same regime structure on CPU; oneDNN's missing FFT engine (explicit FFT beats it 1.84× at r=31) |
| `benchmarks/m4_depthwise_gap.py` | Depthwise stressor: MobileNet/ConvNeXt/RepLKNet layers | The depthwise amortization gap (explicit-FFT advantage over the spatial baseline falls from 8.5× dense to 1.16× depthwise at r=31) |

All timing scripts share `benchmarks/bench_common.py` (median + IQR over
50 warmed-up runs × 3 seeds; two honest error metrics — relative RMS and
peak-normalized max error — against an fp64 direct reference).

Figure generators (`figures/`):
- `make_fig_phase_diagram.py` — amortized-complexity phase diagram (paper Fig. 1)
- `make_fig_amortization.py` — second-order amortization model with measured overlay (paper Fig. 2)

Raw measurements used in the paper are in `results/` (median + IQR over
50 warmed-up runs × 3 seeds; `env_info.txt` records the exact software/driver
versions; `m2_results.txt` is the cuDNN introspection log). See the paper's
Section VII-A for the full protocol and Section VII-F for scope limits — these
are illustrative single-platform measurements, not a cross-architecture
validation.

## Hardware/software used in the paper

- GPU: NVIDIA GeForce GTX 1660 Ti (Turing TU116, no tensor cores, 6 GB), driver CUDA 13.2
- CPU: Intel Core i7-10700 (8C/16T, AVX2)
- PyTorch 2.5.1 + cu121 (bundled cuDNN 9.1), Python 3.11, Windows 11

Results on other platforms will differ in absolute numbers; the paper's claims
concern orderings and regime boundaries, which we encourage you to re-test.

## Quick start

```bash
pip install -r requirements.txt
python benchmarks/m1_regime_sweep.py --quick     # ~1 min smoke test
python benchmarks/m1_regime_sweep.py             # full run, writes m1_results.csv
python benchmarks/m2_numerics_introspect.py      # introspection + stability tax
python benchmarks/m3_cpu_sweep.py                # CPU (no GPU required)
python benchmarks/m4_depthwise_gap.py            # depthwise gap
python figures/make_fig_phase_diagram.py         # regenerate paper Fig. 1
python figures/make_fig_amortization.py          # regenerate paper Fig. 2
```

All explicit convolution paths are validated against an fp64 direct-convolution
reference (~1e-15 relative error in fp64) before being timed.

## Citation

```bibtex
@article{wang2026fastconv,
  title   = {Fast Convolution Algorithms for Deep Neural Networks: A Survey of
             Methods, Numerical Behavior, and Hardware Mappings},
  author  = {Wang, Weiwei and DeBrunner, Victor and DeBrunner, Linda},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

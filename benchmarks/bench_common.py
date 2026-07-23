# -*- coding: utf-8 -*-
"""Shared measurement utilities for the fast-convolution benchmark suite.

Honest error metrics (see review response, 2026-07):
  rel_rms   = ||y - ref||_2 / ||ref||_2         (relative RMS / normalized L2)
  rel_linf  = max|y - ref| / max|ref|           (peak-normalized max error;
              NOT elementwise max relative error, which is ill-defined where
              ref ~ 0 -- peak normalization is the standard DSP convention)
Both are computed in fp64 against an fp64 direct-convolution reference.
"""
import statistics
import torch


def rel_rms(y, ref):
    d = (y.double() - ref).pow(2).sum().sqrt()
    return (d / ref.pow(2).sum().sqrt().clamp_min(1e-30)).item()


def rel_linf(y, ref):
    return ((y.double() - ref).abs().max()
            / ref.abs().max().clamp_min(1e-30)).item()


def time_gpu(fn, warmup=10, iters=50):
    """Return (median_ms, iqr_ms) over `iters` events after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts, (a, b) = [], (torch.cuda.Event(True), torch.cuda.Event(True))
    for _ in range(iters):
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    q1, q3 = ts[len(ts) // 4], ts[(3 * len(ts)) // 4]
    return statistics.median(ts), q3 - q1


def time_cpu(fn, warmup=10, iters=50):
    """Return (median_ms, iqr_ms). Same protocol as time_gpu for parity."""
    import time
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    q1, q3 = ts[len(ts) // 4], ts[(3 * len(ts)) // 4]
    return statistics.median(ts), q3 - q1

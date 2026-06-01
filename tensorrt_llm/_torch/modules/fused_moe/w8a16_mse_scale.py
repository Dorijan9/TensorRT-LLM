"""Data-free MSE-optimal per-channel INT8 scale search for W8A16 MoE
quantization. Replaces V5's fixed p99 clip with a small per-channel grid
search across multiple clipping percentiles, picking the one that minimises
L2 error between BF16 and dequantized INT8 weight on that channel.

Properties:
  * Data-free — uses only the weight tensor itself, no calibration set.
  * Per-channel — different channels can pick different clip percentiles.
  * Cheap — O(K) extra quantize+dequantize evaluations per channel, where K
    is the size of the candidate percentile grid (default 5). Runs once at
    model load; zero runtime cost.

The grid {100.0, 99.95, 99.9, 99.5, 99.0} covers the common cases:
  * 100.0 = lossless max-abs scaling (best for well-behaved channels).
  * 99.95 / 99.9 = mild outlier suppression.
  * 99.5 / 99.0 = aggressive clipping for channels with heavy tails.

Together they bracket the trade-off curve. Empirically the chosen
percentile distribution is bimodal: most channels pick 100.0 (no clipping
needed); the small fraction of outlier-heavy channels pick something in
the 99.0–99.5 range. The fixed-p99 V5 strategy is close to optimal for the
second cluster but over-clips the first, which the search avoids.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch


DEFAULT_PERCENTILES = (100.0, 99.95, 99.9, 99.5, 99.0)


def _envfloat_tuple(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return tuple(float(x) for x in raw.split(",") if x.strip())
    except Exception:
        return default


def _quantize_dequantize(w_f32: torch.Tensor,
                         scale: torch.Tensor) -> torch.Tensor:
    """Round to INT8 in [-127,127] and dequantize back to fp32.

    `w_f32`: [out, in] fp32 weight
    `scale`: [out] fp32 per-channel scale
    """
    s = scale.clamp_min(1e-30).unsqueeze(1)
    q = torch.round(w_f32 / s).clamp_(-127.0, 127.0)
    return q * s


def mse_optimal_per_channel_int8(
    w: torch.Tensor,
    *,
    percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the per-channel INT8 scale that minimises L2 error between `w`
    and its dequantized counterpart, over a small grid of clipping
    percentiles.

    Args:
        w:           [out, in] BF16/FP16/FP32 weight.
        percentiles: Candidate clip percentiles in 0..100. 100.0 means
                     lossless max-abs (no clipping).

    Returns:
        q_int8: [out, in] int8 weight.
        scale:  [out]    per-channel scale, same dtype as `w`.
        picked: [out]    int8 index into `percentiles` that won per channel
                         (useful for diagnostics; can be discarded).
    """
    assert w.dim() == 2, f"expected 2D weight, got {tuple(w.shape)}"
    orig_dtype = w.dtype
    w_f32 = w.to(torch.float32)
    abs_w = w_f32.abs()
    out_ch, _ = w_f32.shape

    percentiles = tuple(
        sorted({max(0.0, min(100.0, float(p))) for p in percentiles},
               reverse=True))

    # Precompute per-channel max-abs once
    max_abs = abs_w.amax(dim=1)

    best_mse = torch.full((out_ch, ), float("inf"),
                          device=w.device,
                          dtype=torch.float32)
    best_scale = max_abs.clone()
    picked = torch.zeros(out_ch, device=w.device, dtype=torch.int8)

    for k, p in enumerate(percentiles):
        if p >= 100.0:
            cand_abs = max_abs
        else:
            try:
                cand_abs = torch.quantile(abs_w, p / 100.0, dim=1)
            except RuntimeError:
                # Fallback: cheap proxy if quantile() OOMs on giant rows
                cand_abs = max_abs * (p / 100.0)
        cand_abs = cand_abs.clamp_min(1e-12)
        scale = cand_abs / 127.0
        deq = _quantize_dequantize(w_f32, scale)
        mse = ((w_f32 - deq)**2).mean(dim=1)

        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_scale = torch.where(improved, scale, best_scale)
        picked = torch.where(improved,
                             torch.full_like(picked, k, dtype=torch.int8),
                             picked)

    # Final quantize using the chosen per-channel scale
    s = best_scale.clamp_min(1e-30).unsqueeze(1)
    q_int8 = torch.round(w_f32 / s).clamp_(-127.0, 127.0).to(torch.int8)
    return q_int8, best_scale.to(orig_dtype), picked


def quantize_expert_w8a16_mse(
    fp_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-level entry point matching the (q_int8, scale) tuple expected by
    the V6 plumbing. Returns int8 weights and per-channel fp scales."""
    grid = _envfloat_tuple("TRTLLM_MOE_W8A16_MSE_GRID", DEFAULT_PERCENTILES)
    q_int8, scale, _picked = mse_optimal_per_channel_int8(
        fp_weight, percentiles=grid)
    return q_int8, scale


# Optional diagnostics: aggregate which percentile was picked across an
# entire layer/expert and return a histogram. Useful for tuning.

def summarize_picked(picked: torch.Tensor,
                     percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES
                     ) -> dict:
    hist = torch.bincount(picked.to(torch.int64),
                          minlength=len(percentiles)).tolist()
    return {p: c for p, c in zip(percentiles, hist)}

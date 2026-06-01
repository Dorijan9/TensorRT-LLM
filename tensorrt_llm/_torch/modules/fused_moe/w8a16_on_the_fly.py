# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
On-the-fly W8A16 (INT8 weight-only per-channel) MoE method.

This is the *correct* implementation of the W8A16 MoE patch — the
previous "shim" approach (subclass + monkey-patched is_weight_only())
failed at weight loading time because:

  1. ``INT8WoqPerChannelFusedMoEMethod.create_weights`` checks the
     checkpoint's ``quant_config.layer_quant_mode.is_int8_weight_only()``
     before allocating INT8 buffers. The Nemotron-3-Nano-30B-A3B-BF16
     checkpoint has ``quant_config == None`` (or BF16), so even after
     monkey-patching the shim's own ``_LayerMode``, the parent's
     dispatch went down the wrong path.

  2. ``load_expert_w3_w1_weight`` passes the source weight tensor
     directly to ``preprocess_weights_for_mixed_gemm(..., weight_dtype=
     torch.int8, ...)``. The source on a BF16 checkpoint is BF16; the
     preprocessor reinterprets each BF16 as 2× INT8, producing a
     buffer of half the expected size. That is the 3712 vs 1856
     mismatch the user observed.

  3. ``load_quant_scales`` reads ``f"{expert_id}.w{i}.weight_scale"``
     keys from the checkpoint dict. These keys do not exist in a BF16
     checkpoint and raise KeyError.

  4. ``CutlassFusedMoE.has_int8_woq_per_channel`` returns True only
     when ``quant_config.layer_quant_mode.is_int8_weight_only()``. The
     shim never flipped this, so ``run_moe`` did not pass
     ``use_int8_woq_per_channel=True`` to ``torch.ops.trtllm.fused_moe``
     and the kernel would have treated the INT8 buffer as BF16 even if
     loading had succeeded.

The proper fix is a **new** ``FusedMoEMethodBase`` subclass that:

  * Allocates INT8 buffers + BF16 scale buffers in ``create_weights``
    *without* consulting ``quant_config``.
  * Reads **BF16** source weights, computes per-row MSE-optimal scales
    in float32, quantizes inline to INT8, **then** calls
    ``preprocess_weights_for_mixed_gemm``.
  * Synthesises scales locally — no checkpoint lookups.
  * Force-flips ``module.has_int8_woq_per_channel`` to ``True`` on the
    instance so ``run_moe`` dispatches to the INT8 weight-only kernel.

Activated by env var ``TRTLLM_MOE_W8A16_ONTHEFLY=1`` (the dispatch
hook in ``fused_moe_cutlass.py`` checks this before falling through to
the standard quant_method selection).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.functional import \
    preprocess_weights_for_mixed_gemm

from ..linear import TensorParallelMode, load_weight_shard
from .quantization import (EplbSupportStatus, FusedMoEMethodBase,
                           FusedMoEQuantScalesINT8WoqPerChannel)

_ENABLE_ENV = "TRTLLM_MOE_W8A16_ONTHEFLY"
_PERCENTILE_ENV = "TRTLLM_W8A16_PERCENTILE_GRID"
_DEFAULT_GRID: Tuple[float, ...] = (100.0, 99.95, 99.9, 99.5, 99.0)


def w8a16_on_the_fly_enabled() -> bool:
    """Read the env var. Allow 1/true/yes/on; default off."""
    raw = os.environ.get(_ENABLE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _percentile_grid() -> Tuple[float, ...]:
    raw = os.environ.get(_PERCENTILE_ENV, "").strip()
    if not raw:
        return _DEFAULT_GRID
    try:
        return tuple(float(x) for x in raw.split(","))
    except ValueError:
        logger.warning(f"Invalid {_PERCENTILE_ENV}={raw!r}; using default grid")
        return _DEFAULT_GRID


# ---------------------------------------------------------------------------
# MSE-optimal per-row INT8 scale search (data-free).
#
# The MoE expert weight comes in with shape [K, N] where N is the output-row
# axis. After ``preprocess_weights_for_mixed_gemm`` the layout becomes
# packed INT8 with row=K, col=N. The per-row scale is stored as one BF16
# value per output channel (N values per expert).
# ---------------------------------------------------------------------------


def _mse_optimal_per_row_quantize(
    weight_bf16_kn: torch.Tensor,
    *,
    percentile_grid: Sequence[float] = _DEFAULT_GRID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [K, N] BF16 weight to [K, N] INT8 with per-output-channel
    (i.e. per-N) BF16 scales chosen by MSE-optimal percentile clip search.

    Returns (int8_kn, scale_n) where:
        int8_kn.dtype == torch.int8 and shape == [K, N]
        scale_n.dtype == weight_bf16_kn.dtype and shape == [N]

    Pure float32 math on the input device; ~10-50 ms per expert on A100.
    """
    assert weight_bf16_kn.ndim == 2, (
        f"Expected 2-D [K,N] weight, got shape {tuple(weight_bf16_kn.shape)}")
    w = weight_bf16_kn.float()
    K, N = w.shape
    abs_w = w.abs()

    pgrid = torch.tensor(percentile_grid, dtype=torch.float32,
                         device=w.device) / 100.0
    # Per-column percentile clip candidates.
    try:
        clips = torch.quantile(abs_w, pgrid, dim=0)  # [P, N]
    except RuntimeError:
        clips = torch.quantile(abs_w.cpu(), pgrid.cpu(), dim=0).to(w.device)
    scale_pc = (clips / 127.0).clamp_min(1e-12)  # [P, N]

    w_norm = w.norm(dim=0).clamp_min(1e-12)  # [N]
    rel = torch.empty(len(percentile_grid), N, dtype=torch.float32,
                      device=w.device)
    for pi in range(len(percentile_grid)):
        s = scale_pc[pi]  # [N]
        q = torch.round(w / s).clamp_(-127.0, 127.0)
        dq = q * s
        rel[pi] = (w - dq).norm(dim=0) / w_norm

    best_idx = rel.argmin(dim=0)  # [N]
    best_scale = scale_pc.gather(0, best_idx.unsqueeze(0)).squeeze(0)  # [N]

    # Final quantize with the chosen scales.
    int8_kn = torch.round(w / best_scale).clamp_(-127.0, 127.0).to(torch.int8)
    return int8_kn, best_scale.to(weight_bf16_kn.dtype)


# ---------------------------------------------------------------------------
# MoE method class.
# ---------------------------------------------------------------------------


class OnTheFlyINT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):
    """W8A16 MoE method that quantizes the BF16 checkpoint to INT8 at
    weight-load time, with no precomputed scales in the checkpoint.

    Mirrors ``INT8WoqPerChannelFusedMoEMethod`` in terms of buffer
    layout and ``setup_quant_scales``, so once weights are loaded the
    standard ``torch.ops.trtllm.fused_moe`` INT8 weight-only kernel is
    used unmodified.
    """

    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

    def __init__(self) -> None:
        super().__init__()
        self._percentile_grid = _percentile_grid()
        # Diagnostic counters surfaced via logger after weight loading.
        self._n_tensors_quantized = 0

    # -- buffer allocation ---------------------------------------------------

    def create_weights(self, module: torch.nn.Module):
        from tensorrt_llm._utils import get_sm_version

        module.sm_version = get_sm_version()
        # The CUTLASS mixed-gemm preprocessor expects the *Ampere*
        # layout on SM>=90 too — match the existing INT8 woq path.
        module.sm_version = (80 if module.sm_version >= 90 else
                             module.sm_version)
        module.preprocessor = preprocess_weights_for_mixed_gemm

        # The packed INT8 layout produced by the preprocessor is
        # [E, K, N] viewed as int8; same shape as the existing
        # INT8WoqPerChannelFusedMoEMethod path.
        weight_dtype = torch.int8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.hidden_size,
                              module.intermediate_size_per_partition * 2)
        w2_weight_shape = (module.expert_size_per_partition,
                           module.intermediate_size_per_partition,
                           module.hidden_size)

        fc31_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition * 2,
                        dtype=module.dtype),
            requires_grad=False,
        )
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        fc2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.hidden_size,
                        dtype=module.dtype),
            requires_grad=False,
        )
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        # Allocate w3_w1_weight / w2_weight as INT8.
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self._online_eplb_not_supported(module)
        self.setup_quant_scales(module)

        # ------------------------------------------------------------------
        # Force the dispatch flag on the module instance so that
        # ``CutlassFusedMoE.run_moe`` passes
        # ``use_int8_woq_per_channel=True`` to ``torch.ops.trtllm.fused_moe``.
        #
        # The base CutlassFusedMoE class exposes ``has_int8_woq_per_channel``
        # as an @property that consults ``quant_config``. We can't override
        # @property values via instance __dict__, so we patch the *class*
        # to honour an instance-level override attribute.
        # ------------------------------------------------------------------
        _ensure_has_int8_woq_per_channel_instance_override(module)
        module._w8a16_onthefly_active = True

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesINT8WoqPerChannel(
            fc31_weight_scale=module.fc31_weight_scale,
            fc2_weight_scale=module.fc2_weight_scale,
        )

    # -- expert weight loading ---------------------------------------------

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """Load BF16 w1/w3, quantize on the fly, store INT8 + scale."""
        assert module.dtype in (torch.float16, torch.bfloat16), (
            f"On-the-fly W8A16 requires BF16/FP16 activations, "
            f"got dtype={module.dtype}")

        w1_shard = load_weight_shard(w1_weight, module.tp_size,
                                     module.tp_rank,
                                     TensorParallelMode.COLUMN)
        w3_shard = load_weight_shard(w3_weight, module.tp_size,
                                     module.tp_rank,
                                     TensorParallelMode.COLUMN)
        # The existing INT8 path concatenates w3 then w1 along dim=0
        # (the "N" dimension when viewed as [N, K]). Keep the same
        # ordering so fc31_weight_scale's column layout matches.
        w31 = torch.cat([w3_shard, w1_shard], dim=0)  # [N, K] in HF layout

        # Move to the destination device so the MSE search and
        # quantize/preprocess all run on GPU (vastly faster).
        w31_dev = w31.to(dst_w3_w1_weight.device, dtype=module.dtype,
                         non_blocking=True)

        # Transpose to [K, N] for per-N-column scale search & for the
        # preprocessor (which wants column-major / [K, N]).
        w31_kn = w31_dev.T.contiguous()

        int8_kn, scale_n = _mse_optimal_per_row_quantize(
            w31_kn, percentile_grid=self._percentile_grid)

        # Preprocessor wants INT8 input in [K, N] layout. Output is a
        # packed buffer that views as int8 with the same total bytes as
        # ``dst_w3_w1_weight``.
        packed = module.preprocessor(int8_kn.contiguous(), torch.int8,
                                     module.dtype, module.sm_version)
        packed = packed.contiguous()

        # dst is typed int8 already; copy as int8.
        dst_w3_w1_weight.copy_(packed.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

        # Find this expert's slice in fc31_weight_scale and write it.
        # dst_w3_w1_weight is module.w3_w1_weight.data[expert_idx];
        # locate the matching expert_idx by pointer identity to keep
        # the API surface the same as the existing INT8 path.
        expert_idx = _find_expert_idx(dst_w3_w1_weight,
                                      module.w3_w1_weight.data)
        module.fc31_weight_scale.data[expert_idx].copy_(
            scale_n.to(module.fc31_weight_scale.dtype), non_blocking=True)
        self._n_tensors_quantized += 1

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """Load BF16 w2, quantize on the fly, store INT8 + scale."""
        assert module.dtype in (torch.float16, torch.bfloat16), (
            f"On-the-fly W8A16 requires BF16/FP16 activations, "
            f"got dtype={module.dtype}")

        w2_shard = load_weight_shard(w2_weight, module.tp_size,
                                     module.tp_rank, TensorParallelMode.ROW)
        w2_dev = w2_shard.to(dst_w2_weight.device, dtype=module.dtype,
                             non_blocking=True)
        # HF stores w2 as [N, K] where N=hidden_size, K=intermediate.
        # The preprocessor wants [K, N], so transpose.
        w2_kn = w2_dev.T.contiguous()

        int8_kn, scale_n = _mse_optimal_per_row_quantize(
            w2_kn, percentile_grid=self._percentile_grid)

        packed = module.preprocessor(int8_kn.contiguous(), torch.int8,
                                     module.dtype, module.sm_version)
        packed = packed.contiguous()
        dst_w2_weight.copy_(packed.view(dst_w2_weight.dtype),
                            non_blocking=True)

        expert_idx = _find_expert_idx(dst_w2_weight, module.w2_weight.data)
        module.fc2_weight_scale.data[expert_idx].copy_(
            scale_n.to(module.fc2_weight_scale.dtype), non_blocking=True)
        self._n_tensors_quantized += 1

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Scales were synthesised inline above; nothing to read from
        # the checkpoint dict. (load_quant_scales is the hook the base
        # class calls after all expert weights are loaded.)
        logger.info_once(
            f"W8A16 on-the-fly: quantized {self._n_tensors_quantized} "
            f"MoE expert weight tensors on layer "
            f"{getattr(module, 'layer_idx', '?')}",
            key=f"w8a16_onthefly_layer_{getattr(module, 'layer_idx', '?')}",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_expert_idx(dst_slice: torch.Tensor,
                     full_tensor: torch.Tensor) -> int:
    """Recover the expert index a dst slice belongs to.

    ``load_expert_weights_to_dst`` passes ``dst_w3_w1_weights_tensor
    [expert_idx]`` into the loaders. The slice is a contiguous view
    into the parent (E, *expert_shape) tensor starting at offset
    ``expert_idx * numel_per_expert`` in the underlying storage.

    Recovering ``expert_idx`` from the slice alone is robust to the
    caller passing either ``module.w3_w1_weight.data`` *or* a separate
    ``local_shared_w3_w1_tensors`` buffer — both are contiguous on
    their first axis, so ``storage_offset() // numel()`` gives the
    expert index regardless. The ``full_tensor`` argument is retained
    only as a sanity fallback.
    """
    n = dst_slice.numel()
    if n == 0:
        return 0
    # Primary path: storage_offset is in elements, numel is per-expert
    # element count. For a contiguous parent on dim 0, this is exact.
    try:
        return int(dst_slice.storage_offset() // n)
    except Exception:  # pragma: no cover - defensive
        pass
    base_ptr = full_tensor.data_ptr()
    slice_ptr = dst_slice.data_ptr()
    stride_bytes = full_tensor.stride(0) * full_tensor.element_size()
    if stride_bytes <= 0:
        return 0
    return int((slice_ptr - base_ptr) // stride_bytes)


def _ensure_has_int8_woq_per_channel_instance_override(
        module: torch.nn.Module) -> None:
    """Make ``module.has_int8_woq_per_channel`` honour an instance
    attribute when set.

    The base ``CutlassFusedMoE`` class defines ``has_int8_woq_per_channel``
    as a Python @property that reads ``self.quant_config``. We can't
    override a class-level @property via instance __dict__. Instead, we
    monkey-patch the *class* once (idempotent) so the property checks
    for an instance attribute ``_force_int8_woq_per_channel`` and
    short-circuits when present.

    Doing this at the class level is global, but it's keyed on a
    private attribute name that only this module sets, so it's safe to
    apply unconditionally.
    """
    cls = type(module)
    sentinel_attr = "_w8a16_onthefly_property_patched"
    # Walk MRO looking for the class that defines the property.
    target_cls = None
    for c in cls.__mro__:
        if "has_int8_woq_per_channel" in c.__dict__:
            target_cls = c
            break
    if target_cls is None:
        # Property doesn't exist; nothing to patch (older versions).
        module._force_int8_woq_per_channel = True
        return
    if getattr(target_cls, sentinel_attr, False):
        module._force_int8_woq_per_channel = True
        return

    original_prop = target_cls.__dict__["has_int8_woq_per_channel"]
    if not isinstance(original_prop, property):
        # Already overridden as a plain attribute somewhere; just set.
        module._force_int8_woq_per_channel = True
        return

    original_fget = original_prop.fget

    def patched_fget(self):  # type: ignore[no-untyped-def]
        if getattr(self, "_force_int8_woq_per_channel", False):
            return True
        return original_fget(self)

    setattr(target_cls, "has_int8_woq_per_channel", property(patched_fget))
    setattr(target_cls, sentinel_attr, True)
    module._force_int8_woq_per_channel = True

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
On-the-fly W8A16 (INT8 weight-only per-channel) MoE method — v18 / v19.

v19 is identical to v18 in content (the prior v18 commit was a stub with no
edits attached). v18/v19 layer two fixes on top of v17:

  1. **Buffer orientation fix** (Python, this file): the per-expert
     INT8 weight buffer must be ``(hidden, expand_inter)`` after the
     leading expert axis — NOT ``(expand_inter, hidden)``. This
     matches what ``preprocess_weights_for_mixed_gemm`` produces
     ([K=hidden, N=expand_inter] view, packed as int8) and what the
     C++ op reads as ``fc1_expert_weights.sizes()[2]`` for the inter
     dimension (see ``moeOp.cpp::runMoe`` INT8-woq branch).

  2. **moeOp.cpp validation fix** (C++, separate edit): the
     ``mUseINT8WoqPerChannel`` validation branch in both ``runMoe``
     (line ~347) and ``runMoeMinLantency`` (line ~553) now mirrors
     the existing non-woq else-branch by conditioning the ``* 2``
     factor on ``isGatedActivation(base_activation_type)``. Non-gated
     activations (Relu2 for Nemotron-H, Identity, ReLU, SiLU, Gelu)
     pass the ``fc1.inter == fc2.inter`` check; gated activations
     (Swiglu, Geglu, SwigluBias) keep the ``* 2``.

The downstream path is already correct for non-gated INT8-woq:

  * ``moe_kernels.cu:2782`` workspace ``factor = is_gated ? 2 : 1``.
  * ``moe_kernels.cu:2884-2891`` glu intermediate buffer only allocated
    when ``is_gated_activation`` (or fp8).
  * ``moe_kernels.cu:3105`` ``fc1_out_size = is_gated ? 2*inter : inter``.
  * ``moe_kernels.cu:2404-2407`` ``Relu2`` dispatches to
    ``IdentityAdaptor<cutlass::epilogue::thread::Relu2>`` (non-gated).
  * ``moe_gemm_template_dispatch.h:973`` ``Relu2`` epilogue tag wired
    through to ``EpilogueOpDefaultRelu2`` for all (T, WeightType).
  * ``moe_gemm_template_dispatch.h:656`` ``supportsFusedGatedActivation``
    returns false for ``Relu2`` AND for differing T/WeightType, so the
    SM80 INT8-woq path correctly takes the non-fused activation route.

Activation: ``TRTLLM_MOE_W8A16_ONTHEFLY=1`` (default off).
Optional: ``TRTLLM_W8A16_PERCENTILE_GRID="100,99.95,99.9,99.5,99.0"``.

Rebuild scope:
  * Python file changes: hot-applied, no rebuild.
  * moeOp.cpp change: ``libtensorrt_llm.so`` only.
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
# MSE-optimal per-row INT8 scale search (data-free, MSE-clip grid search).
# ---------------------------------------------------------------------------


def _mse_optimal_per_row_quantize(
    weight_bf16_kn: torch.Tensor,
    *,
    percentile_grid: Sequence[float] = _DEFAULT_GRID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [K, N] BF16 weight to [K, N] INT8 with per-N BF16 scales
    chosen by MSE-optimal percentile clip search.
    """
    assert weight_bf16_kn.ndim == 2, (
        f"Expected 2-D [K,N] weight, got shape {tuple(weight_bf16_kn.shape)}")
    w = weight_bf16_kn.float()
    K, N = w.shape
    abs_w = w.abs()
    pgrid = torch.tensor(percentile_grid, dtype=torch.float32,
                         device=w.device) / 100.0
    try:
        clips = torch.quantile(abs_w, pgrid, dim=0)
    except RuntimeError:
        clips = torch.quantile(abs_w.cpu(), pgrid.cpu(), dim=0).to(w.device)
    scale_pc = (clips / 127.0).clamp_min(1e-12)
    w_norm = w.norm(dim=0).clamp_min(1e-12)
    rel = torch.empty(len(percentile_grid), N, dtype=torch.float32,
                      device=w.device)
    for pi in range(len(percentile_grid)):
        s = scale_pc[pi]
        q = torch.round(w / s).clamp_(-127.0, 127.0)
        dq = q * s
        rel[pi] = (w - dq).norm(dim=0) / w_norm
    best_idx = rel.argmin(dim=0)
    best_scale = scale_pc.gather(0, best_idx.unsqueeze(0)).squeeze(0)
    int8_kn = torch.round(w / best_scale).clamp_(-127.0, 127.0).to(torch.int8)
    return int8_kn, best_scale.to(weight_bf16_kn.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_nonempty(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.numel() > 0 and t.shape[0] > 0


def _find_expert_idx(dst_slice: torch.Tensor,
                     full_tensor: torch.Tensor) -> int:
    n = dst_slice.numel()
    if n == 0:
        return 0
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
    """Idempotently patch the @property so an instance flag wins."""
    cls = type(module)
    sentinel_attr = "_w8a16_onthefly_property_patched"
    target_cls = None
    for c in cls.__mro__:
        if "has_int8_woq_per_channel" in c.__dict__:
            target_cls = c
            break
    if target_cls is None:
        module._force_int8_woq_per_channel = True
        return
    if getattr(target_cls, sentinel_attr, False):
        module._force_int8_woq_per_channel = True
        return

    original_prop = target_cls.__dict__["has_int8_woq_per_channel"]
    if not isinstance(original_prop, property):
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


# ---------------------------------------------------------------------------
# MoE method class.
# ---------------------------------------------------------------------------


class OnTheFlyINT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):
    """W8A16 MoE method: quantizes BF16 expert weights to INT8 at load
    time. v18: correct buffer orientation + non-gated support."""

    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

    def __init__(self) -> None:
        super().__init__()
        self._percentile_grid = _percentile_grid()
        self._n_tensors_quantized = 0

    def create_weights(self, module: torch.nn.Module):
        from tensorrt_llm._utils import get_sm_version

        module.sm_version = get_sm_version()
        # Match the existing INT8 woq path which targets the Ampere
        # layout on SM>=90 too.
        module.sm_version = (80 if module.sm_version >= 90 else
                             module.sm_version)
        module.preprocessor = preprocess_weights_for_mixed_gemm

        # --------------------------------------------------------------
        # Buffer layout (v18 fix):
        # ``expand_intermediate_size_per_partition`` = intermediate for
        # non-gated activations (Relu2 — Nemotron-H), 2*intermediate for
        # gated (Swiglu).
        #
        # The packed INT8 buffer that ``preprocess_weights_for_mixed_gemm``
        # produces has shape ``[K=hidden, N=expand_inter]`` per expert,
        # viewed as int8. The C++ op reads
        # ``fc1_expert_weights.sizes()[2]`` as the inter dimension on
        # the INT8-woq path (moeOp.cpp::runMoe). Therefore the per-
        # expert buffer shape must be ``(hidden, expand_inter)`` — NOT
        # ``(expand_inter, hidden)`` as v17 had. The existing
        # INT8WoqPerChannelFusedMoEMethod uses the same orientation.
        # --------------------------------------------------------------
        expand_inter = module.expand_intermediate_size_per_partition
        weight_dtype = torch.int8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.hidden_size,
                              expand_inter)
        # fc2 layout (transposed) — matches the existing INT8-woq path:
        # [E, intermediate, hidden].
        w2_weight_shape = (module.expert_size_per_partition,
                           module.intermediate_size_per_partition,
                           module.hidden_size)

        fc31_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        expand_inter,
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

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self._online_eplb_not_supported(module)
        self.setup_quant_scales(module)

        _ensure_has_int8_woq_per_channel_instance_override(module)
        module._w8a16_onthefly_active = True

        logger.info_once(
            f"OnTheFlyINT8WoqPerChannelFusedMoEMethod (v18) active: "
            f"layout per expert: fc1={tuple(w3_w1_weight_shape[1:])}, "
            f"fc2={tuple(w2_weight_shape[1:])} "
            f"(is_gated_activation={module.is_gated_activation}, "
            f"expand_inter={expand_inter}, hidden={module.hidden_size}, "
            f"intermediate={module.intermediate_size_per_partition})",
            key="w8a16_onthefly_v18_buffer_shapes",
        )

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesINT8WoqPerChannel(
            fc31_weight_scale=module.fc31_weight_scale,
            fc2_weight_scale=module.fc2_weight_scale,
        )

    # -- expert weight loading --------------------------------------------

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: Optional[torch.Tensor],
                                 w3_weight: Optional[torch.Tensor],
                                 dst_w3_w1_weight: torch.Tensor):
        """Load BF16 up-proj (and optional gate-proj), MSE-quantize,
        run preprocess_weights_for_mixed_gemm, write INT8 + scale."""
        assert module.dtype in (torch.float16, torch.bfloat16), (
            f"On-the-fly W8A16 requires BF16/FP16 activations, "
            f"got dtype={module.dtype}")
        assert w1_weight is not None, (
            "On-the-fly W8A16: w1_weight (up_proj) must be provided")

        w1_shard = load_weight_shard(w1_weight, module.tp_size,
                                     module.tp_rank,
                                     TensorParallelMode.COLUMN)
        w3_shard = (load_weight_shard(w3_weight, module.tp_size,
                                      module.tp_rank,
                                      TensorParallelMode.COLUMN)
                    if _is_nonempty(w3_weight) else None)

        if w3_shard is not None:
            # Gated: [w3 | w1] along the N axis.
            assert module.is_gated_activation, (
                "Both w1 and w3 weights provided but module is configured "
                "for non-gated activation; refusing to silently concat")
            w31_nk = torch.cat([w3_shard, w1_shard], dim=0)
        else:
            # Non-gated (Nemotron-H Relu2): single up_proj. Matches the
            # unquantized loader's "gate_proj should be empty" branch
            # at quantization.py:574-577.
            w31_nk = w1_shard

        expected_n = module.expand_intermediate_size_per_partition
        assert w31_nk.shape[0] == expected_n, (
            f"On-the-fly W8A16 shape mismatch: source w31 has "
            f"N={w31_nk.shape[0]} but expand_intermediate={expected_n} "
            f"(is_gated={module.is_gated_activation}, w3_present="
            f"{w3_shard is not None})")

        w31_dev = w31_nk.to(dst_w3_w1_weight.device, dtype=module.dtype,
                            non_blocking=True)
        # Preprocessor wants [K=hidden, N=expand_inter].
        w31_kn = w31_dev.T.contiguous()
        int8_kn, scale_n = _mse_optimal_per_row_quantize(
            w31_kn, percentile_grid=self._percentile_grid)

        packed = module.preprocessor(int8_kn.contiguous(), torch.int8,
                                     module.dtype, module.sm_version)
        packed = packed.contiguous()
        # Sanity-check the packed tensor matches the destination buffer
        # so we catch any layout mismatch loudly rather than silently
        # corrupting weights.
        assert packed.numel() == dst_w3_w1_weight.numel(), (
            f"v18 W8A16 packed-tensor size mismatch: packed={packed.shape} "
            f"({packed.numel()} elems) vs dst={dst_w3_w1_weight.shape} "
            f"({dst_w3_w1_weight.numel()} elems). "
            f"Source w31_nk={w31_nk.shape}, w31_kn={w31_kn.shape}, "
            f"is_gated={module.is_gated_activation}")
        dst_w3_w1_weight.copy_(
            packed.view(dst_w3_w1_weight.dtype).view_as(dst_w3_w1_weight),
            non_blocking=True)

        expert_idx = _find_expert_idx(dst_w3_w1_weight,
                                      module.w3_w1_weight.data)
        module.fc31_weight_scale.data[expert_idx].copy_(
            scale_n.to(module.fc31_weight_scale.dtype), non_blocking=True)
        self._n_tensors_quantized += 1

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        assert module.dtype in (torch.float16, torch.bfloat16), (
            f"On-the-fly W8A16 requires BF16/FP16 activations, "
            f"got dtype={module.dtype}")
        assert w2_weight is not None, (
            "On-the-fly W8A16: w2_weight (down_proj) must be provided")

        w2_shard = load_weight_shard(w2_weight, module.tp_size,
                                     module.tp_rank, TensorParallelMode.ROW)
        w2_dev = w2_shard.to(dst_w2_weight.device, dtype=module.dtype,
                             non_blocking=True)
        # HF layout [N=hidden, K=intermediate] → preprocessor wants
        # [K=intermediate, N=hidden]. After packing the dst layout is
        # [E, intermediate, hidden] per expert.
        w2_kn = w2_dev.T.contiguous()
        int8_kn, scale_n = _mse_optimal_per_row_quantize(
            w2_kn, percentile_grid=self._percentile_grid)

        packed = module.preprocessor(int8_kn.contiguous(), torch.int8,
                                     module.dtype, module.sm_version)
        packed = packed.contiguous()
        assert packed.numel() == dst_w2_weight.numel(), (
            f"v18 W8A16 fc2 packed-tensor size mismatch: "
            f"packed={packed.shape} ({packed.numel()} elems) vs "
            f"dst={dst_w2_weight.shape} ({dst_w2_weight.numel()} elems).")
        dst_w2_weight.copy_(
            packed.view(dst_w2_weight.dtype).view_as(dst_w2_weight),
            non_blocking=True)

        expert_idx = _find_expert_idx(dst_w2_weight, module.w2_weight.data)
        module.fc2_weight_scale.data[expert_idx].copy_(
            scale_n.to(module.fc2_weight_scale.dtype), non_blocking=True)
        self._n_tensors_quantized += 1

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        logger.info_once(
            f"W8A16 on-the-fly (v18): quantized {self._n_tensors_quantized} "
            f"MoE expert weight tensors on layer "
            f"{getattr(module, 'layer_idx', '?')}",
            key=f"w8a16_onthefly_v18_layer_"
            f"{getattr(module, 'layer_idx', '?')}",
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
On-the-fly W8A16 (INT8 weight-only per-channel) MoE method.

This is v17 — the *non-gated-aware* fix of the v16 architecture.

v16 had a correct architecture (sibling FusedMoEMethodBase subclass that
allocates INT8 buffers, quantizes BF16 source inline, and force-flips
``has_int8_woq_per_channel``) but copied the buffer layout from
``INT8WoqPerChannelFusedMoEMethod`` verbatim — which is hard-coded for
**gated** activations (``intermediate * 2`` columns, ``cat([w3, w1])``
assembly). Nemotron-H uses non-gated Relu2 with a single up_proj
(``w3 == empty``), producing the same 3712 vs 1856 mismatch the user
originally reported.

v17 fixes this by:

  1. Sizing ``w3_w1_weight`` and ``fc31_weight_scale`` with
     ``module.expand_intermediate_size_per_partition``, which is
     ``intermediate`` for non-gated activations (Relu2, Identity)
     and ``2 * intermediate`` for gated (Swiglu, Geglu, etc.). This
     mirrors ``UnquantizedFusedMoEMethod.create_weights``.

  2. In ``load_expert_w3_w1_weight``, detecting the empty-w3 case
     (``w3_weight is None`` or zero-sized first dim) and quantizing
     **only** ``w1_weight`` (the up_proj) into the full dst buffer
     — matching the unquantized loader's gate_proj-empty branch
     (``quantization.py:574-577``).

  3. When both w1 and w3 are present (the gated case), keep the
     existing ``cat([w3, w1])`` flow so the patch continues to work
     for gated models that may opt-in.

The C++ INT8 weight-only fused_moe kernel already supports
``ActivationType::Relu2`` with the non-gated ``IdentityAdaptor`` path:

  * ``cpp/.../moe_gemm_template_dispatch.h:973`` — Relu2 case in the
    activation switch for ``MoeGemmRunner<T, WeightType, ...>::
    moeGemmBiasAct``. BF16xINT8 specialisation is compiled via
    ``moe_gemm_kernels_bf16_uint8.cu``.
  * ``cpp/.../moe_kernels.cu:2404-2407`` — Relu2 uses
    ``IdentityAdaptor<cutlass::epilogue::thread::Relu2>`` (NOT
    ``GLUAdaptor``), confirming non-gated layout.

So the fix is purely Python-side; no .so rebuild required.

Activated by env var ``TRTLLM_MOE_W8A16_ONTHEFLY=1`` (the dispatch
hook in ``fused_moe_cutlass.py`` checks this before falling through to
the standard quant_method selection). Default off.
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
# MSE-optimal per-row INT8 scale search (data-free).
# ---------------------------------------------------------------------------


def _mse_optimal_per_row_quantize(
    weight_bf16_kn: torch.Tensor,
    *,
    percentile_grid: Sequence[float] = _DEFAULT_GRID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [K, N] BF16 weight to [K, N] INT8 with per-output-channel
    (per-N) BF16 scales chosen by MSE-optimal percentile clip search.

    Returns (int8_kn, scale_n).
    """
    assert weight_bf16_kn.ndim == 2, (
        f"Expected 2-D [K,N] weight, got shape {tuple(weight_bf16_kn.shape)}")
    w = weight_bf16_kn.float()
    K, N = w.shape
    abs_w = w.abs()
    pgrid = torch.tensor(percentile_grid, dtype=torch.float32,
                         device=w.device) / 100.0
    try:
        clips = torch.quantile(abs_w, pgrid, dim=0)  # [P, N]
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
# MoE method class.
# ---------------------------------------------------------------------------


def _is_nonempty(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.numel() > 0 and t.shape[0] > 0


class OnTheFlyINT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):
    """W8A16 MoE method: quantizes BF16 expert weights to INT8 at load
    time with no precomputed scales. Supports both gated activations
    (Swiglu/Geglu — w1+w3 concatenated) and non-gated activations
    (Relu2/Identity — single up_proj, w3 empty).
    """

    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

    def __init__(self) -> None:
        super().__init__()
        self._percentile_grid = _percentile_grid()
        self._n_tensors_quantized = 0

    # -- buffer allocation ---------------------------------------------------

    def create_weights(self, module: torch.nn.Module):
        from tensorrt_llm._utils import get_sm_version

        module.sm_version = get_sm_version()
        # Match the existing INT8 woq path which targets the Ampere
        # layout on SM>=90 too.
        module.sm_version = (80 if module.sm_version >= 90 else
                             module.sm_version)
        module.preprocessor = preprocess_weights_for_mixed_gemm

        # --------------------------------------------------------------
        # Critical v17 change: use ``expand_intermediate_size_per_partition``
        # (= intermediate * 2 for gated, intermediate * 1 for non-gated)
        # rather than hard-coded ``intermediate * 2``. Mirrors
        # ``UnquantizedFusedMoEMethod.create_weights`` so the buffer
        # shape matches what the model loader expects to copy into.
        # --------------------------------------------------------------
        expand_inter = module.expand_intermediate_size_per_partition
        weight_dtype = torch.int8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              expand_inter, module.hidden_size)
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

        # Force ``has_int8_woq_per_channel`` to True on this instance so
        # ``run_moe`` dispatches to the INT8 weight-only kernel.
        _ensure_has_int8_woq_per_channel_instance_override(module)
        module._w8a16_onthefly_active = True

        logger.info_once(
            f"OnTheFlyINT8WoqPerChannelFusedMoEMethod active: "
            f"expand_intermediate={expand_inter} "
            f"(is_gated_activation={module.is_gated_activation}), "
            f"hidden={module.hidden_size}, "
            f"intermediate={module.intermediate_size_per_partition}",
            key="w8a16_onthefly_buffer_shapes",
        )

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesINT8WoqPerChannel(
            fc31_weight_scale=module.fc31_weight_scale,
            fc2_weight_scale=module.fc2_weight_scale,
        )

    # -- expert weight loading ---------------------------------------------

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: Optional[torch.Tensor],
                                 w3_weight: Optional[torch.Tensor],
                                 dst_w3_w1_weight: torch.Tensor):
        """Load BF16 w1 (and optionally w3), quantize on the fly, store
        INT8 + scale.

        Handles both the gated case (w1 + w3 both present, cat-ed) and
        the non-gated case (w3 is None or empty — Nemotron-H Relu2).
        """
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

        # Assemble the source tensor in [N, K] HF layout.
        if w3_shard is not None:
            # Gated case: concat w3 then w1 along the N axis. This
            # matches the existing INT8WoqPerChannel and the gated
            # epilogue's expectation of ``[w3_chunk | w1_chunk]``.
            assert module.is_gated_activation, (
                "Both w1 and w3 weights provided but module is configured "
                "for non-gated activation; refusing to silently concat")
            w31_nk = torch.cat([w3_shard, w1_shard], dim=0)
        else:
            # Non-gated case: w3 is empty (Nemotron-H Relu2). The dst
            # buffer is already sized to (expand_inter, K) where
            # expand_inter == intermediate (not intermediate*2), so we
            # write the single up_proj directly.
            assert not module.is_gated_activation or (
                w1_shard.shape[0]
                == module.expand_intermediate_size_per_partition), (
                    "Module is gated but w3 is empty/missing; cannot "
                    "fill the second chunk of fc31_weight")
            w31_nk = w1_shard

        # Sanity-check the source N dimension matches what we allocated.
        expected_n = module.expand_intermediate_size_per_partition
        assert w31_nk.shape[0] == expected_n, (
            f"On-the-fly W8A16 shape mismatch: source w31 has "
            f"N={w31_nk.shape[0]} but expand_intermediate={expected_n} "
            f"(is_gated={module.is_gated_activation}, w3_present="
            f"{w3_shard is not None})")

        w31_dev = w31_nk.to(dst_w3_w1_weight.device, dtype=module.dtype,
                            non_blocking=True)
        # Preprocessor wants [K, N] (after the unsqueeze(0) → 3D).
        w31_kn = w31_dev.T.contiguous()
        int8_kn, scale_n = _mse_optimal_per_row_quantize(
            w31_kn, percentile_grid=self._percentile_grid)

        packed = module.preprocessor(int8_kn.contiguous(), torch.int8,
                                     module.dtype, module.sm_version)
        packed = packed.contiguous()
        # The dst buffer's per-element dtype is int8; packed is int8.
        dst_w3_w1_weight.copy_(packed.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

        expert_idx = _find_expert_idx(dst_w3_w1_weight,
                                      module.w3_w1_weight.data)
        module.fc31_weight_scale.data[expert_idx].copy_(
            scale_n.to(module.fc31_weight_scale.dtype), non_blocking=True)
        self._n_tensors_quantized += 1

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """Load BF16 w2 (down_proj), quantize on the fly, store INT8."""
        assert module.dtype in (torch.float16, torch.bfloat16), (
            f"On-the-fly W8A16 requires BF16/FP16 activations, "
            f"got dtype={module.dtype}")
        assert w2_weight is not None, (
            "On-the-fly W8A16: w2_weight (down_proj) must be provided")

        w2_shard = load_weight_shard(w2_weight, module.tp_size,
                                     module.tp_rank, TensorParallelMode.ROW)
        w2_dev = w2_shard.to(dst_w2_weight.device, dtype=module.dtype,
                             non_blocking=True)
        # HF layout [N=hidden, K=intermediate] → preprocessor wants [K, N].
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
        # Scales were synthesised inline in the per-expert loaders.
        # Nothing to read from the checkpoint dict.
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
    into the parent (E, *expert_shape) tensor.
    """
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
    """Idempotently patch the @property so an instance flag wins.

    ``CutlassFusedMoE.has_int8_woq_per_channel`` is defined as a Python
    @property reading ``self.quant_config``. For BF16 checkpoints that
    returns False, so ``run_moe`` would not pass
    ``use_int8_woq_per_channel=True`` to the kernel. We patch the
    property on the class that defines it so it honours an instance
    attribute ``_force_int8_woq_per_channel`` when set. The patch is
    idempotent (guarded by a sentinel class attribute) and global, but
    keyed on a private attribute name only this module sets.
    """
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

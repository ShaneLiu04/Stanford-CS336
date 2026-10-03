"""Memory-efficient scaled dot-product attention.

The PyTorch implementation uses tiled online softmax in the forward pass and
recomputes one score tile at a time in the backward pass.  Consequently, no
``(batch, n_queries, n_keys)`` attention matrix is materialized.

When Triton is available, :class:`TritonFlashAttention` uses custom Triton
kernels for both its CUDA forward and backward passes.  The backward pass
precomputes the per-query softmax correction and recomputes score tiles from
the saved log-sum-exp; it never materializes a full attention matrix.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # Triton is optional (and normally unavailable on CPU).
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_QUERY_TILE = 64
_KEY_TILE = 64


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape (batch, sequence, head_dim)")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_floating_point():
        raise TypeError("q, k, and v must be floating-point tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same sequence length")
    if k.shape[1] == 0:
        raise ValueError("attention requires at least one key/value")


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _causal_mask(
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    device: torch.device,
) -> torch.Tensor:
    queries = torch.arange(query_start, query_end, device=device)
    keys = torch.arange(key_start, key_end, device=device)
    return queries[:, None] >= keys[None, :]


def _torch_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention with online softmax over key tiles."""
    batch, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    acc_dtype = _accumulator_dtype(q.dtype)
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(q)
    # LSE is intentionally the sole saved tensor with shape (B, Nq).
    lse = torch.empty((batch, n_queries), dtype=acc_dtype, device=q.device)

    for query_start in range(0, n_queries, _QUERY_TILE):
        query_end = min(query_start + _QUERY_TILE, n_queries)
        q_tile = q[:, query_start:query_end].to(acc_dtype)
        rows = query_end - query_start
        row_max = torch.full(
            (batch, rows), -torch.inf, dtype=acc_dtype, device=q.device
        )
        row_sum = torch.zeros((batch, rows), dtype=acc_dtype, device=q.device)
        accumulator = torch.zeros(
            (batch, rows, head_dim), dtype=acc_dtype, device=q.device
        )

        for key_start in range(0, n_keys, _KEY_TILE):
            key_end = min(key_start + _KEY_TILE, n_keys)
            k_tile = k[:, key_start:key_end].to(acc_dtype)
            v_tile = v[:, key_start:key_end].to(acc_dtype)
            scores = torch.bmm(q_tile, k_tile.transpose(1, 2)) * scale
            if is_causal:
                mask = _causal_mask(
                    query_start, query_end, key_start, key_end, q.device
                )
                scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(row_max, tile_max)
            # A causal tile can be wholly masked.  Mapping inf-inf to zero
            # makes its rescaling factor one and its probabilities zero.
            old_scale = torch.exp(row_max - new_max)
            old_scale = torch.nan_to_num(old_scale, nan=1.0)
            probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            new_sum = row_sum * old_scale + probabilities.sum(dim=-1)
            accumulator = (
                accumulator * old_scale.unsqueeze(-1)
                + torch.bmm(probabilities, v_tile)
            )
            row_max, row_sum = new_max, new_sum

        output[:, query_start:query_end] = (accumulator / row_sum.unsqueeze(-1)).to(
            q.dtype
        )
        lse[:, query_start:query_end] = row_max + torch.log(row_sum)

    return output, lse


def _torch_backward_tiled(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute score tiles from LSE and apply the exact softmax derivative."""
    _, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    acc_dtype = _accumulator_dtype(q.dtype)
    scale = 1.0 / math.sqrt(head_dim)
    dq = torch.zeros_like(q, dtype=acc_dtype)
    dk = torch.zeros_like(k, dtype=acc_dtype)
    dv = torch.zeros_like(v, dtype=acc_dtype)

    for query_start in range(0, n_queries, _QUERY_TILE):
        query_end = min(query_start + _QUERY_TILE, n_queries)
        q_tile = q[:, query_start:query_end].to(acc_dtype)
        do_tile = grad_output[:, query_start:query_end].to(acc_dtype)
        o_tile = output[:, query_start:query_end].to(acc_dtype)
        lse_tile = lse[:, query_start:query_end]
        # D_i = sum_j dP_ij P_ij = dot(dO_i, O_i).
        softmax_correction = (do_tile * o_tile).sum(dim=-1)
        dq_tile = torch.zeros_like(q_tile)

        for key_start in range(0, n_keys, _KEY_TILE):
            key_end = min(key_start + _KEY_TILE, n_keys)
            k_tile = k[:, key_start:key_end].to(acc_dtype)
            v_tile = v[:, key_start:key_end].to(acc_dtype)
            scores = torch.bmm(q_tile, k_tile.transpose(1, 2)) * scale
            if is_causal:
                mask = _causal_mask(
                    query_start, query_end, key_start, key_end, q.device
                )
                scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)

            probabilities = torch.exp(scores - lse_tile.unsqueeze(-1))
            dp = torch.bmm(do_tile, v_tile.transpose(1, 2))
            ds = probabilities * (dp - softmax_correction.unsqueeze(-1))
            dq_tile.add_(torch.bmm(ds, k_tile), alpha=scale)
            dk[:, key_start:key_end].add_(
                torch.bmm(ds.transpose(1, 2), q_tile), alpha=scale
            )
            dv[:, key_start:key_end].add_(
                torch.bmm(probabilities.transpose(1, 2), do_tile)
            )

        dq[:, query_start:query_end] = dq_tile

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention(torch.autograd.Function):
    """Portable tiled FlashAttention-style custom autograd operation."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _check_inputs(q, k, v)
        output, lse = _torch_forward_tiled(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _torch_backward_tiled(
            grad_output, q, k, v, output, lse, ctx.is_causal
        )
        return dq, dk, dv, None


if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        q_ptrs = (
            q_ptr
            + batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + dim_offsets[None, :] * stride_qd
        )
        q_tile = tl.load(
            q_ptrs,
            mask=(query_offsets[:, None] < n_queries)
            & (dim_offsets[None, :] < head_dim),
            other=0.0,
        )
        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, n_keys, BLOCK_N):
            current_keys = key_start + key_offsets
            k_ptrs = (
                k_ptr
                + batch_index * stride_kb
                + current_keys[None, :] * stride_kk
                + dim_offsets[:, None] * stride_kd
            )
            k_tile = tl.load(
                k_ptrs,
                mask=(current_keys[None, :] < n_keys)
                & (dim_offsets[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q_tile, k_tile) * scale
            valid = (query_offsets[:, None] < n_queries) & (
                current_keys[None, :] < n_keys
            )
            if IS_CAUSAL:
                valid = valid & (query_offsets[:, None] >= current_keys[None, :])
            scores = tl.where(valid, scores, -float("inf"))

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            old_scale = tl.exp(row_max - new_max)
            old_scale = tl.where(new_max == -float("inf"), 1.0, old_scale)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)

            v_ptrs = (
                v_ptr
                + batch_index * stride_vb
                + current_keys[:, None] * stride_vk
                + dim_offsets[None, :] * stride_vd
            )
            v_tile = tl.load(
                v_ptrs,
                mask=(current_keys[:, None] < n_keys)
                & (dim_offsets[None, :] < head_dim),
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(v_tile.dtype), v_tile
            )
            row_max = new_max

        output_ptrs = (
            output_ptr
            + batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dim_offsets[None, :] * stride_od
        )
        tl.store(
            output_ptrs,
            accumulator / row_sum[:, None],
            mask=(query_offsets[:, None] < n_queries)
            & (dim_offsets[None, :] < head_dim),
        )
        tl.store(
            lse_ptr + batch_index * n_queries + query_offsets,
            row_max + tl.log(row_sum),
            mask=query_offsets < n_queries,
        )

    @triton.jit
    def _flash_attention_delta_kernel(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_gob: tl.constexpr,
        stride_goq: tl.constexpr,
        stride_god: tl.constexpr,
        n_queries: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        mask = (query_offsets[:, None] < n_queries) & (
            dim_offsets[None, :] < head_dim
        )
        output = tl.load(
            output_ptr
            + batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dim_offsets[None, :] * stride_od,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr
            + batch_index * stride_gob
            + query_offsets[:, None] * stride_goq
            + dim_offsets[None, :] * stride_god,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(output * grad_output, axis=1)
        tl.store(
            delta_ptr + batch_index * n_queries + query_offsets,
            delta,
            mask=query_offsets < n_queries,
        )

    @triton.jit
    def _flash_attention_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        dq_ptr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_gob: tl.constexpr,
        stride_goq: tl.constexpr,
        stride_god: tl.constexpr,
        stride_dqb: tl.constexpr,
        stride_dqq: tl.constexpr,
        stride_dqd: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)
        query_dim_mask = (query_offsets[:, None] < n_queries) & (
            dim_offsets[None, :] < head_dim
        )
        q = tl.load(
            q_ptr
            + batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + dim_offsets[None, :] * stride_qd,
            mask=query_dim_mask,
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_ptr
            + batch_index * stride_gob
            + query_offsets[:, None] * stride_goq
            + dim_offsets[None, :] * stride_god,
            mask=query_dim_mask,
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch_index * n_queries + query_offsets,
            mask=query_offsets < n_queries,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + batch_index * n_queries + query_offsets,
            mask=query_offsets < n_queries,
            other=0.0,
        )
        dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, n_keys, BLOCK_N):
            current_keys = key_start + key_offsets
            key_dim_mask = (current_keys[:, None] < n_keys) & (
                dim_offsets[None, :] < head_dim
            )
            k = tl.load(
                k_ptr
                + batch_index * stride_kb
                + current_keys[:, None] * stride_kk
                + dim_offsets[None, :] * stride_kd,
                mask=key_dim_mask,
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + batch_index * stride_vb
                + current_keys[:, None] * stride_vk
                + dim_offsets[None, :] * stride_vd,
                mask=key_dim_mask,
                other=0.0,
            )
            valid = (query_offsets[:, None] < n_queries) & (
                current_keys[None, :] < n_keys
            )
            if IS_CAUSAL:
                valid = valid & (query_offsets[:, None] >= current_keys[None, :])
            scores = tl.dot(q, tl.trans(k)) * scale
            probabilities = tl.where(
                valid, tl.exp(scores - lse[:, None]), 0.0
            )
            dp = tl.dot(grad_output, tl.trans(v))
            ds = probabilities * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)

        tl.store(
            dq_ptr
            + batch_index * stride_dqb
            + query_offsets[:, None] * stride_dqq
            + dim_offsets[None, :] * stride_dqd,
            dq * scale,
            mask=query_dim_mask,
        )

    @triton.jit
    def _flash_attention_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_gob: tl.constexpr,
        stride_goq: tl.constexpr,
        stride_god: tl.constexpr,
        stride_dkb: tl.constexpr,
        stride_dkk: tl.constexpr,
        stride_dkd: tl.constexpr,
        stride_dvb: tl.constexpr,
        stride_dvk: tl.constexpr,
        stride_dvd: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        query_offsets_in_block = tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        key_dim_mask = (key_offsets[:, None] < n_keys) & (
            dim_offsets[None, :] < head_dim
        )
        k = tl.load(
            k_ptr
            + batch_index * stride_kb
            + key_offsets[:, None] * stride_kk
            + dim_offsets[None, :] * stride_kd,
            mask=key_dim_mask,
            other=0.0,
        )
        v = tl.load(
            v_ptr
            + batch_index * stride_vb
            + key_offsets[:, None] * stride_vk
            + dim_offsets[None, :] * stride_vd,
            mask=key_dim_mask,
            other=0.0,
        )
        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

        for query_start in range(0, n_queries, BLOCK_M):
            query_offsets = query_start + query_offsets_in_block
            query_dim_mask = (query_offsets[:, None] < n_queries) & (
                dim_offsets[None, :] < head_dim
            )
            q = tl.load(
                q_ptr
                + batch_index * stride_qb
                + query_offsets[:, None] * stride_qq
                + dim_offsets[None, :] * stride_qd,
                mask=query_dim_mask,
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr
                + batch_index * stride_gob
                + query_offsets[:, None] * stride_goq
                + dim_offsets[None, :] * stride_god,
                mask=query_dim_mask,
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + batch_index * n_queries + query_offsets,
                mask=query_offsets < n_queries,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr + batch_index * n_queries + query_offsets,
                mask=query_offsets < n_queries,
                other=0.0,
            )
            valid = (query_offsets[:, None] < n_queries) & (
                key_offsets[None, :] < n_keys
            )
            if IS_CAUSAL:
                valid = valid & (query_offsets[:, None] >= key_offsets[None, :])
            scores = tl.dot(q, tl.trans(k)) * scale
            probabilities = tl.where(
                valid, tl.exp(scores - lse[:, None]), 0.0
            )
            dp = tl.dot(grad_output, tl.trans(v))
            ds = probabilities * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
            dv += tl.dot(
                tl.trans(probabilities.to(grad_output.dtype)), grad_output
            )

        tl.store(
            dk_ptr
            + batch_index * stride_dkb
            + key_offsets[:, None] * stride_dkk
            + dim_offsets[None, :] * stride_dkd,
            dk * scale,
            mask=key_dim_mask,
        )
        tl.store(
            dv_ptr
            + batch_index * stride_dvb
            + key_offsets[:, None] * stride_dvk
            + dim_offsets[None, :] * stride_dvd,
            dv,
            mask=key_dim_mask,
        )


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not q.is_cuda:
        raise ValueError("TritonFlashAttention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Triton forward supports float16, bfloat16, and float32")
    head_dim = q.shape[-1]
    if head_dim > 256:
        raise ValueError("Triton forward supports head dimensions up to 256")

    q_contiguous = q.contiguous()
    k_contiguous = k.contiguous()
    v_contiguous = v.contiguous()
    batch, n_queries, _ = q.shape
    n_keys = k.shape[1]
    output = torch.empty_like(q_contiguous)
    lse = torch.empty((batch, n_queries), dtype=torch.float32, device=q.device)
    block_m, block_n = 32, 32
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_queries, block_m), batch)
    _flash_attention_forward_kernel[grid](
        q_contiguous,
        k_contiguous,
        v_contiguous,
        output,
        lse,
        *q_contiguous.stride(),
        *k_contiguous.stride(),
        *v_contiguous.stride(),
        *output.stride(),
        n_queries,
        n_keys,
        head_dim,
        1.0 / math.sqrt(head_dim),
        IS_CAUSAL=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output, lse


def _triton_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run tiled Triton kernels for the exact softmax backward equations."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not q.is_cuda:
        raise ValueError("TritonFlashAttention requires CUDA tensors")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = output.contiguous()
    grad_output = grad_output.contiguous()
    batch, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((batch, n_queries), dtype=torch.float32, device=q.device)
    block_m, block_n = 32, 32
    block_d = triton.next_power_of_2(head_dim)

    query_grid = (triton.cdiv(n_queries, block_m), batch)
    _flash_attention_delta_kernel[query_grid](
        output,
        grad_output,
        delta,
        *output.stride(),
        *grad_output.stride(),
        n_queries,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _flash_attention_dq_kernel[query_grid](
        q,
        k,
        v,
        grad_output,
        lse,
        delta,
        dq,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_output.stride(),
        *dq.stride(),
        n_queries,
        n_keys,
        head_dim,
        1.0 / math.sqrt(head_dim),
        IS_CAUSAL=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    key_grid = (triton.cdiv(n_keys, block_n), batch)
    _flash_attention_dkdv_kernel[key_grid](
        q,
        k,
        v,
        grad_output,
        lse,
        delta,
        dk,
        dv,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_output.stride(),
        *dk.stride(),
        *dv.stride(),
        n_queries,
        n_keys,
        head_dim,
        1.0 / math.sqrt(head_dim),
        IS_CAUSAL=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return dq, dk, dv


class TritonFlashAttention(torch.autograd.Function):
    """CUDA FlashAttention using tiled Triton forward and backward kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _check_inputs(q, k, v)
        output, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _triton_backward(
            grad_output, q, k, v, output, lse, ctx.is_causal
        )
        return dq, dk, dv, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Apply the portable PyTorch custom-autograd implementation."""
    return FlashAttention.apply(q, k, v, is_causal)


# Explicit aliases make the two autograd Function implementations easy to wire
# into assignment adapters without imposing a particular naming convention.
FlashAttentionPyTorch = FlashAttention
FlashAttentionTriton = TritonFlashAttention


__all__ = [
    "FlashAttention",
    "FlashAttentionPyTorch",
    "FlashAttentionTriton",
    "TritonFlashAttention",
    "flash_attention",
]

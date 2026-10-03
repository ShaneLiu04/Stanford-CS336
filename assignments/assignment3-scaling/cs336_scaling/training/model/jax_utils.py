from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import Array
from jax.core import AbstractValue
from jaxtyping import Float, Int, PyTree, Shaped

from cs336_scaling.utils import format_bytes


def tree_rearrange[T: PyTree](tree: T, pattern: str, **axes_lengths: int) -> T:
    return jax.tree.map(
        lambda x: rearrange(x, pattern, **axes_lengths),
        tree,
    )


def show_memory_analysis(f, *args, **kwargs) -> int:
    compiled = f.lower(*args, **kwargs).compile()
    analysis = compiled.memory_analysis()

    estimated_memory = (
        max(analysis.output_size_in_bytes, analysis.argument_size_in_bytes)
        + analysis.temp_size_in_bytes
    )
    print(
        f"Estimated memory usage for `{f.__qualname__}`: {format_bytes(estimated_memory)}",
    )

    return estimated_memory


def _clean_gather_impl(
    src: Shaped[Array, " gather_axis *gather_shape"],
    indices: Int[Array, " *indices_shape"],
) -> Float[Array, " indices_shape gather_shape"]:
    indices_shape = indices.shape
    if indices.ndim > 1:
        indices = indices.flatten()
    gather_shape = src.shape[1:]
    results = src.at[indices].get(
        wrap_negative_indices=False,
        mode="promise_in_bounds",
    )
    gathered = results.reshape(*indices_shape, *gather_shape)
    return gathered


@jax.custom_vjp
@jax.custom_batching.custom_vmap
def clean_gather(
    src: Shaped[Array, " gather_axis *gather_shape"],
    indices: Int[Array, " *indices_shape"],
) -> Float[Array, " indices_shape gather_shape"]:
    return _clean_gather_impl(src, indices)


@clean_gather.def_vmap
def clean_gather_vmap_rule(axis_size, in_batched, src, indices):
    assert in_batched == [False, True], f"{in_batched=}"
    if indices.ndim == 1:
        return clean_gather(src, indices), True
    flattened_indices = tree_rearrange(
        indices, "batch prev -> (batch prev)", batch=axis_size
    )
    flattened_out = clean_gather(src, flattened_indices)
    out = tree_rearrange(
        flattened_out, "(batch prev) ... -> batch prev ...", batch=axis_size
    )
    return out, True


def clean_gather_bwd_impl(src_type: AbstractValue, indices: jnp.ndarray, g):
    gather_shape = src_type.shape[1:]  # ty:ignore[unresolved-attribute]
    if indices.ndim > 1:
        indices = indices.flatten()
        g = g.reshape(-1, *gather_shape)
    grads = (
        jnp.zeros(src_type.shape, src_type.dtype, out_sharding=src_type.sharding)  # ty:ignore[unresolved-attribute]
        .at[indices]
        .add(g, wrap_negative_indices=False, mode="promise_in_bounds")
    )
    return grads


class SrcType(eqx.Module):
    src_type: AbstractValue = eqx.field(static=True)


def clean_gather_fwd_rule(src: jnp.ndarray, indices: jnp.ndarray):
    src_type = SrcType(src_type=jax.typeof(src))
    res = (
        src_type,
        indices,
    )
    return clean_gather(src, indices), res


@partial(jax.custom_vjp)
@jax.custom_batching.custom_vmap
def clean_gather_bwd(res, g):
    src_type, indices = res
    dsrc = clean_gather_bwd_impl(src_type.src_type, indices, g)
    return dsrc, None


@clean_gather_bwd.def_vmap
def clean_gather_bwd_vmap_rule(axis_size, in_batched, res, g):
    src_shape_dtype, indices = res
    assert jax.tree.leaves(in_batched) == [True, True]
    if indices.ndim == 1:
        return clean_gather_bwd(res, g), (False, None)
    flattened_indices, flattened_g = tree_rearrange(
        (indices, g), "batch prev ... -> (batch prev) ...", batch=axis_size
    )
    out = clean_gather_bwd_impl(
        src_shape_dtype.src_type, flattened_indices, flattened_g
    )
    return (out, None), (False, None)


def clean_gather_bwd_fwd(res, g):
    _src_shape_dtype, indices = res
    return clean_gather_bwd(res, g), indices


def clean_gather_bwd_bwd(res, g):
    ddg = clean_gather_bwd_bwd_impl(res, g)
    return (None, None), ddg


def clean_gather_bwd_bwd_impl(res, g):
    ddsrc, _ = g
    indices = res
    ddg = clean_gather(ddsrc, indices)
    return ddg


clean_gather_bwd.defvjp(clean_gather_bwd_fwd, clean_gather_bwd_bwd)
clean_gather.defvjp(clean_gather_fwd_rule, clean_gather_bwd)


def count_params(model, *, trainable_only: bool = True) -> int:
    filt = eqx.is_inexact_array if trainable_only else eqx.is_array
    return sum(x.size for x in jax.tree_util.tree_leaves(model) if filt(x))

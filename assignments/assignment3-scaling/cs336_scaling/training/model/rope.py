from functools import partial

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from equinox import nn
from jaxtyping import Array, Float, Int


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
) -> Float[Array, " 2 seq_len dim/2"]:
    d = jnp.arange(0, dim, 2) / dim
    freqs = jnp.array(theta) ** -d
    t = jnp.arange(seq_len)

    freqs = einsum(t, freqs, "t, f -> t f")
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)

    # Stack along the first dimension to shape (2, seq_len, freq)
    return einx.rearrange("seq freq, seq freq -> (1 + 1) seq freq", cos, sin)  # ty:ignore[invalid-return-type]


def apply_rotary_emb(
    x: Float[Array, " seq d_model"],
    freqs_cis: Float[Array, " 2 seq_len freq"],
    positions: Int[Array, " seq"] | None = None,
    neox_style: bool = False,
):
    """
    Apply RoPE to the input x.
    If positions is None, use the default positions (0, 1, 2, ..., seq - 1).
    """

    # Unzip the embedding dimension into two halves (alternating elements)
    if neox_style:
        x1, x2 = rearrange(x, "... (xy half_d) -> xy ... half_d", xy=2)
    else:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)

    if positions is None:
        cos, sin = freqs_cis[:, : x.shape[-2], :]
    else:
        cos, sin = freqs_cis[:, positions, :]

    x1_rot = cos * x1 - sin * x2
    x2_rot = sin * x1 + cos * x2

    # Zip the two half dimensions back together
    if neox_style:
        result = einx.rearrange(
            "... x_half, ... x_half -> ... ((1 + 1) x_half)", x1_rot, x2_rot
        )
    else:
        result = einx.rearrange(
            "... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot
        )
    return result


class BasicRotaryEmbedding(eqx.Module):
    """
    RoPE embedding layer, ideally shared between all layers of the model.
    """

    # freqs_cis: Float[Array, " 2 seq_len freq"]
    freqs_cis_index: nn.StateIndex[Float[Array, " 2 seq_len freq"] | None]
    theta: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, theta: float):
        self.theta = theta
        self.dim = dim
        self.freqs_cis_index = nn.StateIndex(None)

    def apply(
        self,
        x: Float[Array, " seq dim"],
        state: nn.State,
        positions: Int[Array, " seq"] | None = None,
        neox_style: bool = True,
    ) -> tuple[Float[Array, " seq dim"], nn.State]:
        assert positions is None, "Position indices are not implemented yet"
        assert x.ndim == 2
        state = self.update_cache(state, x.shape[0])
        freqs_cis = state.get(self.freqs_cis_index)

        assert freqs_cis is not None

        return apply_rotary_emb(x, freqs_cis, positions, neox_style=neox_style), state

    def apply_with_heads(
        self,
        x: Float[Array, " seq heads dim"],
        state: nn.State,
        positions: Int[Array, " seq"] | None = None,
        neox_style: bool = True,
    ) -> tuple[Float[Array, " seq heads dim"], nn.State]:
        return jax.vmap(
            partial(self.apply, neox_style=neox_style),
            in_axes=(1, None, None),
            out_axes=(1, None),
        )(x, state, positions)

    def update_cache(self, state: nn.State, seq_len: int) -> nn.State:
        freqs_cis = state.get(self.freqs_cis_index)
        if freqs_cis is None or freqs_cis.shape[1] < seq_len:
            freqs_cis = precompute_freqs_cis(self.dim, seq_len, self.theta)
            state = state.set(self.freqs_cis_index, freqs_cis)
        return state

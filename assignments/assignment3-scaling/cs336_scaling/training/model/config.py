from typing import Literal

from cs336_scaling.schemas.base import FrozenForbidExtraModel


class BasicTransformerConfig(FrozenForbidExtraModel):
    attention_bias: bool
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: int
    tie_word_embeddings: bool
    dtype: Literal["float32", "bfloat16"]
    vocab_size: int

    @property
    def jax_dtype(self):
        import jax.numpy as jnp

        return jnp.dtype(self.dtype)

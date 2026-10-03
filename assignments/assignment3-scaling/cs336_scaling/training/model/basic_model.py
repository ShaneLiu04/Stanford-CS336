from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from flash_hog.jax.attention import dot_product_attention as fhog_dot_product_attention
from jax import P
from jaxtyping import Array, Float, Int

import cs336_scaling.training.model._eqx_state_patch  # noqa: F401
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.model.jax_utils import clean_gather
from cs336_scaling.training.model.rope import BasicRotaryEmbedding


class BasicAttention(eqx.Module):
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm
    rotary: BasicRotaryEmbedding
    config: BasicTransformerConfig = eqx.field(static=True)

    def __init__(
        self,
        config: BasicTransformerConfig,
        *,
        key: jnp.ndarray,
    ):
        self.config = config

        q_proj_key, k_proj_key, v_proj_key, o_proj_key = jax.random.split(key, 4)
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            use_bias=config.attention_bias,
            key=q_proj_key,
            dtype=config.jax_dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            key=k_proj_key,
            dtype=config.jax_dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            key=v_proj_key,
            dtype=config.jax_dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            use_bias=config.attention_bias,
            key=o_proj_key,
            dtype=config.jax_dtype,
        )

        self.q_norm = nn.RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            use_bias=False,
            dtype=config.jax_dtype,
        )
        self.k_norm = nn.RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            use_bias=False,
            dtype=config.jax_dtype,
        )

        self.rotary = BasicRotaryEmbedding(  #
            dim=config.head_dim,
            theta=config.rope_theta,
        )

    def __call__(
        self: Self,
        hidden_state: Float[Array, "seq d_model"],
        position_ids: Int[Array, " seq"] | None,
        state: nn.State,
    ):
        sequence_length = hidden_state.shape[0]

        query_state = jax.vmap(self.q_proj)(hidden_state)
        key_state = jax.vmap(self.k_proj)(hidden_state)
        value_state = jax.vmap(self.v_proj)(hidden_state)

        query_state = jax.vmap(jax.vmap(self.q_norm))(
            query_state.reshape(
                sequence_length,
                self.config.num_attention_heads,
                self.config.head_dim,
            )
        )

        key_state = jax.vmap(jax.vmap(self.k_norm))(
            key_state.reshape(
                sequence_length,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
        )

        value_state = value_state.reshape(
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )

        query_state, state = self.rotary.apply_with_heads(
            query_state, state=state, positions=position_ids
        )
        key_state, state = self.rotary.apply_with_heads(
            key_state, state=state, positions=position_ids
        )

        query_state, key_state = jax.tree.map(
            lambda x: x.astype(value_state.dtype), (query_state, key_state)
        )

        if jax.default_backend() == "cpu":
            attention = jax.nn.dot_product_attention(
                query=query_state,
                key=key_state,
                value=value_state,
                is_causal=True,
                implementation="xla",
            )
        else:
            attention = fhog_dot_product_attention(
                query=query_state,
                key=key_state,
                value=value_state,
                is_causal=True,
            )

        attn_output = jax.vmap(self.o_proj)(
            attention.reshape(
                sequence_length,
                self.config.num_attention_heads * self.config.head_dim,
            )
        )

        return attn_output, state


class BasicMLP(eqx.Module):
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(self, config: BasicTransformerConfig, *, key: jnp.ndarray):
        super().__init__()
        gate_proj_key, up_proj_key, down_proj_key = jax.random.split(key, 3)
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            key=gate_proj_key,
            use_bias=False,
            dtype=config.jax_dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            key=up_proj_key,
            use_bias=False,
            dtype=config.jax_dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            key=down_proj_key,
            use_bias=False,
            dtype=config.jax_dtype,
        )

    def __call__(self, x: Float[Array, " seq d_model"]):
        down_proj = self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BasicDecoderLayer(eqx.Module):
    self_attn: BasicAttention
    mlp: BasicMLP
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, config: BasicTransformerConfig, *, key: jnp.ndarray):
        super().__init__()
        attn_key, mlp_key = jax.random.split(key)
        self.self_attn = BasicAttention(config=config, key=attn_key)
        self.mlp = BasicMLP(config, key=mlp_key)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_bias=False,
            dtype=config.jax_dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_bias=False,
            dtype=config.jax_dtype,
        )

    def __call__(
        self,
        *,
        hidden_state: Float[Array, "seq d_model"],
        position_ids: Int[Array, " seq"] | None,
        state: nn.State,
    ):
        residual = hidden_state

        hidden_state = jax.vmap(self.input_layernorm)(hidden_state)

        hidden_state, state = self.self_attn(
            hidden_state=hidden_state,
            position_ids=position_ids,
            state=state,
        )
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = jax.vmap(self.post_attention_layernorm)(hidden_state)
        hidden_state = jax.vmap(self.mlp)(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state, state


def all_gather_tree(tree, axis_name: str):
    def all_gather_weight(x):
        if axis_name in jax.typeof(x).vma:
            return jax.lax.all_gather(x, axis_name=axis_name, tiled=True)
        return x

    return jax.tree.map(all_gather_weight, tree)


class BasicCausalLM(eqx.Module):
    embed_tokens: nn.Embedding
    layers: BasicDecoderLayer
    norm: nn.RMSNorm
    lm_head: nn.Linear | None

    def __init__(self, config: BasicTransformerConfig, *, key: jnp.ndarray):
        super().__init__()
        embed_key, lm_head_key, layers_key = jax.random.split(key, num=3)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, key=embed_key, dtype=config.jax_dtype
        )

        rotary_layer = BasicDecoderLayer(config, key=key).self_attn.rotary
        layers = jax.vmap(lambda key: BasicDecoderLayer(config, key=key))(
            jax.random.split(layers_key, num=config.num_hidden_layers)
        )
        self.layers = eqx.tree_at(
            where=lambda m: m.self_attn.rotary, pytree=layers, replace=rotary_layer
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_bias=False,
            dtype=config.jax_dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                key=lm_head_key,
                dtype=config.jax_dtype,
            )

    def __call__(
        self,
        *,
        input_ids: Int[Array, " seq"],
        position_ids: Int[Array, " seq"] | None,
        state: nn.State,
    ) -> tuple[Float[Array, "seq vocab"], nn.State]:
        embed_tokens = jax.lax.all_gather(
            self.embed_tokens, axis_name="fsdp", tiled=True, axis=1
        )
        hidden_state = jax.vmap(clean_gather, in_axes=(None, 0))(
            embed_tokens.weight, input_ids
        )

        state = self.layers.self_attn.rotary.update_cache(
            state, seq_len=input_ids.shape[0]
        )

        def scan_fn_layers(state, hidden_state, layer):
            layer = all_gather_tree(layer, axis_name="fsdp")
            hidden_state, _new_state = layer(
                hidden_state=hidden_state,
                position_ids=position_ids,
                state=state,
            )
            return hidden_state, None

        hidden_state, _ = jax.lax.scan(
            partial(scan_fn_layers, state), hidden_state, self.layers
        )

        hidden_state = jax.vmap(self.norm)(hidden_state)

        if self.lm_head is None:
            tied_weight = jax.lax.all_gather(
                self.embed_tokens.weight, axis_name="fsdp", tiled=True, axis=1
            )
            hidden_state = hidden_state @ tied_weight.T
        else:
            lm_head = jax.lax.all_gather(self.lm_head, axis_name="fsdp", tiled=True)
            hidden_state = jax.vmap(lm_head)(hidden_state)

        return hidden_state, state

    def apply_sharding(
        self,
        mesh: jax.sharding.Mesh,
    ) -> Self:
        model = self
        sharding_specs = [
            (lambda m: m.embed_tokens.weight, P(None, "fsdp")),
            (lambda m: m.norm.weight, P()),
            (lambda m: m.layers.mlp.down_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.mlp.gate_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.mlp.up_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.input_layernorm.weight, P()),
            (lambda m: m.layers.post_attention_layernorm.weight, P()),
            (lambda m: m.layers.self_attn.k_norm.weight, P(None)),
            (lambda m: m.layers.self_attn.q_norm.weight, P(None)),
            (lambda m: m.layers.self_attn.q_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.self_attn.k_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.self_attn.v_proj.weight, P(None, "fsdp")),
            (lambda m: m.layers.self_attn.o_proj.weight, P(None, "fsdp")),
        ]
        if model.lm_head is not None:
            sharding_specs.append((lambda m: m.lm_head.weight, P("fsdp")))

        for where, weight_sharding_spec in sharding_specs:
            weight_sharding_spec = jax.NamedSharding(
                mesh=mesh, spec=weight_sharding_spec
            )
            model = eqx.tree_at(
                where,
                model,
                replace_fn=lambda weight: jax.reshard(weight, weight_sharding_spec),
            )
        return model

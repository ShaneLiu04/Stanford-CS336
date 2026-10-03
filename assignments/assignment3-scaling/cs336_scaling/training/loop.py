from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from equinox import nn
from jax import Array
from jax.sharding import PartitionSpec as P
from jaxtyping import Float

from cs336_scaling.training.data import Batch
from cs336_scaling.training.model.basic_model import BasicCausalLM
from cs336_scaling.training.training_config import TrainingConfig


@partial(
    jax.shard_map,
    out_specs=P(),
    axis_names={"fsdp"},
)
def sharded_sequence_loss(
    model: BasicCausalLM, state: nn.State, batch: Batch[Array]
) -> Float[Array, ""]:
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=0)
    def _sequence_loss_impl(
        model: BasicCausalLM,
        state: eqx.nn.State,
        batch: Batch[Array],
    ) -> Float[Array, ""]:
        """Token-masked mean CE loss for a single sequence."""
        logits, _state = model(
            state=state, input_ids=batch.input_ids, position_ids=None
        )
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch.labels
        )
        return per_token.sum() / batch.sequence_length()

    per_example_sequence_loss = _sequence_loss_impl(model, state, batch)
    numerator = jax.lax.psum(per_example_sequence_loss.sum(), axis_name="fsdp")
    denom = batch.n_sequences()
    return numerator / denom


def train_model(
    model: BasicCausalLM,
    state: eqx.nn.State,
    train_data: Batch[Array],
    training_config: TrainingConfig,
    opt_state: optax.OptState,
) -> tuple[BasicCausalLM, eqx.nn.State, Array, optax.OptState]:
    """Inner training loop over pre-batched data.

    ``train_data`` and ``data_weights`` must already carry a leading "steps" axis of length
    ``training_config.optimizer_steps_per_eval``. The scan runs one optimizer step per
    entry along that axis.
    """
    optimizer = training_config.optimizer_config.build(training_config)

    # Pre-populate the rotary cache for the full sequence length once, rather than on every step.
    # This makes the model's nn.State invariant inside the training scan so we don't need to thread it.
    state = model.layers.self_attn.rotary.update_cache(
        state, seq_len=train_data.sequence_length()
    )

    def step_fn(mdl__opt_st, batch_step: Batch[Array]):
        mdl, opt_st = mdl__opt_st

        loss, grads = jax.value_and_grad(sharded_sequence_loss)(mdl, state, batch_step)

        updates, opt_st = optimizer.update(grads, opt_st, mdl)
        mdl = eqx.apply_updates(mdl, updates)

        return (mdl, opt_st), loss

    (trained_model, opt_state), losses = jax.lax.scan(
        step_fn,
        (model, opt_state),
        train_data,
    )

    return trained_model, state, losses, opt_state


def val_model(
    model: BasicCausalLM,
    state: eqx.nn.State,
    val_data: Batch[Array],
) -> tuple[Array, eqx.nn.State]:
    """Evaluate mean per-sequence loss over pre-batched ``val_data``.

    ``val_data`` is rearranged into validation batches before scanning.
    """
    # Pre-populate the rotary cache so state is invariant inside the scan.
    state = model.layers.self_attn.rotary.update_cache(
        state, seq_len=val_data.sequence_length()
    )

    def accum_step(total_loss__total_count, batch_step: Batch[Array]):
        total_loss, total_count = total_loss__total_count

        mean_loss = sharded_sequence_loss(
            model,
            state,  # ty:ignore[too-many-positional-arguments]
            batch_step,
        )
        batch_token_count = jnp.asarray(
            batch_step.n_sequences(), dtype=total_count.dtype
        )
        return (
            total_loss + mean_loss * batch_token_count,  # ty:ignore[unsupported-operator]
            total_count + batch_token_count,
        ), None

    init = (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    (total_loss, total_count), _ = jax.lax.scan(accum_step, init, val_data)

    return total_loss / total_count, state


class OuterLossResult(eqx.Module):
    val_loss: Array
    train_losses: Array
    model: BasicCausalLM
    state: nn.State
    opt_state: optax.OptState


@partial(
    jax.jit,
    static_argnames=("training_config",),
)
def outer_loss(
    model: BasicCausalLM,
    state: eqx.nn.State,
    train_data: Batch[Array],
    val_data: Batch[Array],
    training_config: TrainingConfig,
    opt_state: optax.OptState,
) -> OuterLossResult:
    trained_model, state, train_losses, opt_state = train_model(
        model,
        state=state,
        train_data=train_data,
        training_config=training_config,
        opt_state=opt_state,
    )
    val_loss, state = val_model(trained_model, state, val_data)
    return OuterLossResult(
        val_loss=val_loss,
        train_losses=train_losses,
        model=trained_model,
        state=state,
        opt_state=opt_state,
    )

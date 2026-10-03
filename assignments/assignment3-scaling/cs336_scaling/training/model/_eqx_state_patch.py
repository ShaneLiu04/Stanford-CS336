import jax.numpy as jnp
from equinox.nn._stateful import (
    State,
    StateIndex,
    _Sentinel,
    _sentinel,
    _state_error,
    _Value,
    jtu,
)


def _State_set(self, item: StateIndex[_Value], value: _Value) -> "State":
    """Sets a new value for an [`equinox.nn.StateIndex`][], **and returns the
    updated state**.

    **Arguments:**

    - `item`: an [`equinox.nn.StateIndex`][].
    - `value`: the new value associated with that index.

    **Returns:**

    A new [`equinox.nn.State`][] object, with the update.

    As a safety guard against accidentally writing `state.set(item, value)` without
    assigning it to a new value, then the old object (`self`) will become invalid.
    """
    if isinstance(self._state, _Sentinel):
        raise ValueError(_state_error)
    if type(item) is not StateIndex:
        raise ValueError("Can only use `eqx.nn.StateIndex`s as state keys.")
    # old_value = self._state[item.marker]
    value = jtu.tree_map(jnp.asarray, value)
    ## Commented out for more flexibility
    # old_struct = jax.eval_shape(lambda: old_value)
    # new_struct = jax.eval_shape(lambda: value)
    # if tree_equal(old_struct, new_struct) is not True:
    #     old_repr = tree_pformat(old_struct, struct_as_array=True)
    #     new_repr = tree_pformat(new_struct, struct_as_array=True)
    #     raise ValueError(
    #         "Old and new values have different structures/shapes/dtypes. The old "
    #         f"value is {old_repr} and the new value is {new_repr}."
    #     )
    ## End of commented out code
    state = self._state.copy()  # pyright: ignore
    state[item.marker] = value
    new_self = object.__new__(State)
    new_self._state = state
    self._state = _sentinel
    return new_self


State.set = _State_set  # ty:ignore[invalid-assignment]

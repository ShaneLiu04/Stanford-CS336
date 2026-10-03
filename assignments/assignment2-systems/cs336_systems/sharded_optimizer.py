"""Optimizer-state sharding for replicated data-parallel models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.optim import Optimizer


T = TypeVar("T")


class ShardedOptimizer(Optimizer):
    """Shard an optimizer's parameters and state across distributed ranks.

    Parameters are assigned using ``global_unique_index % world_size``.  Each
    rank runs ``optimizer_cls`` only on the parameters it owns, after which the
    updated parameters are broadcast from their owners to all other ranks.

    ``state_dict`` is a rank-local shard whose state keys are global parameter
    indices.  Consequently, one checkpoint shard should be saved and restored
    per rank (with the same world size and parameter ordering).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ) -> None:
        self.optimizer_cls = optimizer_cls
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._sharded_ready = False
        self._global_params: list[torch.Tensor] = []
        self._param_to_index: dict[int, int] = {}

        normalized = self._normalize_initial_params(params)
        super().__init__(normalized, kwargs)

        self._index_new_parameters()
        local_groups = [self._local_group(group) for group in self.param_groups]
        self._local_optimizer = optimizer_cls(local_groups, **kwargs)
        self._copy_local_options_to_public_groups()
        self.defaults = self._local_optimizer.defaults
        self.state = self._local_optimizer.state
        self._sharded_ready = True

    @staticmethod
    def _parameter_from_item(item: Any) -> torch.Tensor:
        # Recent PyTorch versions allow (name, parameter) tuples.
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            return item[1]
        return item

    def _normalize_initial_params(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items = list(params)
        if items and isinstance(items[0], Mapping):
            groups = [dict(group) for group in items]
        else:
            groups = [{"params": items}]

        seen: set[int] = set()
        for group in groups:
            group_params = list(group["params"])
            unique_params = []
            unique_names = []
            names = group.get("param_names")
            for position, item in enumerate(group_params):
                parameter = self._parameter_from_item(item)
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    unique_params.append(item)
                    if names is not None:
                        unique_names.append(names[position])
            group["params"] = unique_params
            if names is not None:
                group["param_names"] = unique_names
        return groups

    def _index_new_parameters(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                key = id(parameter)
                if key not in self._param_to_index:
                    self._param_to_index[key] = len(self._global_params)
                    self._global_params.append(parameter)

    def _is_local(self, parameter: torch.Tensor) -> bool:
        return self._param_to_index[id(parameter)] % self._world_size == self._rank

    def _local_group(self, group: dict[str, Any]) -> dict[str, Any]:
        local_positions = [
            position for position, parameter in enumerate(group["params"]) if self._is_local(parameter)
        ]
        result = {key: value for key, value in group.items() if key not in {"params", "param_names"}}
        result["params"] = [group["params"][position] for position in local_positions]
        if "param_names" in group:
            result["param_names"] = [group["param_names"][position] for position in local_positions]
        return result

    def _copy_local_options_to_public_groups(self) -> None:
        for public_group, local_group in zip(self.param_groups, self._local_optimizer.param_groups, strict=True):
            params = public_group["params"]
            param_names = public_group.get("param_names")
            public_group.update({key: value for key, value in local_group.items() if key not in {"params", "param_names"}})
            public_group["params"] = params
            if param_names is not None:
                public_group["param_names"] = param_names

    def _copy_public_options_to_local_groups(self) -> None:
        for public_group, local_group in zip(self.param_groups, self._local_optimizer.param_groups, strict=True):
            for key, value in public_group.items():
                if key not in {"params", "param_names"}:
                    local_group[key] = value

    @torch.no_grad()
    def step(self, closure: Callable[[], T] | None = None) -> T | None:
        self._copy_public_options_to_local_groups()
        loss = self._local_optimizer.step(closure)

        if self._world_size > 1:
            for index, parameter in enumerate(self._global_params):
                dist.broadcast(parameter, src=index % self._world_size)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # The public optimizer contains every parameter, so this also clears
        # non-owned gradients (which the local optimizer does not know about).
        super().zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not self._sharded_ready:
            super().add_param_group(param_group)
            return

        group = dict(param_group)
        incoming = list(group["params"])
        unique = []
        unique_names = []
        names = group.get("param_names")
        seen = set(self._param_to_index)
        for position, item in enumerate(incoming):
            parameter = self._parameter_from_item(item)
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(item)
                if names is not None:
                    unique_names.append(names[position])
        group["params"] = unique
        if names is not None:
            group["param_names"] = unique_names

        super().add_param_group(group)
        new_public_group = self.param_groups[-1]
        self._index_new_parameters()
        self._local_optimizer.add_param_group(self._local_group(new_public_group))
        self._copy_local_options_to_public_groups()
        self.state = self._local_optimizer.state

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer-state shard using global state keys."""
        self._copy_public_options_to_local_groups()
        local_state_dict = self._local_optimizer.state_dict()

        local_id_to_global_id: dict[int, int] = {}
        for saved_group, live_group in zip(
            local_state_dict["param_groups"],
            self._local_optimizer.param_groups,
            strict=True,
        ):
            for local_id, parameter in zip(saved_group["params"], live_group["params"], strict=True):
                local_id_to_global_id[local_id] = self._param_to_index[id(parameter)]

        state = {
            local_id_to_global_id[local_id]: value
            for local_id, value in local_state_dict["state"].items()
        }
        param_groups = []
        for group in self.param_groups:
            saved_group = {key: value for key, value in group.items() if key != "params"}
            saved_group["params"] = [self._param_to_index[id(parameter)] for parameter in group["params"]]
            param_groups.append(saved_group)
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a shard produced by :meth:`state_dict`."""
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state dict has a different number of parameter groups "
                f"({len(saved_groups)} instead of {len(self.param_groups)})"
            )

        for saved_group, public_group in zip(saved_groups, self.param_groups, strict=True):
            if len(saved_group["params"]) != len(public_group["params"]):
                raise ValueError("loaded state dict contains a parameter group that doesn't match the optimizer")
            params = public_group["params"]
            public_group.update({key: value for key, value in saved_group.items() if key != "params"})
            public_group["params"] = params

        current_local = self._local_optimizer.state_dict()
        global_id_to_local_id: dict[int, int] = {}
        for saved_group, live_group in zip(
            current_local["param_groups"],
            self._local_optimizer.param_groups,
            strict=True,
        ):
            for local_id, parameter in zip(saved_group["params"], live_group["params"], strict=True):
                global_id_to_local_id[self._param_to_index[id(parameter)]] = local_id

        local_state = {
            global_id_to_local_id[global_id]: value
            for global_id, value in state_dict["state"].items()
            if global_id in global_id_to_local_id
        }
        local_groups = []
        for public_group, current_group in zip(self.param_groups, current_local["param_groups"], strict=True):
            group = {key: value for key, value in public_group.items() if key not in {"params", "param_names"}}
            group["params"] = current_group["params"]
            if "param_names" in public_group:
                group["param_names"] = [
                    public_group["param_names"][position]
                    for position, parameter in enumerate(public_group["params"])
                    if self._is_local(parameter)
                ]
            local_groups.append(group)

        self._local_optimizer.load_state_dict({"state": local_state, "param_groups": local_groups})
        self.defaults = self._local_optimizer.defaults
        self.state = self._local_optimizer.state


__all__ = ["ShardedOptimizer"]

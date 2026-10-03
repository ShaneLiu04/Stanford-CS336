"""A small, teaching-oriented fully sharded data-parallel wrapper.

Only weights belonging to :class:`cs336_basics.model.Linear` and
:class:`cs336_basics.model.Embedding` are sharded.  They are stored as flat
FP32 shards between iterations, all-gathered by module forward hooks, and
reduced back to FP32 gradient shards by
:meth:`FullyShardedDataParallel.finish_gradient_synchronization`.  Parameters
of all other module types remain replicated and their gradients are averaged.

This deliberately favours a clear parameter/optimizer interface over peak
memory efficiency: gathered parameters remain materialized until backward is
finished.  Consequently it does not overlap reduce-scatter with backward and
does not have the peak-memory behaviour of production PyTorch FSDP.  Uneven
parameters are supported by padding each rank's flat shard.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

__all__ = ["FullyShardedDataParallel"]


@dataclass
class _ShardInfo:
    """Bookkeeping for one sharded parameter."""

    name: str
    shape: torch.Size
    numel: int
    shard_numel: int
    master_shard: torch.Tensor
    is_full: bool = False


class FullyShardedDataParallel(nn.Module):
    """Shard ``cs336_basics`` Linear and Embedding weights across ranks.

    The wrapped model is available as :attr:`module`.  ``parameters()`` yields
    the wrapped model's actual ``Parameter`` objects, so an ordinary optimizer
    can be constructed directly from ``fsdp_model.parameters()``.  Call
    :meth:`finish_gradient_synchronization` after ``backward()`` and before
    ``optimizer.step()``.

    Args:
        module: Model to wrap.  Every rank must construct the same model.
        compute_dtype: Optional dtype used for gathered Linear/Embedding
            weights and their communication.  Local master shards are FP32.
        process_group: Optional process group.  The default group is used when
            omitted.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if compute_dtype is not None and not compute_dtype.is_floating_point:
            raise TypeError("compute_dtype must be a floating-point dtype")

        self.module = module
        self.compute_dtype = compute_dtype
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self._shards: dict[nn.Parameter, _ShardInfo] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Importing here keeps this module importable before cs336_basics is
        # installed and, more importantly, identifies precisely the two
        # assignment layer classes that the exercise asks us to shard.
        from cs336_basics.model import Embedding, Linear

        names = {parameter: name for name, parameter in module.named_parameters()}
        seen: set[nn.Parameter] = set()
        for submodule in module.modules():
            if not isinstance(submodule, (Linear, Embedding)):
                continue
            parameter = submodule.weight
            if parameter not in seen:
                seen.add(parameter)
                self._make_shard(parameter, names[parameter])
            self._hooks.append(submodule.register_forward_pre_hook(self._forward_pre_hook))
            # During an ordinary forward the full weight is still resident.
            # This hook also makes recomputation/custom invocation robust.
            self._hooks.append(submodule.register_full_backward_pre_hook(self._backward_pre_hook))

    def _make_shard(self, parameter: nn.Parameter, name: str) -> None:
        if not parameter.is_floating_point():
            raise TypeError(f"cannot shard non-floating parameter {name!r}")

        original_shape = parameter.shape
        original_numel = parameter.numel()
        full = parameter.detach().to(torch.float32).contiguous().view(-1)
        if self.world_size > 1:
            # Rank zero is authoritative, as in conventional data parallel
            # initialization.  This also handles models seeded differently.
            dist.broadcast(full, src=0, group=self.process_group)

        shard_numel = (full.numel() + self.world_size - 1) // self.world_size
        padded_numel = shard_numel * self.world_size
        if padded_numel != full.numel():
            padded = full.new_zeros(padded_numel)
            padded[: full.numel()].copy_(full)
            full = padded
        start = self.rank * shard_numel
        parameter.data = full.narrow(0, start, shard_numel).clone()
        self._shards[parameter] = _ShardInfo(
            name=name,
            shape=original_shape,
            numel=original_numel,
            shard_numel=shard_numel,
            master_shard=parameter.data,
        )

    def _materialize(self, parameter: nn.Parameter) -> None:
        info = self._shards[parameter]
        if info.is_full:
            return
        # Keep a reference to the optimizer-updated FP32 storage before
        # replacing Parameter.data with a lower-precision gathered tensor.
        info.master_shard = parameter.data
        local = info.master_shard.to(self.compute_dtype or torch.float32).contiguous()
        if self.world_size == 1:
            gathered = local
        else:
            pieces = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(pieces, local, group=self.process_group)
            gathered = torch.cat(pieces)
        parameter.data = gathered[: info.numel].view(info.shape)
        info.is_full = True

    def _forward_pre_hook(self, submodule: nn.Module, inputs: tuple[object, ...]) -> None:
        del inputs
        self._materialize(submodule.weight)

    def _backward_pre_hook(self, submodule: nn.Module, grad_output: tuple[torch.Tensor, ...]) -> None:
        del grad_output
        self._materialize(submodule.weight)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """Average gradients and restore FP32 local parameter shards."""

        for parameter in self.module.parameters():
            info = self._shards.get(parameter)
            if info is None:
                if parameter.grad is not None and self.world_size > 1:
                    dist.all_reduce(parameter.grad, group=self.process_group)
                    parameter.grad.div_(self.world_size)
                continue

            if not info.is_full:
                # A sharded layer not used by this graph has no gradient.
                continue
            start = self.rank * info.shard_numel

            if parameter.grad is None:
                local_grad = None
            else:
                full_grad = parameter.grad.detach()
                if self.world_size > 1:
                    dist.all_reduce(full_grad, group=self.process_group)
                    full_grad.div_(self.world_size)
                full_grad = full_grad.to(torch.float32).reshape(-1)
                local_grad = self._padded_slice(full_grad, start, info.shard_numel)

            parameter.grad = None
            parameter.data = info.master_shard
            if local_grad is not None:
                parameter.grad = local_grad
            info.is_full = False

    @staticmethod
    def _padded_slice(tensor: torch.Tensor, start: int, length: int) -> torch.Tensor:
        result = tensor.new_zeros(length)
        available = max(0, min(length, tensor.numel() - start))
        if available:
            result[:available].copy_(tensor.narrow(0, start, available))
        return result
    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Return detached FP32 full parameters without changing shard state."""

        result: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            info = self._shards.get(parameter)
            if info is None:
                result[name] = parameter.detach().clone()
            elif info.is_full:
                result[name] = parameter.detach().to(torch.float32).clone()
            else:
                local = parameter.detach().to(torch.float32).contiguous()
                if self.world_size == 1:
                    gathered = local
                else:
                    pieces = [torch.empty_like(local) for _ in range(self.world_size)]
                    dist.all_gather(pieces, local, group=self.process_group)
                    gathered = torch.cat(pieces)
                result[name] = gathered[: info.numel].view(info.shape).clone()
        return result


from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class DistributedDataParallel(nn.Module):
    """A small DDP wrapper that overlaps per-parameter reduction with backward."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before constructing "
                "DistributedDataParallel"
            )

        self.module = module
        self._world_size = dist.get_world_size()
        self._pending_reductions: list[
            tuple[dist.Work | None, torch.Tensor, nn.Parameter]
        ] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._broadcast_module_state()
        self._register_gradient_hooks()

    @staticmethod
    def _unique_tensors(tensors: Any) -> list[torch.Tensor]:
        seen: set[int] = set()
        unique: list[torch.Tensor] = []
        for tensor in tensors:
            tensor_id = id(tensor)
            if tensor_id not in seen:
                seen.add(tensor_id)
                unique.append(tensor)
        return unique

    def _broadcast_module_state(self) -> None:
        # Parameters and persistent/non-persistent buffers are both part of the
        # module's runtime state. Deduplication is necessary for tied tensors.
        tensors = self._unique_tensors(
            list(self.module.parameters()) + list(self.module.buffers())
        )
        with torch.no_grad():
            for tensor in tensors:
                dist.broadcast(tensor, src=0)

    def _register_gradient_hooks(self) -> None:
        parameters = self._unique_tensors(self.module.parameters())
        for parameter in parameters:
            if not parameter.requires_grad:
                continue

            captured_grads: list[torch.Tensor] = []

            def capture_gradient(
                gradient: torch.Tensor,
                *,
                captured_grads: list[torch.Tensor] = captured_grads,
            ) -> torch.Tensor:
                # Preserve each backward's contribution separately. Returning
                # zeros prevents a local gradient from being mixed with an
                # already globally-reduced gradient on repeated backward calls.
                captured_grads.append(gradient.detach())
                return torch.zeros_like(gradient)

            def reduce_gradient(
                param: nn.Parameter,
                *,
                captured_grads: list[torch.Tensor] = captured_grads,
            ) -> None:
                if not captured_grads:
                    return
                gradient = captured_grads.pop(0).clone()
                gradient.div_(self._world_size)
                work = dist.all_reduce(gradient, async_op=True)
                self._pending_reductions.append((work, gradient, param))

            self._hook_handles.append(parameter.register_hook(capture_gradient))
            self._hook_handles.append(
                parameter.register_post_accumulate_grad_hook(reduce_gradient)
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for outstanding reductions and accumulate their results."""
        pending, self._pending_reductions = self._pending_reductions, []
        for work, gradient, parameter in pending:
            if work is not None:
                work.wait()
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)


__all__ = ["DistributedDataParallel"]

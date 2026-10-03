"""Tests for FullyShardedDataParallel (FSDP) implementation."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from .adapters import (
    fsdp_gather_full_params,
    fsdp_on_after_backward,
    get_fsdp,
)
from .common import (
    _cleanup_process_group,
    _setup_process_group,
)


class ToyFSDPModel(nn.Module):
    """Simple model using cs336_basics Linear and Embedding for FSDP testing."""

    def __init__(self, vocab_size=100, d_model=64, d_ff=128):
        super().__init__()
        from cs336_basics.model import Embedding, Linear, RMSNorm

        self.embedding = Embedding(vocab_size, d_model)
        self.norm1 = RMSNorm(d_model)
        self.linear1 = Linear(d_model, d_ff)
        self.norm2 = RMSNorm(d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm1(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.lm_head(x)
        return x


def _apply_mixed_precision_hooks(model, compute_dtype):
    """
    Apply hooks to a non-parallel model that replicate FSDP's mixed-precision
    behavior: cast Linear/Embedding weights to compute_dtype for
    forward/backward, keep master weights and optimizer updates in fp32.
    """
    from cs336_basics.model import Embedding, Linear

    for mod in model.modules():
        if not isinstance(mod, (Linear, Embedding)):
            continue

        # Forward: cast weight to compute_dtype, restore fp32 after
        def make_fwd_pre(dt):
            def hook(m, inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)

            return hook

        def make_fwd_post():
            def hook(m, inp, out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        mod.register_forward_hook(make_fwd_post())

        # Linear backward needs the weight in compute_dtype for grad_input
        if isinstance(mod, Linear):

            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        # After gradient is computed, restore fp32 weight and cast grad to fp32
        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("compute_dtype", [None, torch.float16], ids=["fp32", "fp16"])
def test_fsdp_correctness(compute_dtype):
    """Test that FSDP produces the same results as non-parallel training."""
    world_size = 2
    mp.spawn(
        _test_fsdp_correctness,
        args=(world_size, compute_dtype),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_correctness(rank: int, world_size: int, compute_dtype):
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    base_model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)

    # Non-parallel baseline (with matching mixed-precision if needed)
    non_parallel_model = deepcopy(base_model)
    if compute_dtype is not None:
        _apply_mixed_precision_hooks(non_parallel_model, compute_dtype)

    # FSDP model
    fsdp_model = get_fsdp(deepcopy(base_model), compute_dtype=compute_dtype)

    loss_fn = nn.CrossEntropyLoss()
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    non_parallel_optimizer = torch.optim.SGD(non_parallel_model.parameters(), lr=0.01)

    # Generate data
    torch.manual_seed(123)
    batch_size = 20
    seq_len = 8
    all_input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    all_labels = torch.randint(0, 100, (batch_size,), device=device)

    local_bs = batch_size // world_size

    for step in range(3):
        fsdp_optimizer.zero_grad(set_to_none=True)
        non_parallel_optimizer.zero_grad(set_to_none=True)

        # Non-parallel: forward on all data
        non_parallel_out = non_parallel_model(all_input_ids)
        non_parallel_loss = loss_fn(non_parallel_out[:, -1, :].float(), all_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # FSDP: each rank sees a different subset
        offset = rank * local_bs
        local_input = all_input_ids[offset : offset + local_bs]
        local_labels = all_labels[offset : offset + local_bs]
        fsdp_out = fsdp_model(local_input)
        fsdp_loss = loss_fn(fsdp_out[:, -1, :].float(), local_labels)
        fsdp_loss.backward()

        fsdp_on_after_backward(fsdp_model, fsdp_optimizer)
        fsdp_optimizer.step()

        # Compare all parameters
        full_params = fsdp_gather_full_params(fsdp_model)
        for name, np_param in non_parallel_model.named_parameters():
            fsdp_full = full_params[name]
            if compute_dtype is None:
                assert torch.allclose(np_param.data, fsdp_full, atol=1e-6, rtol=1e-4), (
                    f"Step {step}: Parameter {name} mismatch. Max diff: {(np_param.data - fsdp_full).abs().max().item()}"
                )
            else:
                assert torch.allclose(np_param.data, fsdp_full, atol=1e-4, rtol=1e-4), (
                    f"Step {step}: Parameter {name} mismatch. Max diff: {(np_param.data - fsdp_full).abs().max().item()}"
                )

        # Shuffle data
        torch.manual_seed(42 + step)
        perm = torch.randperm(batch_size)
        all_input_ids = all_input_ids[perm]
        all_labels = all_labels[perm]

    _cleanup_process_group()


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("compute_dtype", [None, torch.float16], ids=["fp32", "fp16"])
def test_fsdp_gradient_sync(compute_dtype):
    """Test that gradients are properly synchronized and correctly shaped."""
    world_size = 2
    mp.spawn(
        _test_fsdp_gradient_sync,
        args=(world_size, compute_dtype),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_gradient_sync(rank: int, world_size: int, compute_dtype):
    from cs336_basics.model import Embedding, Linear

    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)
    fsdp_model = get_fsdp(model, compute_dtype=compute_dtype)

    # Each rank gets different data
    torch.manual_seed(rank)
    input_ids = torch.randint(0, 100, (4, 8), device=device)

    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    out = fsdp_model(input_ids)
    loss = out.sum()
    loss.backward()

    fsdp_on_after_backward(fsdp_model, fsdp_optimizer)

    # After sync, every parameter must have a gradient matching its data shape
    # and in the master weight dtype (fp32), regardless of compute_dtype
    for name, param in fsdp_model.module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert param.grad.shape == param.data.shape, f"Gradient shape {param.grad.shape} != data shape {param.data.shape} for {name}"
            assert param.grad.dtype == param.data.dtype, f"Gradient dtype {param.grad.dtype} != data dtype {param.data.dtype} for {name}"

    # Replicated (non-FSDP) parameter gradients must be identical across ranks
    for name, param in fsdp_model.module.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.rsplit(".", 1)
        mod = dict(fsdp_model.module.named_modules())[parts[0]] if len(parts) == 2 else fsdp_model.module
        if isinstance(mod, (Linear, Embedding)):
            continue
        gathered = [torch.zeros_like(param.grad) for _ in range(world_size)]
        dist.all_gather(gathered, param.grad)
        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4, rtol=1e-4), (
                f"Replicated gradient for {name} differs between rank 0 and rank {r}. Max diff: {(gathered[0] - gathered[r]).abs().max().item()}"
            )

    _cleanup_process_group()

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr


@dataclass
class TrainConfig:
    train_data: str
    val_data: str
    output_dir: str = "outputs/run"
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float | None = 10_000.0
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 5_000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    save_checkpoints: bool = True
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = False
    remove_rmsnorm: bool = False
    use_post_norm: bool = False
    ffn_type: str = "swiglu"
    tie_embeddings: bool = False
    tied_embedding_init_std: float = 0.02
    resume: str | None = None


def _load_tokens(path: str) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.load(file_path, mmap_mode="r")
    return np.memmap(file_path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainConfig,
) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
        logits = model(inputs)
        losses.append(cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item())
    model.train()
    return float(sum(losses) / len(losses))


def train(config: TrainConfig) -> dict[str, object]:
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": config.device,
        "device_name": torch.cuda.get_device_name(config.device) if config.device.startswith("cuda") else "CPU",
    }
    train_data = _load_tokens(config.train_data)
    val_data = _load_tokens(config.val_data)
    model = BasicsTransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
        config.remove_rmsnorm,
        config.use_post_norm,
        config.ffn_type,
        config.tie_embeddings,
        config.tied_embedding_init_std,
    ).to(config.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    environment["parameter_count"] = parameter_count
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    start_step = load_checkpoint(config.resume, model, optimizer) if config.resume else 0
    runtime_model = cast(torch.nn.Module, torch.compile(model)) if config.compile_model else model

    use_amp = config.device.startswith("cuda") and config.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = (
        start_step * config.batch_size * config.context_length * config.gradient_accumulation_steps
    )
    best_val = math.inf
    log_path = output_dir / "metrics.jsonl"
    if start_step == 0:
        log_path.unlink(missing_ok=True)
    if config.resume and log_path.exists():
        previous_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        if previous_records:
            best_val = min(float(record["val_loss"]) for record in previous_records)
    for step in range(start_step, config.max_steps):
        lr = get_cosine_lr(step, config.max_lr, config.min_lr, config.warmup_steps, config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        try:
            for _ in range(config.gradient_accumulation_steps):
                inputs, targets = get_batch(train_data, config.batch_size, config.context_length, config.device)
                with torch.autocast(device_type=config.device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                    logits = runtime_model(inputs)
                    micro_loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = micro_loss / config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += micro_loss.detach().item()
        except torch.OutOfMemoryError:
            failure = {
                "step": step,
                "status": "oom",
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
            }
            with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(failure) + "\n")
            raise
        gradient_norm = clip_gradient(runtime_model.parameters(), config.max_grad_norm)
        optimizer.step()

        if step % config.eval_interval == 0 or step + 1 == config.max_steps:
            val_loss = evaluate(runtime_model, val_data, config)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            elapsed = time.perf_counter() - started
            tokens_seen = (
                (step + 1)
                * config.batch_size
                * config.context_length
                * config.gradient_accumulation_steps
            )
            now = time.perf_counter()
            record = {
                "step": step,
                "train_loss": accumulated_loss / config.gradient_accumulation_steps,
                "val_loss": val_loss,
                "lr": lr,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_second": (tokens_seen - last_log_tokens) / max(now - last_log_time, 1e-9),
                "peak_memory_mib": (
                    torch.cuda.max_memory_allocated(config.device) / 2**20
                    if config.device.startswith("cuda")
                    else 0.0
                ),
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(config.device)
                    if config.device.startswith("cuda")
                    else 0
                ),
            }
            with log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            last_log_time, last_log_tokens = now, tokens_seen
            if is_best and config.save_checkpoints:
                save_checkpoint(model, optimizer, step + 1, output_dir / "checkpoint-best.pt")
        if config.save_checkpoints and (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step + 1, output_dir / f"checkpoint-{step + 1:07d}.pt")

    if config.save_checkpoints:
        save_checkpoint(model, optimizer, config.max_steps, output_dir / "checkpoint-final.pt")
    summary: dict[str, object] = {
        "steps": config.max_steps,
        "best_val_loss": best_val,
        "elapsed_seconds": time.perf_counter() - started,
        "tokens_seen": (
            config.max_steps
            * config.batch_size
            * config.context_length
            * config.gradient_accumulation_steps
        ),
        "parameter_count": parameter_count,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(config.device) / 2**20 if config.device.startswith("cuda") else 0.0
        ),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else 0
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


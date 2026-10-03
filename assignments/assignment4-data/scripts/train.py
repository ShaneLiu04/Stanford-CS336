"""
Train a language model on one or multiple GPUs.

The training config is hard-coded below. Pass the path to your GPT-2-tokenized
dataset with `--train-bin`.

To run single-GPU training:

```
uv run python scripts/train.py --train-bin /root/data/your_data.bin
```

To run the final multi-GPU training job on Modal with 8 B200 GPUs:

```
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import wandb
from rich.pretty import pprint as pprint
from rich.traceback import install
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr
from cs336_basics.train_config import Config, PathsConfig
from cs336_data.modal_utils import MODAL_SECRETS, VOLUME_MOUNTS, app, build_image

logger = logging.getLogger(__name__)
N_GPUS = 8
DEFAULT_MODAL_VALID_BIN = "/shared-data/tokenized_paloma_c4_100_domains_validation.bin"
DEFAULT_MODAL_MODEL_OUTPUT = "/root/data/output/your_data"
LOCAL_DATA_DIR = Path("/tmp/data")
EPHEMERAL_DISK_MB = 52_4288

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

install(show_locals=True)


def train_from_config(cfg: Config) -> None:
    train_data = np.memmap(cfg.paths.train_bin, dtype=np.uint16, mode="r")
    dev_data = np.memmap(cfg.paths.valid_bin, dtype=np.uint16, mode="r")
    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )

    # Wrap model in DDP, if we're using it.
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        seed = cfg.training.seed + ddp_rank  # each process gets a different seed
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
        if is_master_process:
            logger.info("Using DDP")
    else:
        seed = cfg.training.seed
        ddp_world_size = 1
        is_master_process = True

    if is_master_process:
        pprint(cfg)
        pprint(model)
        logger.info(
            "Total number of tokens per training step: "
            + str(
                cfg.training.gradient_accumulation_steps
                * ddp_world_size
                * cfg.training.train_batch_size
                * cfg.model.context_length
            )
        )
        if cfg.training.wandb_project and cfg.training.wandb_entity:
            wandb.init(
                # Set the project where this run will be logged
                entity=cfg.training.wandb_entity,
                project=cfg.training.wandb_project,
                config=asdict(cfg),
                name=cfg.paths.model_output.name,
            )

    # Seed each process differently so we can be sure that they
    # see different data batches.
    # NOTE: This assumes that you're using torch RNG, you may have
    # to seed numpy too as well if your code uses numpy random functions.
    torch.manual_seed(seed)

    # Save the model config
    if is_master_process:
        cfg.paths.model_output.mkdir(parents=True, exist_ok=True)
        model_config_output_path = cfg.paths.model_output / "model_config.json"
        logger.info(f"Saving model config to {model_config_output_path}")
        model_config = model.config
        with open(model_config_output_path, "w") as f:
            json.dump(model_config, f, indent=4)

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.training.dtype]
    if is_master_process:
        logger.info(f"Using dtype: {torch_dtype}")

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)

    # Move model to the device
    model = model.to(cfg.training.device)

    # compile the model, requires torch 2.0
    if cfg.training.compile:
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": cfg.training.weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
        fused=True,
    )

    # Get the first batch
    batch_x, batch_y = get_batch(
        train_data,
        batch_size=cfg.training.train_batch_size,
        context_length=cfg.model.context_length,
        device=cfg.training.device,
    )
    for i in (pbar := trange(cfg.training.train_steps, desc="Training", disable=not is_master_process)):
        lr = get_cosine_lr(
            i,
            max_learning_rate=cfg.training.lr,
            min_learning_rate=cfg.training.lr * 0.1,
            warmup_iters=int(cfg.training.train_steps * cfg.training.warmup_ratio),
            cosine_cycle_iters=cfg.training.train_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step_idx in range(cfg.training.gradient_accumulation_steps):
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = micro_step_idx == cfg.training.gradient_accumulation_steps - 1

            with amp_ctx:
                logits = model(batch_x)

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch_x, next_batch_y = get_batch(
                    train_data,
                    batch_size=cfg.training.train_batch_size,
                    context_length=cfg.model.context_length,
                    device=cfg.training.device,
                )

                # Calculate the loss with the logits
                loss = (
                    F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    / cfg.training.gradient_accumulation_steps
                )

            loss.backward()

            batch_x = next_batch_x
            batch_y = next_batch_y

        if cfg.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_float = loss.item() * cfg.training.gradient_accumulation_steps

        if is_master_process:
            pbar.set_description(f"Training step {i}, Loss: {loss_float:.4f}")
            if cfg.training.wandb_project and i % cfg.training.log_interval == 0:
                wandb.log({"train_loss": loss_float, "lr": lr}, step=i)

        if i != 0 and i % cfg.training.eval_interval == 0 and is_master_process:
            dev_loss = estimate_dev_loss(
                model=model,
                dev_dataset=dev_data,
                batch_size=cfg.training.eval_batch_size,
                eval_iters=cfg.training.eval_iterations,
                device=cfg.training.device,
                context_length=cfg.model.context_length,
            )
            logger.info(f"Estimated validation loss: {dev_loss}")
            if cfg.training.wandb_project:
                wandb.log({"eval_loss": dev_loss}, step=i)

            if cfg.training.save_checkpoints:
                model_weights_output_path = cfg.paths.model_output / f"step_{i:010d}" / "model.pt"
                model_weights_output_path.parent.mkdir(parents=True, exist_ok=True)

                # Need both config and weights to load the model
                # Write config:
                with open(model_weights_output_path.parent / "model_config.json", "w") as f:
                    json.dump(model_config, f, indent=4)

                # Write weights:
                torch.save(model.state_dict(), model_weights_output_path)

    # Calculate final estimated dev loss
    if is_master_process:
        dev_loss = estimate_dev_loss(
            model=model,
            dev_dataset=dev_data,
            batch_size=cfg.training.eval_batch_size,
            eval_iters=cfg.training.eval_iterations,
            device=cfg.training.device,
            context_length=cfg.model.context_length,
        )
        logger.info(f"Final estimated validation loss: {dev_loss}")
        if cfg.training.wandb_project:
            wandb.log({"eval_loss": dev_loss}, step=cfg.training.train_steps)

        # Save the model weights
        model_weights_output_path = cfg.paths.model_output / "model.pt"
        logger.info(f"Saving model weights to {model_weights_output_path}")
        torch.save(model.state_dict(), model_weights_output_path)

    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def estimate_dev_loss(
    model: BasicsTransformerLM,
    dev_dataset: npt.NDArray,
    batch_size: int,
    eval_iters: int,
    device: str,
    context_length: int,
):
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in tqdm(range(eval_iters)):
        batch_x, batch_y = get_batch(
            dev_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[k] = loss.item()

    model.train()
    return losses.mean()


def copy_data_to_local_disk(*paths: Path) -> list[Path]:
    """Copy data files from the network volume to Modal's ephemeral disk."""
    print("Copying data to local disk")
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(path.stat().st_size for path in paths)
    ephemeral_bytes = EPHEMERAL_DISK_MB * 1_000_000
    if total_bytes > ephemeral_bytes:
        raise ValueError(f"Data files require {total_bytes / 1_000_000:.0f} MB, which exceeds {EPHEMERAL_DISK_MB=}")

    local_paths: list[Path] = []
    for source in paths:
        destination = LOCAL_DATA_DIR / source.name
        local_paths.append(destination)
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            logger.info(f"Local data copy already exists: {destination}")
            continue

        copy_started_at = time.perf_counter()
        tmp_destination = destination.with_suffix(f"{destination.suffix}.tmp")
        logger.info(f"Copying {source} to {destination}")
        shutil.copy2(source, tmp_destination)
        tmp_destination.rename(destination)
        logger.info(f"Copied {source} in {time.perf_counter() - copy_started_at:.1f}s")
    print("Finished copying data to local disk")

    return local_paths


def run_modal_worker() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", default=DEFAULT_MODAL_VALID_BIN)
    args = parser.parse_args()

    train_from_config(
        Config(
            paths=PathsConfig(
                train_bin=Path(args.train_bin),
                valid_bin=Path(args.valid_bin),
                model_output=Path(DEFAULT_MODAL_MODEL_OUTPUT),
            ),
        )
    )


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    secrets=MODAL_SECRETS,
    gpu=f"B200:{N_GPUS}",
    timeout=60 * 60 * 12,
    ephemeral_disk=EPHEMERAL_DISK_MB,
)
def run_modal_training(
    train_bin: str,
) -> None:
    local_train_bin, local_valid_bin = copy_data_to_local_disk(Path(train_bin), Path(DEFAULT_MODAL_VALID_BIN))

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={N_GPUS}",
        "train.py",
        "--modal-worker",
        "--train-bin",
        str(local_train_bin),
        "--valid-bin",
        str(local_valid_bin),
    ]

    subprocess.run(command, cwd="/root", check=True)


@app.local_entrypoint()
def modal_main(
    train_bin: str,
) -> None:
    run_modal_training.remote(
        train_bin=train_bin,
    )


if __name__ == "__main__":
    if "--modal-worker" in sys.argv:
        sys.argv.remove("--modal-worker")
        run_modal_worker()
    else:
        run_modal_worker()

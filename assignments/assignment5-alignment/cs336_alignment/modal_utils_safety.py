"""Modal helpers for running optional safety/RLHF supplement jobs.

Usage: see modal_utils.py.
"""

from __future__ import annotations

import subprocess

import modal

from cs336_alignment.modal_utils import (
    GPU,
    MAX_CONTAINERS,
    SUNET_ID,
    image,
    quote_command,
    wandb_secret,
)


app = modal.App(f"cs336-a5-supplement-{SUNET_ID}")

SHARED_VOLUME_NAME = "cs336-a5-supplement"
SHARED_VOLUME_ENVIRONMENT = "cs336-shared-data"
SHARED_VOLUME_MOUNT_PATH = "/mnt/cs336-a5-supplement"
RESULTS_VOLUME_NAME = f"cs336-a5-supplement-results-{SUNET_ID}"
RESULTS_VOLUME_MOUNT_PATH = "/mnt/cs336-a5-supplement-results"

BASE_MODEL_PATH = f"{SHARED_VOLUME_MOUNT_PATH}/models/Meta-Llama-3.1-8B"
JUDGE_MODEL_PATH = f"{SHARED_VOLUME_MOUNT_PATH}/models/Llama-3.3-70B-Instruct"
SFT_DATA_DIR = (
    f"{SHARED_VOLUME_MOUNT_PATH}/data/safety_augmented_ultrachat_200k_single_turn"
)
SFT_TRAIN_PATH = f"{SFT_DATA_DIR}/train.jsonl.gz"
SFT_DEV_PATH = f"{SFT_DATA_DIR}/test.jsonl.gz"
RUN_TIMEOUT_SECONDS = 20 * 60 * 60

shared_volume = modal.Volume.from_name(
    SHARED_VOLUME_NAME,
    environment_name=SHARED_VOLUME_ENVIRONMENT,
)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNTS = {
    SHARED_VOLUME_MOUNT_PATH: shared_volume,
    RESULTS_VOLUME_MOUNT_PATH: results_volume,
}


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes=VOLUME_MOUNTS,
    secrets=[wandb_secret],
)
def run_command(command: list[str]) -> str:
    command_str = quote_command(command)
    print(command_str, flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        results_volume.commit()
    return command_str


def submit_commands(commands: list[list[str]]) -> None:
    print(
        f"Submitting {len(commands)} Modal supplement jobs "
        f"with max_containers={MAX_CONTAINERS}, gpu={GPU}, "
        f"timeout={RUN_TIMEOUT_SECONDS}s.",
        flush=True,
    )
    failures = []
    for idx, result in enumerate(run_command.map(commands, return_exceptions=True)):
        command_str = quote_command(commands[idx])
        if isinstance(result, BaseException):
            print(f"Failed: {command_str}", flush=True)
            print(f"Error: {result!r}", flush=True)
            failures.append(command_str)
        else:
            print(f"Completed: {result}", flush=True)

    if failures:
        print(f"{len(failures)} of {len(commands)} Modal jobs failed.", flush=True)
        raise SystemExit(1)

"""Modal helpers for running assignment 5 jobs.

Example usage:

import sys
from cs336_alignment.modal_utils import app, quote_command, submit_commands

def build_run_commands(args):
    # Suppose args.seeds = '0,1,2,3'
    return [
        [sys.executable, "-u", "scripts/grpo.py", "--seed", seed]
        for seed in args.seeds.split(',')
    ]

@app.local_entrypoint(name=...)
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)
"""

from __future__ import annotations

import shlex
import subprocess

import modal


SUNET_ID = "TODO"  # NOTE: modal_utils.py should remain unchanged other than adding your SUNET_ID.
if SUNET_ID == "TODO":
    raise ValueError("Please set SUNET_ID in cs336_alignment/modal_utils.py before running Modal jobs.")


GPU = "B200:2"
MAX_CONTAINERS = 4
REMOTE_ROOT = "/root"
RUN_TIMEOUT_SECONDS = 60 * 60
WANDB_SECRET_NAME = "wandb"

app = modal.App(f"cs336-a5-rlvr-{SUNET_ID}")
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_sync(extras=["gpu"])
    .workdir(REMOTE_ROOT)
    .add_local_dir("cs336_alignment", f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("data", f"{REMOTE_ROOT}/data")
    .add_local_dir("experiments", f"{REMOTE_ROOT}/experiments")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock")
)


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
)
def run_command(command: list[str]) -> str:
    command_str = quote_command(command)
    print(command_str, flush=True)
    subprocess.run(command, check=True)
    return command_str


def submit_commands(commands: list[list[str]]) -> None:
    print(
        f"Submitting {len(commands)} Modal jobs "
        f"with max_containers={MAX_CONTAINERS}, gpu={GPU}, "
        f"timeout={RUN_TIMEOUT_SECONDS}s.",
        flush=True,
    )
    failures = []
    for idx, result in enumerate(run_command.map(commands, return_exceptions=True)):
        command = commands[idx]
        command_str = quote_command(command)
        if isinstance(result, BaseException):
            print(f"Failed: {command_str}", flush=True)
            print(f"Error: {result!r}", flush=True)
            failures.append(command_str)
        else:
            print(f"Completed: {result}", flush=True)

    if failures:
        print(f"{len(failures)} of {len(commands)} Modal jobs failed.", flush=True)
        raise SystemExit(1)

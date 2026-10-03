"""
Small vLLM helpers for server lifecycle, completion requests, and NCCL weight sync.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


@dataclass
class VLLMServer:
    model_id: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu: int = 1
    seed: int = 0
    load_format: str = "auto"
    logging_level: str = "ERROR"
    gpu_memory_utilization: float = 0.9
    launch_server: bool = True
    startup_timeout: int = 600
    shutdown_timeout: int = 30

    def __post_init__(self) -> None:
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        self.weight_sync_group = None

    def start(self) -> None:
        if self.launch_server:
            kill_existing_vllm_server(self.port)
            self.process = start_server(
                model_id=self.model_id,
                host=self.host,
                port=self.port,
                gpu=self.gpu,
                seed=self.seed,
                load_format=self.load_format,
                logging_level=self.logging_level,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            atexit.register(self.stop)
        wait_for_server(self.base_url, self.process, self.startup_timeout)

    def stop(self) -> None:
        stop_server(self.process, timeout=self.shutdown_timeout)

    def init_weight_sync(self, policy_device: str):
        self.weight_sync_group = init_weight_sync(self.base_url, policy_device)
        return self.weight_sync_group

    def sync_policy_weights(self, policy: torch.nn.Module) -> None:
        if self.weight_sync_group is None:
            raise RuntimeError("Call init_weight_sync before sync_policy_weights.")
        sync_policy_weights(policy, self.base_url, self.weight_sync_group)

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
        return generate_completions(
            vllm_base_url=self.base_url,
            model_id=self.model_id,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body)


def kill_existing_vllm_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        result = subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        if result.returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass


def start_server(
    model_id: str,
    host: str,
    port: int,
    gpu: int,
    seed: int,
    load_format: str,
    logging_level: str,
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_LOGGING_LEVEL"] = logging_level
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        "1",
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--load-format",
        load_format,
    ]
    logger.info("Starting vLLM server: %s", " ".join(command))
    return subprocess.Popen(command, env=env, start_new_session=True)


def wait_for_server(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def stop_server(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def generate_completions(
    vllm_base_url: str,
    model_id: str,
    prompts: list[str],
    sampling_params: dict,
    batch_size: int | None = None,
) -> list[VLLMCompletion]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    prompt_batches = [prompts]
    if batch_size is not None:
        prompt_batches = [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]

    completions = []
    for prompt_batch in prompt_batches:
        payload = {
            "model": model_id,
            "prompt": prompt_batch,
            "temperature": sampling_params["temperature"],
            "max_tokens": sampling_params["max_tokens"],
            "n": sampling_params["n"],
            "seed": sampling_params["seed"],
            "return_token_ids": True,
        }
        if sampling_params.get("stop") is not None:
            payload["stop"] = sampling_params["stop"]
            payload["include_stop_str_in_output"] = sampling_params.get("include_stop_str_in_output", False)

        response = _http_json("POST", f"{vllm_base_url}/v1/completions", payload, timeout=3600)
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        completions.extend(
            VLLMCompletion(
                text=choice["text"],
                token_ids=choice.get("token_ids") or [],
                finish_reason=choice.get("finish_reason"),
            )
            for choice in choices
        )
    return completions


def init_weight_sync(vllm_base_url: str, policy_device: str):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    inference_world_size = _http_json("GET", f"{vllm_base_url}/get_world_size", timeout=10)["world_size"]
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()
    init_info = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": world_size,
    }

    torch.cuda.set_device(torch.device(policy_device))
    with ThreadPoolExecutor(max_workers=1) as executor:
        init_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/init_weight_transfer_engine",
            {"init_info": init_info},
            60,
        )
        weight_sync_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_future.result()

    return weight_sync_group


def sync_policy_weights(policy: torch.nn.Module, vllm_base_url: str, weight_sync_group) -> None:
    """Copy policy weights into vLLM and invalidate caches derived from old weights."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    weights = list(policy.named_parameters())
    update_info = {
        "names": [name for name, _ in weights],
        "dtype_names": [str(tensor.dtype).split(".")[-1] for _, tensor in weights],
        "shapes": [list(tensor.shape) for _, tensor in weights],
        "packed": True,
    }

    torch.cuda.set_device(next(policy.parameters()).device)
    _http_json("POST", f"{vllm_base_url}/pause", timeout=60)
    with ThreadPoolExecutor(max_workers=1) as executor:
        update_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/update_weights",
            {"update_info": update_info},
            300,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(weights),
            trainer_args=NCCLTrainerSendWeightsArgs(
                group=weight_sync_group,
                packed=True,
            ),
        )
        update_future.result()
    _http_json("POST", f"{vllm_base_url}/reset_prefix_cache", timeout=60)
    _http_json("POST", f"{vllm_base_url}/resume", timeout=60)

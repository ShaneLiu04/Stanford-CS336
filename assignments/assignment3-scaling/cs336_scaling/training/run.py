import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import equinox.nn as nn
import jax
import modal
import wandb
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from cs336_scaling.api.internal import (
    CompletedPayload,
    FailedPayload,
)
from cs336_scaling.internal_client import (
    post_internal_finish,
    post_internal_worker_event,
)
from cs336_scaling.log.setup import configure_logging
from cs336_scaling.modal_utils import MODAL_SECRETS, VOLUME_MOUNTS, app, build_image
from cs336_scaling.schemas.event import (
    PreemptedEvent,
    RunHeartbeatEvent,
    RunStartedEvent,
    ValidationLossEvent,
)
from cs336_scaling.schemas.experiment import TimeoutReason, UnexpectedReason
from cs336_scaling.tokenized_data import DCLMData
from cs336_scaling.training.data import Batch, download_data
from cs336_scaling.training.loop import OuterLossResult, outer_loss
from cs336_scaling.training.model.basic_model import (
    BasicCausalLM,
    BasicTransformerConfig,
)
from cs336_scaling.training.model.jax_utils import (
    count_params,
    show_memory_analysis,
    tree_rearrange,
)
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig
from cs336_scaling.utils import format_bytes, format_params

logger = logging.getLogger(__name__)


def default_training_config() -> TrainingConfig:
    return TrainingConfig(
        architecture_config=BasicTransformerConfig(
            attention_bias=False,
            head_dim=64,
            hidden_size=448,
            intermediate_size=1280,
            num_attention_heads=7,
            num_hidden_layers=9,
            num_key_value_heads=7,
            rms_norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=False,
            dtype="bfloat16",
            vocab_size=32000,
        ),
        optimizer_config=AdamWConfig(),
        train_batch_size=128,
        val_batch_size=32,
        total_train_tokens=16 * 2**16,
        max_runtime_seconds=30,
        model_seed=0,
    )


@dataclass
class TrainingResult:
    val_losses: list[float]
    estimated_memory: float
    training_time: float
    model_params: int


@app.function(
    image=build_image(include_tailscale=True),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
    secrets=MODAL_SECRETS,
    gpu="B200",
    ephemeral_disk=1099511,
)
def run_training(
    training_config: TrainingConfig, sunet_id: str, experiment_id: int
) -> TrainingResult:
    train_started_at: float | None = None
    modal_function_call_id = modal.current_function_call_id()
    if modal_function_call_id is None:
        raise ValueError(
            "modal function call id is unavailable outside a Modal function call"
        )
    (output_path := Path(f"output/{sunet_id}/{experiment_id}")).mkdir(
        parents=True, exist_ok=True
    )
    configure_logging(log_file=output_path.resolve() / "log.log", log_format="json")
    (output_path / "training_config.json").write_text(
        training_config.model_dump_json(indent=2) + "\n"
    )

    log_extra = {
        "experiment_id": experiment_id,
        "sunet_id": sunet_id,
        "training_config_unique_id": training_config.unique_id,
        "modal_function_call_id": modal_function_call_id,
    }
    logger.info(
        "training_started",
        extra=log_extra
        | {
            "training_config": training_config.model_dump(mode="json"),
            "output_path": output_path,
        },
    )
    wandb_run = wandb.init(
        entity="hashimoto-group",
        project="cs336-scaling-2026",
        name=f"{sunet_id}-{experiment_id}-{training_config.unique_id}",
        id=f"{sunet_id}-{experiment_id}-{training_config.unique_id}",
        group=sunet_id,
        config=training_config.model_dump(mode="json") | log_extra,
        dir=output_path,
        resume="allow",
    )
    try:
        post_internal_worker_event(
            experiment_id,
            RunStartedEvent(
                run_id=modal_function_call_id,
                wandb_path=wandb_run.path,
            ),
            log_extra=log_extra,
        )
    except Exception:
        logger.exception("training_started_event_report_failed", extra=log_extra)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

    data = DCLMData()

    def elapsed_training_seconds() -> float:
        if train_started_at is None:
            return 0.0
        return time.perf_counter() - train_started_at

    def run() -> TrainingResult:
        nonlocal train_started_at
        mesh = jax.make_mesh(
            (jax.device_count(),), ("fsdp",), axis_types=(AxisType.Explicit,)
        )
        jax.set_mesh(mesh)
        data_sharding = NamedSharding(mesh, P(None, "fsdp", None))

        def put_batched_data(batch: Batch, batch_size: int) -> Batch[Array]:
            return jax.device_put(
                tree_rearrange(
                    batch.to_jax(),
                    "(loops batch) ... -> loops batch ...",
                    batch=batch_size,
                ),
                data_sharding,
            )

        def train_data(
            batch_idx: int,
        ) -> Batch[Array]:
            return put_batched_data(
                Batch.train_data(
                    data_result,
                    batch_idx=batch_idx,
                    training_config=training_config,
                ),
                batch_size=training_config.train_batch_size,
            )

        logger.info("data_download_started", extra=log_extra)
        data_download_started_at = time.perf_counter()
        data_result = download_data(data.try_load(), training_config)
        data_download_seconds = time.perf_counter() - data_download_started_at
        logger.info(
            "data_download_completed",
            extra=log_extra
            | {
                "download_seconds": round(data_download_seconds, 3),
                "train_files": len(data_result.train_filepaths),
                "n_train_tokens_per_chunk": data_result.n_train_tokens_per_chunk,
                "n_validation_tokens": data_result.n_validation_tokens,
                "validation_filepath": str(data_result.validation_filepath),
            },
        )
        logger.info("validation_data_loading_started", extra=log_extra)
        val_data = put_batched_data(
            Batch.val_data(
                data_result,
                training_config=training_config,
            ),
            batch_size=training_config.val_batch_size,
        )
        logger.info("validation_data_loading_completed", extra=log_extra)

        @jax.jit
        def make_model() -> tuple[BasicCausalLM, nn.State]:
            model, state = cast(
                tuple[BasicCausalLM, nn.State],
                nn.make_with_state(BasicCausalLM)(
                    training_config.architecture_config,
                    key=jax.random.PRNGKey(training_config.model_seed),
                ),
            )
            model = model.apply_sharding(mesh)
            return model, state

        model, state = make_model()

        model_params = count_params(model)

        logger.info(
            "model_created",
            extra=log_extra | {"model_params": model_params},
        )
        wandb.log({"model_params": model_params})
        print(f"training a {format_params(model_params)} model")

        optimizer = training_config.optimizer_config.build(training_config)
        opt_state = optimizer.init(model)

        logger.info("compilation_started", extra=log_extra)
        compile_start = time.perf_counter()
        estimated_memory = show_memory_analysis(
            outer_loss,
            model,
            state,
            train_data(batch_idx=0),
            val_data,
            training_config,
            opt_state,
        )
        compile_seconds = time.perf_counter() - compile_start
        logger.info(
            "compilation_completed",
            extra=log_extra
            | {
                "compile_seconds": round(compile_seconds, 3),
                "estimated_memory_bytes": estimated_memory,
            },
        )
        wandb.log(
            {
                "compile_seconds": compile_seconds,
                "estimated_memory_bytes": estimated_memory,
            }
        )
        print(f"compiled in {compile_seconds:.2f}s")
        logger.info("training_loop_started", extra=log_extra)
        timeout_report_lock = threading.Lock()
        timeout_reported: bool = False
        val_losses: list[float] = []

        def post_training_timed_out(elapsed_seconds: float) -> None:
            nonlocal timeout_reported
            with timeout_report_lock:
                if timeout_reported:
                    return
                post_internal_finish(
                    experiment_id,
                    FailedPayload(
                        result_type="failed",
                        used_runtime_seconds=elapsed_seconds,
                        reason=TimeoutReason(partial_val_losses=val_losses),
                    ),
                    log_extra=log_extra,
                )
                timeout_reported = True

        def cancel_modal_call_after_timeout() -> None:
            grace_max_runtime = training_config.max_runtime_seconds * 1.05 + 10
            assert train_started_at is not None
            cancel_at = train_started_at + grace_max_runtime
            while (remaining_seconds := cancel_at - time.perf_counter()) > 0:
                elapsed_seconds = elapsed_training_seconds()
                try:
                    post_internal_worker_event(
                        experiment_id,
                        RunHeartbeatEvent(
                            run_id=modal_function_call_id,
                            elapsed_seconds=elapsed_seconds,
                        ),
                        log_extra=log_extra,
                    )
                except Exception:
                    logger.exception(
                        "run_heartbeat_event_report_failed",
                        extra=log_extra
                        | {"elapsed_seconds": round(elapsed_seconds, 3)},
                    )
                time.sleep(min(60, remaining_seconds))
            elapsed_seconds = elapsed_training_seconds()
            assert elapsed_seconds > training_config.max_runtime_seconds
            logger.warning(
                "training_runtime_exceeded_canceling_modal_call",
                extra=log_extra
                | {
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "max_runtime_seconds": training_config.max_runtime_seconds,
                },
            )
            try:
                post_training_timed_out(elapsed_seconds)
            finally:
                modal.FunctionCall.from_id(modal_function_call_id).cancel(
                    terminate_containers=True
                )

        def raise_if_training_timed_out(
            elapsed_seconds: float, max_runtime_seconds: float
        ) -> None:
            if elapsed_seconds <= max_runtime_seconds:
                return
            post_training_timed_out(elapsed_seconds)
            raise TimeoutError(f"training exceeded {max_runtime_seconds}s")

        def post_validation_loss_event(val_losses: list[float]) -> None:
            try:
                post_internal_worker_event(
                    experiment_id,
                    ValidationLossEvent(
                        run_id=modal_function_call_id,
                        val_losses=val_losses,
                    ),
                    log_extra=log_extra,
                )
            except Exception:
                logger.exception(
                    "validation_loss_event_report_failed",
                    extra=log_extra | {"val_losses": val_losses},
                )

        res: OuterLossResult | None = None
        train_started_at = time.perf_counter()
        threading.Thread(target=cancel_modal_call_after_timeout, daemon=True).start()
        for i in range(training_config.n_evals):
            raise_if_training_timed_out(
                elapsed_training_seconds(),
                training_config.max_runtime_seconds,
            )

            chunk_started_at = time.perf_counter()
            logger.info(
                "training_chunk_started",
                extra=log_extra | {"chunk_index": i},
            )
            res: OuterLossResult = outer_loss(
                model,
                state,  # ty:ignore[too-many-positional-arguments]
                train_data(batch_idx=i),
                val_data,
                training_config,
                opt_state,
            )  # ty:ignore[invalid-assignment]
            model = res.model
            opt_state = res.opt_state
            chunk_seconds = time.perf_counter() - chunk_started_at
            tokens_seen = (i + 1) * training_config.eval_every_tokens
            val_loss = res.val_loss.item()
            logger.info(
                "training_chunk_completed",
                extra=log_extra
                | {
                    "chunk_index": i,
                    "chunk_seconds": round(chunk_seconds, 3),
                    "val_loss": val_loss,
                },
            )
            val_losses.append(val_loss)
            val_event_executor.submit(post_validation_loss_event, list(val_losses))
            wandb.log(
                {
                    "chunk": i,
                    "tokens": tokens_seen,
                    "train_progress_pct": tokens_seen
                    / training_config.total_train_tokens
                    * 100.0,
                    "optimizer_step": (i + 1)
                    * training_config.optimizer_steps_per_eval,
                    "train_loss": res.train_losses.mean().item(),
                    "val_loss": val_loss,
                    "chunk_seconds": chunk_seconds,
                }
            )
            print(f"{res.val_loss} {res.train_losses}")
        if res is None:
            raise RuntimeError("training_config did not schedule any training chunks")

        jax.block_until_ready(res)
        train_time = elapsed_training_seconds()
        raise_if_training_timed_out(train_time, training_config.max_runtime_seconds)
        print(res.train_losses)
        print(
            f"Trained to {res.val_loss=:.10f} in {train_time:.2f}s and used {format_bytes(estimated_memory)} memory"
        )
        print(val_losses)
        logger.info(
            "training_completed",
            extra=log_extra
            | {
                "train_seconds": round(train_time, 3),
                "final_val_loss": res.val_loss.item(),
                "estimated_memory_bytes": estimated_memory,
            },
        )
        wandb.log(
            {
                "train_seconds": train_time,
                "final_val_loss": res.val_loss.item(),
            }
        )
        wandb.finish()

        post_internal_finish(
            experiment_id,
            CompletedPayload(
                result_type="completed",
                used_runtime_seconds=train_time,
                val_losses=val_losses,
            ),
            log_extra=log_extra,
        )

        return TrainingResult(
            val_losses=val_losses,
            estimated_memory=estimated_memory,
            training_time=train_time,
            model_params=model_params,
        )

    val_event_executor = ThreadPoolExecutor(max_workers=1)
    try:
        return run()
    except TimeoutError:
        logger.warning("training_timeout_already_reported", extra=log_extra)
        raise
    except KeyboardInterrupt:
        elapsed_seconds = elapsed_training_seconds()
        preempted_log_extra = log_extra | {
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        logger.warning("preempted", extra=preempted_log_extra)
        try:
            post_internal_worker_event(
                experiment_id,
                PreemptedEvent(run_id=modal_function_call_id),
                log_extra=preempted_log_extra,
            )
        except Exception:
            logger.exception(
                "training_preempted_event_report_failed",
                extra=preempted_log_extra,
            )
        raise
    except Exception as exc:
        elapsed_seconds = elapsed_training_seconds()
        logger.exception(
            "training_unexpected_error",
            extra=log_extra
            | {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": round(elapsed_seconds, 3),
            },
        )
        try:
            post_internal_finish(
                experiment_id,
                FailedPayload(
                    result_type="failed",
                    used_runtime_seconds=elapsed_seconds,
                    reason=UnexpectedReason(failure=f"{type(exc).__name__}: {exc}"),
                ),
                log_extra=log_extra,
            )
        except Exception:
            logger.exception("training_unexpected_error_report_failed", extra=log_extra)
        raise
    finally:
        val_event_executor.shutdown(wait=True)


@app.local_entrypoint()
def modal_main():
    training_config = default_training_config()
    print(
        "final result",
        run_training.remote(
            training_config,
            sunet_id="local",
            experiment_id=1,
        ),
    )


if __name__ == "__main__":
    run_training.local(default_training_config(), sunet_id="local", experiment_id=2)

from typing import Annotated, Literal

from pydantic import Field

from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.schemas.experiment import FailReason


class QueueEvent(FrozenForbidExtraModel):
    log_type: Literal["queued"] = "queued"


class DispatchedEvent(FrozenForbidExtraModel):
    run_id: str
    log_type: Literal["dispatched"] = "dispatched"


class CompleteEvent(FrozenForbidExtraModel):
    log_type: Literal["completed"] = "completed"


class FailedEvent(FrozenForbidExtraModel):
    reason: FailReason
    log_type: Literal["failed"] = "failed"


class RunStartedEvent(FrozenForbidExtraModel):
    run_id: str
    wandb_path: str | None
    log_type: Literal["run_started"] = "run_started"


class ValidationLossEvent(FrozenForbidExtraModel):
    run_id: str
    val_losses: list[float]
    log_type: Literal["validation_loss"] = "validation_loss"


class RunHeartbeatEvent(FrozenForbidExtraModel):
    run_id: str
    elapsed_seconds: float
    log_type: Literal["run_heartbeat"] = "run_heartbeat"


class PreemptedEvent(FrozenForbidExtraModel):
    run_id: str
    log_type: Literal["preempted"] = "preempted"


type WorkerEvent = Annotated[
    RunStartedEvent | ValidationLossEvent | RunHeartbeatEvent | PreemptedEvent,
    Field(discriminator="log_type"),
]


type EventType = Annotated[
    QueueEvent
    | DispatchedEvent
    | CompleteEvent
    | FailedEvent
    | RunStartedEvent
    | ValidationLossEvent
    | RunHeartbeatEvent
    | PreemptedEvent,
    Field(discriminator="log_type"),
]

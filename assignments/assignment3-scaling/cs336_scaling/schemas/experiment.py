import datetime as dt
from typing import Annotated, Literal

from pydantic import Field

from cs336_scaling.schemas.base import FrozenForbidExtraModel


class QueuedExperimentStatus(FrozenForbidExtraModel):
    queued_at: dt.datetime
    status_type: Literal["queued"] = "queued"


class RunningExperimentStatus(FrozenForbidExtraModel):
    queued_at: dt.datetime
    dispatched_at: dt.datetime
    run_id: str
    val_losses: list[float] = Field(default_factory=list)
    status_type: Literal["running"] = "running"


class CompletedExperimentStatus(FrozenForbidExtraModel):
    queued_at: dt.datetime
    dispatched_at: dt.datetime
    run_id: str
    used_runtime_seconds: float
    val_losses: list[float]
    completed_at: dt.datetime
    status_type: Literal["completed"] = "completed"


class TimeoutReason(FrozenForbidExtraModel):
    partial_val_losses: list[float]
    reason: Literal["timeout"] = "timeout"


class UnexpectedReason(FrozenForbidExtraModel):
    failure: str
    reason: Literal["unexpected"] = "unexpected"


type FailReason = Annotated[
    TimeoutReason | UnexpectedReason,
    Field(discriminator="reason"),
]


class FailedExperimentStatus(FrozenForbidExtraModel):
    queued_at: dt.datetime
    dispatched_at: dt.datetime
    run_id: str
    used_runtime_seconds: float
    reason: FailReason
    failed_at: dt.datetime
    status_type: Literal["failed"] = "failed"


type ExperimentStatus = Annotated[
    QueuedExperimentStatus
    | RunningExperimentStatus
    | CompletedExperimentStatus
    | FailedExperimentStatus,
    Field(discriminator="status_type"),
]

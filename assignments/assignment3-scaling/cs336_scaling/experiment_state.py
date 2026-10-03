import datetime as dt

from cs336_scaling.db.tables import EventLogTable, ExperimentTable
from cs336_scaling.schemas.event import CompleteEvent, DispatchedEvent, FailedEvent
from cs336_scaling.schemas.experiment import (
    CompletedExperimentStatus,
    FailedExperimentStatus,
    FailReason,
    QueuedExperimentStatus,
    RunningExperimentStatus,
)


def mark_experiment_running(
    experiment: ExperimentTable,
    *,
    dispatched_at: dt.datetime,
    run_id: str,
) -> EventLogTable:
    match experiment.status:
        case QueuedExperimentStatus(queued_at=queued_at):
            experiment.status = RunningExperimentStatus(
                queued_at=queued_at,
                dispatched_at=dispatched_at,
                run_id=run_id,
                val_losses=[],
            )
        case _:
            raise ValueError("experiment must be queued before dispatch")

    return EventLogTable(
        experiment_id=experiment.id,
        event_type=DispatchedEvent(run_id=run_id),
        created_at=dispatched_at,
    )


def mark_experiment_completed(
    experiment: ExperimentTable,
    *,
    completed_at: dt.datetime,
    used_runtime_seconds: float,
    val_losses: list[float],
) -> EventLogTable:
    match experiment.status:
        case RunningExperimentStatus(
            queued_at=queued_at,
            dispatched_at=dispatched_at,
            run_id=run_id,
        ):
            experiment.status = CompletedExperimentStatus(
                queued_at=queued_at,
                dispatched_at=dispatched_at,
                run_id=run_id,
                used_runtime_seconds=used_runtime_seconds,
                val_losses=val_losses,
                completed_at=completed_at,
            )
        case _:
            raise ValueError("experiment must be running before completion")

    return EventLogTable(
        experiment_id=experiment.id,
        event_type=CompleteEvent(),
        created_at=completed_at,
    )


def mark_experiment_failed(
    experiment: ExperimentTable,
    *,
    failed_at: dt.datetime,
    used_runtime_seconds: float,
    reason: FailReason,
) -> EventLogTable:
    match experiment.status:
        case RunningExperimentStatus(
            queued_at=queued_at,
            dispatched_at=dispatched_at,
            run_id=run_id,
        ):
            experiment.status = FailedExperimentStatus(
                queued_at=queued_at,
                dispatched_at=dispatched_at,
                run_id=run_id,
                used_runtime_seconds=used_runtime_seconds,
                failed_at=failed_at,
                reason=reason,
            )
        case _:
            raise ValueError("experiment must be running before failure")

    return EventLogTable(
        experiment_id=experiment.id,
        event_type=FailedEvent(reason=reason),
        created_at=failed_at,
    )

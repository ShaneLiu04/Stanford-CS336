from typing import Annotated, Literal, assert_never

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status

from cs336_scaling.auth import require_internal_key
from cs336_scaling.config import utc_now
from cs336_scaling.db import SessionDep
from cs336_scaling.db.tables import EventLogTable, ExperimentTable
from cs336_scaling.experiment_state import (
    mark_experiment_completed,
    mark_experiment_failed,
)
from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.schemas.event import ValidationLossEvent, WorkerEvent
from cs336_scaling.schemas.experiment import (
    FailReason,
    RunningExperimentStatus,
)

router = APIRouter(
    prefix="/internal/worker",
    dependencies=[Depends(require_internal_key)],
    include_in_schema=False,
)


class CompletedPayload(FrozenForbidExtraModel):
    used_runtime_seconds: float
    val_losses: list[float]
    result_type: Literal["completed"]


class FailedPayload(FrozenForbidExtraModel):
    used_runtime_seconds: float
    reason: FailReason
    result_type: Literal["failed"]


type FinishPayload = CompletedPayload | FailedPayload


@router.post("/{experiment_id}/event", status_code=status.HTTP_204_NO_CONTENT)
def log_worker_event(
    experiment_id: Annotated[int, Path(gt=0)],
    request: Annotated[WorkerEvent, Body()],
    session: SessionDep,
) -> None:
    now = utc_now()
    with session.begin():
        experiment = session.get(
            ExperimentTable,
            experiment_id,
            with_for_update=True,
        )
        if experiment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="experiment not found",
            )
        match request:
            case ValidationLossEvent(run_id=run_id) if (
                isinstance(experiment.status, RunningExperimentStatus)
                and experiment.status.run_id == run_id
            ):
                experiment.status = experiment.status.model_copy(
                    update={"val_losses": request.val_losses}
                )
            case _:
                pass
        session.add(
            EventLogTable(
                experiment_id=experiment.id,
                event_type=request,
                created_at=now,
            )
        )


@router.post("/{experiment_id}/finish", status_code=status.HTTP_204_NO_CONTENT)
def finish_worker(
    experiment_id: Annotated[int, Path(gt=0)],
    request: Annotated[FinishPayload, Body()],
    session: SessionDep,
) -> None:
    now = utc_now()
    with session.begin():
        experiment = session.get(
            ExperimentTable,
            experiment_id,
            with_for_update=True,
        )
        if experiment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="experiment not found",
            )
        try:
            match request:
                case CompletedPayload():
                    session.add(
                        mark_experiment_completed(
                            experiment,
                            completed_at=now,
                            used_runtime_seconds=request.used_runtime_seconds,
                            val_losses=request.val_losses,
                        )
                    )
                case FailedPayload():
                    session.add(
                        mark_experiment_failed(
                            experiment,
                            failed_at=now,
                            used_runtime_seconds=request.used_runtime_seconds,
                            reason=request.reason,
                        )
                    )
                case _:
                    assert_never(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="you can only report a finish if currrently running",
            ) from exc

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from cs336_scaling.auth import CurrentUserDep
from cs336_scaling.budget import compute_user_budget_summary
from cs336_scaling.config import utc_now
from cs336_scaling.db import SessionDep
from cs336_scaling.db.tables import (
    EventLogTable,
    ExperimentTable,
    FinalSubmissionTable,
    UserTable,
)
from cs336_scaling.schemas import (
    BudgetSummary,
    ExperimentResponse,
    FinalSubmissionRequest,
    FinalSubmissionResponse,
    SubmitResponse,
)
from cs336_scaling.schemas.event import QueueEvent
from cs336_scaling.schemas.experiment import QueuedExperimentStatus
from cs336_scaling.training.training_config import TrainingConfig

router = APIRouter()


@router.post("/submit", response_model=SubmitResponse)
def submit_experiment(
    training_config: TrainingConfig,
    current_user: CurrentUserDep,
    session: SessionDep,
) -> SubmitResponse:
    now = utc_now()
    with session.begin():
        locked_user = session.scalar(
            select(UserTable)
            .where(UserTable.sunet_id == current_user.sunet_id)
            .with_for_update()
        )
        if locked_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
            )

        budget_summary = compute_user_budget_summary(
            session=session,
            user_sunet_id=current_user.sunet_id,
        )
        if training_config.max_runtime_seconds > budget_summary.remaining_seconds:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "insufficient budget",
                    "budget_summary": budget_summary.model_dump(mode="json"),
                },
            )

        training_config_unique_id = training_config.unique_id
        duplicate_experiment_id = session.scalar(
            select(ExperimentTable.id).where(
                ExperimentTable.user_sunet_id == current_user.sunet_id,
                ExperimentTable.training_config_unique_id == training_config_unique_id,
            )
        )
        if duplicate_experiment_id is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="experiment already exists for this training config",
            )

        experiment = ExperimentTable(
            user_sunet_id=current_user.sunet_id,
            training_config_unique_id=training_config_unique_id,
            training_config=training_config,
            status=QueuedExperimentStatus(queued_at=now),
        )
        session.add(experiment)
        session.flush()
        session.add(
            EventLogTable(
                experiment_id=experiment.id, event_type=QueueEvent(), created_at=now
            )
        )

    return SubmitResponse(
        experiment_id=experiment.id,
        budget_summary=budget_summary.with_reserved_runtime_seconds(
            training_config.max_runtime_seconds
        ),
    )


@router.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(
    current_user: CurrentUserDep,
    session: SessionDep,
) -> list[ExperimentResponse]:
    return [
        ExperimentResponse.from_experiment(experiment)
        for experiment in session.scalars(
            select(ExperimentTable)
            .where(ExperimentTable.user_sunet_id == current_user.sunet_id)
            .order_by(ExperimentTable.queued_at, ExperimentTable.id.desc())
        )
    ]


@router.get("/experiment/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: int,
    current_user: CurrentUserDep,
    session: SessionDep,
) -> ExperimentResponse:
    experiment = session.scalar(
        select(ExperimentTable).where(
            ExperimentTable.id == experiment_id,
            ExperimentTable.user_sunet_id == current_user.sunet_id,
        )
    )
    if experiment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return ExperimentResponse.from_experiment(experiment)


@router.get("/budget", response_model=BudgetSummary)
def get_budget(
    current_user: CurrentUserDep,
    session: SessionDep,
) -> BudgetSummary:
    return compute_user_budget_summary(
        session=session,
        user_sunet_id=current_user.sunet_id,
    )


@router.get("/final_submission", response_model=FinalSubmissionResponse | None)
def get_final_submission(
    current_user: CurrentUserDep,
    session: SessionDep,
) -> FinalSubmissionResponse | None:
    submission = session.scalar(
        select(FinalSubmissionTable).where(
            FinalSubmissionTable.user_sunet_id == current_user.sunet_id
        )
    )
    if submission is None:
        return None
    return FinalSubmissionResponse(
        training_config=submission.training_config,
        predicted_final_loss=submission.predicted_final_loss,
        submitted_at=submission.submitted_at,
    )


@router.post("/final_submission", response_model=FinalSubmissionResponse)
def save_final_submission(
    submission: FinalSubmissionRequest,
    current_user: CurrentUserDep,
    session: SessionDep,
) -> FinalSubmissionResponse:
    now = utc_now()
    with session.begin():
        session.execute(
            insert(FinalSubmissionTable)
            .values(
                user_sunet_id=current_user.sunet_id,
                training_config=submission.training_config,
                predicted_final_loss=submission.predicted_final_loss,
                submitted_at=now,
            )
            .on_conflict_do_update(
                index_elements=[FinalSubmissionTable.user_sunet_id],
                set_={
                    "training_config": submission.training_config,
                    "predicted_final_loss": submission.predicted_final_loss,
                    "submitted_at": now,
                },
            )
        )
    return FinalSubmissionResponse(
        training_config=submission.training_config,
        predicted_final_loss=submission.predicted_final_loss,
        submitted_at=now,
    )

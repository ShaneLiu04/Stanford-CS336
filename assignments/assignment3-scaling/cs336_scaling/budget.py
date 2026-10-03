from sqlalchemy import ColumnElement, Float, case, func, select
from sqlalchemy import cast as sa_cast
from sqlalchemy.orm import Session

from cs336_scaling.config import settings_from_env
from cs336_scaling.db.tables import ExperimentTable
from cs336_scaling.schemas import BudgetSummary

FINISHED_EXPERIMENT_STATUSES = ("completed", "failed")
ACTIVE_EXPERIMENT_STATUSES = ("queued", "running")


def _budget_seconds_expression() -> ColumnElement[float]:
    status_type = ExperimentTable.status["status_type"].astext
    max_runtime_seconds = sa_cast(
        ExperimentTable.training_config["max_runtime_seconds"].astext,
        Float,
    )
    used_runtime_seconds = sa_cast(
        ExperimentTable.status["used_runtime_seconds"].astext,
        Float,
    )
    return case(
        (
            status_type.in_(FINISHED_EXPERIMENT_STATUSES),
            func.least(
                func.greatest(1.0, used_runtime_seconds),
                max_runtime_seconds,
            ),
        ),
        (
            status_type.in_(ACTIVE_EXPERIMENT_STATUSES),
            max_runtime_seconds,
        ),
        else_=0.0,
    )


def compute_user_budget_summary(
    session: Session,
    user_sunet_id: str,
) -> BudgetSummary:
    total_used = session.scalar(
        select(func.coalesce(func.sum(_budget_seconds_expression()), 0.0)).where(
            ExperimentTable.user_sunet_id == user_sunet_id
        )
    )
    if total_used is None:
        raise RuntimeError("Expected budget usage query to return a numeric total")
    total_budget_seconds = settings_from_env().total_budget_seconds
    return BudgetSummary(
        used_seconds=total_used,
        remaining_seconds=total_budget_seconds - total_used,
        total_budget_seconds=total_budget_seconds,
    )

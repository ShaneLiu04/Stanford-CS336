from dataclasses import dataclass

from sqlalchemy import DateTime, Integer, cast, func, literal, select
from sqlalchemy.orm import Session

from cs336_scaling.db.tables import ExperimentTable


@dataclass(frozen=True)
class DispatchSelection:
    experiments: list[ExperimentTable]
    currently_running_jobs: int
    zero_running_user_experiments: int


def select_experiments_for_dispatch(
    session: Session,
    *,
    max_concurrent_workers: int,
) -> DispatchSelection:
    status_type = ExperimentTable.status["status_type"].astext
    queued_at = cast(
        ExperimentTable.status["queued_at"].astext, DateTime(timezone=True)
    )

    running_by_user = (
        select(
            ExperimentTable.user_sunet_id,
            cast(func.count(), Integer).label("running_count"),
        )
        .where(status_type == "running")
        .group_by(ExperimentTable.user_sunet_id)
        .cte("running_by_user")
    )
    capacity = (
        select(
            func.greatest(
                literal(max_concurrent_workers) - cast(func.count(), Integer),
                0,
            ).label("available"),
            cast(func.count(), Integer).label("currently_running_jobs"),
        )
        .where(status_type == "running")
        .cte("capacity")
    )
    ranked_queued = (
        select(
            ExperimentTable.id,
            func.coalesce(running_by_user.c.running_count, 0).label(
                "initial_running_count"
            ),
            (
                func.coalesce(running_by_user.c.running_count, 0)
                + func.row_number()
                .over(
                    partition_by=ExperimentTable.user_sunet_id,
                    order_by=(queued_at, ExperimentTable.id),
                )
                .self_group()
                - 1
            ).label("effective_running_count"),
            queued_at.label("queued_at"),
        )
        .outerjoin(
            running_by_user,
            running_by_user.c.user_sunet_id == ExperimentTable.user_sunet_id,
        )
        .where(status_type == "queued")
        .cte("ranked_queued")
    )
    selected = (
        select(
            ranked_queued.c.id,
            ranked_queued.c.initial_running_count,
            ranked_queued.c.effective_running_count,
            ranked_queued.c.queued_at,
        )
        .order_by(
            ranked_queued.c.effective_running_count,
            ranked_queued.c.queued_at,
            ranked_queued.c.id,
        )
        .limit(select(capacity.c.available).scalar_subquery())
        .cte("selected")
    )

    rows = session.execute(
        select(
            ExperimentTable,
            selected.c.effective_running_count,
            capacity.c.currently_running_jobs,
        )
        .join(selected, selected.c.id == ExperimentTable.id)
        .join(capacity, literal(True))
        .order_by(
            selected.c.effective_running_count,
            selected.c.queued_at,
            ExperimentTable.id,
        )
    ).all()

    return DispatchSelection(
        experiments=[row.ExperimentTable for row in rows],
        currently_running_jobs=(
            rows[0].currently_running_jobs
            if rows
            else session.scalar(select(capacity.c.currently_running_jobs)) or 0
        ),
        zero_running_user_experiments=sum(
            1 for row in rows if row.effective_running_count == 0
        ),
    )

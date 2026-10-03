import datetime as dt

from cs336_scaling.db.tables import ExperimentTable, UserTable
from cs336_scaling.scheduler.experiment_selector import select_experiments_for_dispatch
from cs336_scaling.schemas.experiment import (
    QueuedExperimentStatus,
    RunningExperimentStatus,
)
from cs336_scaling.training.run import default_training_config


def test_select_experiments_for_dispatch_orders_by_running_count_then_queue_time(
    db_session_factory,
):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    model_seed = 0

    def get_unique_model_seed() -> int:
        nonlocal model_seed
        model_seed += 1
        return model_seed

    with db_session_factory.begin() as session:
        session.add_all(
            [
                UserTable(sunet_id="user1", api_key="user1-key"),
                UserTable(sunet_id="user2", api_key="user2-key"),
                UserTable(sunet_id="user3", api_key="user3-key"),
            ]
        )
        session.flush()

        for user_sunet_id in ("user2", "user3"):
            for running_index in range(2):
                session.add(
                    _experiment(
                        user_sunet_id,
                        RunningExperimentStatus(
                            queued_at=now,
                            dispatched_at=now,
                            run_id=f"{user_sunet_id}-{running_index}",
                        ),
                        model_seed=get_unique_model_seed(),
                    )
                )

        for _ in range(10):
            session.add(
                _experiment(
                    "user1",
                    QueuedExperimentStatus(queued_at=now + dt.timedelta(minutes=10)),
                    model_seed=get_unique_model_seed(),
                )
            )
            session.add(
                _experiment(
                    "user2",
                    QueuedExperimentStatus(queued_at=now + dt.timedelta(minutes=19)),
                    model_seed=get_unique_model_seed(),
                )
            )
            session.add(
                _experiment(
                    "user3",
                    QueuedExperimentStatus(queued_at=now + dt.timedelta(minutes=9)),
                    model_seed=get_unique_model_seed(),
                )
            )

    with db_session_factory() as session:
        selection = select_experiments_for_dispatch(
            session,
            max_concurrent_workers=9,
        )

    assert selection.currently_running_jobs == 4
    assert selection.zero_running_user_experiments == 1
    assert [
        (experiment.training_config.model_seed, experiment.user_sunet_id)
        for experiment in selection.experiments
    ] == [
        (5, "user1"),
        (8, "user1"),
        (7, "user3"),
        (11, "user1"),
        (6, "user2"),
    ]


def test_select_experiments_for_dispatch_counts_running_jobs_without_capacity(
    db_session_factory,
):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    with db_session_factory.begin() as session:
        session.add(UserTable(sunet_id="user1", api_key="user1-key"))
        session.flush()
        session.add(
            _experiment(
                "user1",
                RunningExperimentStatus(
                    queued_at=now,
                    dispatched_at=now,
                    run_id="running",
                ),
                model_seed=1,
            )
        )
        session.add(
            _experiment(
                "user1",
                QueuedExperimentStatus(queued_at=now + dt.timedelta(minutes=1)),
                model_seed=2,
            )
        )

    with db_session_factory() as session:
        selection = select_experiments_for_dispatch(
            session,
            max_concurrent_workers=1,
        )

    assert selection.experiments == []
    assert selection.currently_running_jobs == 1
    assert selection.zero_running_user_experiments == 0


def _experiment(
    user_sunet_id: str,
    status: QueuedExperimentStatus | RunningExperimentStatus,
    model_seed: int,
) -> ExperimentTable:
    training_config = default_training_config().model_copy(
        update={"model_seed": model_seed}
    )
    return ExperimentTable(
        user_sunet_id=user_sunet_id,
        training_config_unique_id=training_config.unique_id,
        training_config=training_config,
        status=status,
    )

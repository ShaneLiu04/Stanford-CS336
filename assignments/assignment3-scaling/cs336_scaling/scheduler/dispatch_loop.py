import logging
import time

import modal

from cs336_scaling.config import settings_from_env, utc_now
from cs336_scaling.db import get_session_factory
from cs336_scaling.db.tables import ExperimentTable
from cs336_scaling.experiment_state import mark_experiment_running
from cs336_scaling.log.setup import configure_logging
from cs336_scaling.scheduler.experiment_selector import select_experiments_for_dispatch

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(log_file="logs/dispatch_loop/dispatch.log")
    settings = settings_from_env()
    session_factory = get_session_factory()
    failed_dispatches: set[tuple[int, str]] = set()
    logger.info(
        "dispatcher_started",
        extra={
            "max_concurrent_workers": settings.max_concurrent_workers,
            "dispatch_interval_seconds": settings.dispatch_interval_seconds,
            "launch_mode": settings.launch_mode,
        },
    )
    last_debug_logged_at = 0.0
    while True:
        selection_started_at = time.perf_counter()
        with session_factory() as session:
            dispatch_selection = select_experiments_for_dispatch(
                session, max_concurrent_workers=settings.max_concurrent_workers
            )
        selection_duration_seconds = time.perf_counter() - selection_started_at
        now = time.perf_counter()
        if now - last_debug_logged_at >= 60:
            last_debug_logged_at = now
            logger.debug(
                "dispatch_selection_completed",
                extra={
                    "selected_experiments": len(dispatch_selection.experiments),
                    "currently_running_jobs": dispatch_selection.currently_running_jobs,
                    "zero_running_user_experiments": (
                        dispatch_selection.zero_running_user_experiments
                    ),
                    "selection_duration_seconds": selection_duration_seconds,
                },
            )
        for experiment in dispatch_selection.experiments:
            experiment_key = (experiment.id, experiment.user_sunet_id)
            if experiment_key in failed_dispatches:
                logger.debug(
                    "skipping_previously_failed_dispatch",
                    extra=_experiment_log_extra(experiment),
                )
                continue
            logger.info(
                "dispatching_experiment",
                extra=_experiment_log_extra(experiment),
            )
            run_training = modal.Function.from_name("scaling-brunborg", "run_training")
            run_id = run_training.spawn(
                training_config=experiment.training_config,
                sunet_id=experiment.user_sunet_id,
                experiment_id=experiment.id,
            )
            with session_factory() as session, session.begin():
                db_experiment = session.get(
                    ExperimentTable,
                    experiment.id,
                    with_for_update=True,
                )
                if db_experiment is None:
                    failed_dispatches.add(experiment_key)
                    logger.warning(
                        "experiment_missing_after_modal_spawn",
                        extra=_experiment_log_extra(experiment)
                        | {"run_id": run_id.object_id},
                    )
                    continue
                session.add(
                    mark_experiment_running(
                        db_experiment,
                        dispatched_at=utc_now(),
                        run_id=run_id.object_id,
                    )
                )
                logger.info(
                    "experiment_dispatched",
                    extra=_experiment_log_extra(db_experiment)
                    | {"run_id": run_id.object_id},
                )
        time.sleep(1)


def _experiment_log_extra(experiment: ExperimentTable) -> dict[str, object]:
    return {
        "experiment_id": experiment.id,
        "user_sunet_id": experiment.user_sunet_id,
        "training_config_unique_id": experiment.training_config_unique_id,
    }


if __name__ == "__main__":
    main()

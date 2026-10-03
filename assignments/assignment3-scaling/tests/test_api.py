import datetime as dt

from sqlalchemy import select

from cs336_scaling.config import settings_from_env
from cs336_scaling.db.tables import ExperimentTable, UserTable
from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig


def training_config(
    *,
    hidden_size: int = 128,
    max_runtime_seconds: float = 10,
) -> TrainingConfig:
    return TrainingConfig(
        architecture_config=BasicTransformerConfig(
            attention_bias=False,
            head_dim=2,
            hidden_size=hidden_size * 2,
            intermediate_size=hidden_size * 4,
            num_attention_heads=hidden_size,
            num_hidden_layers=1,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            rope_theta=10_000,
            tie_word_embeddings=False,
            dtype="float32",
            # use_sliding_window=False,
            vocab_size=1_000,
        ),
        optimizer_config=AdamWConfig(),
        train_batch_size=2,
        val_batch_size=1,
        n_evals=2,
        total_train_tokens=2048,
        max_runtime_seconds=max_runtime_seconds,
    )


def test_dashboard_serves_html(client):
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Scaling Experiments Dashboard" in response.text


def test_budget(client, db_session_factory):
    with db_session_factory.begin() as session:
        user = UserTable(sunet_id="alice", api_key="test-api-key")
        session.add(user)

    response = client.get("/budget", headers={"X-API-Key": user.api_key})

    assert response.status_code == 200
    with db_session_factory() as session:
        assert (
            session.scalar(select(UserTable).where(UserTable.sunet_id == "alice"))
            is not None
        )

    assert response.json() == {
        "used_seconds": 0.0,
        "remaining_seconds": settings_from_env().total_budget_seconds,
        "total_budget_seconds": settings_from_env().total_budget_seconds,
    }


def test_submit_jobs(client, db_session_factory):
    with db_session_factory.begin() as session:
        user = UserTable(sunet_id="alice", api_key="test-api-key")
        session.add(user)

    r_budget_before = client.get("/budget", headers={"X-API-Key": user.api_key})
    assert r_budget_before.status_code == 200
    assert r_budget_before.json() == {
        "used_seconds": 0.0,
        "remaining_seconds": settings_from_env().total_budget_seconds,
        "total_budget_seconds": settings_from_env().total_budget_seconds,
    }

    total_budget_seconds = settings_from_env().total_budget_seconds
    used_seconds = 0

    max_runtime_seconds_lst = [10, 30, 20]
    for experiment_id, max_runtime_seconds in enumerate(
        max_runtime_seconds_lst, start=1
    ):
        used_seconds += max_runtime_seconds

        r_submit = client.post(
            "/submit",
            json=training_config(
                hidden_size=100 + experiment_id,
                max_runtime_seconds=max_runtime_seconds,
            ).model_dump(),
            headers={"X-API-Key": user.api_key},
        )
        assert r_submit.status_code == 200
        assert r_submit.json() == {
            "experiment_id": experiment_id,
            "budget_summary": {
                "used_seconds": used_seconds,
                "remaining_seconds": total_budget_seconds - used_seconds,
                "total_budget_seconds": total_budget_seconds,
            },
        }

        r_budget_after = client.get("/budget", headers={"X-API-Key": user.api_key})
        assert r_budget_after.status_code == 200
        assert r_budget_after.json() == {
            "used_seconds": used_seconds,
            "remaining_seconds": total_budget_seconds - used_seconds,
            "total_budget_seconds": total_budget_seconds,
        }

        assert r_submit.json()["budget_summary"] == r_budget_after.json()

    r_experiments = client.get(
        "/experiments", headers={"X-API-key": user.api_key}
    ).json()
    assert [x["experiment_id"] for x in r_experiments] == [1, 2, 3]
    assert [
        x["training_config"]["max_runtime_seconds"] for x in r_experiments
    ] == max_runtime_seconds_lst
    assert [
        x["training_config"]["architecture_config"]["hidden_size"]
        for x in r_experiments
    ] == [202, 204, 206]
    assert [x["status"]["status_type"] for x in r_experiments] == ["queued"] * 3
    assert [x["status"]["queued_at"] for x in r_experiments] == sorted(
        [x["status"]["queued_at"] for x in r_experiments]
    )

    assert ExperimentResponse.model_validate(
        r_experiments[0]
    ).status.queued_at < dt.datetime.now(dt.timezone.utc)

    assert ExperimentResponse.model_validate(
        r_experiments[0]
    ).status.queued_at > dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)

    assert (
        client.get("/experiment/1", headers={"X-API-Key": user.api_key}).json()
        == r_experiments[0]
    )


def test_submit_rejects_duplicate_training_config(client, db_session_factory):
    with db_session_factory.begin() as session:
        user = UserTable(sunet_id="alice", api_key="test-api-key")
        session.add(user)

    config = training_config(hidden_size=256, max_runtime_seconds=15)

    r_submit = client.post(
        "/submit",
        json=config.model_dump(),
        headers={"X-API-Key": user.api_key},
    )
    assert r_submit.status_code == 200
    assert r_submit.json()["experiment_id"] == 1

    r_duplicate = client.post(
        "/submit",
        json=config.model_dump(),
        headers={"X-API-Key": user.api_key},
    )

    assert r_duplicate.status_code == 409
    assert r_duplicate.json() == {
        "detail": "experiment already exists for this training config"
    }

    r_budget = client.get("/budget", headers={"X-API-Key": user.api_key})
    assert r_budget.status_code == 200
    assert r_budget.json() == {
        "used_seconds": config.max_runtime_seconds,
        "remaining_seconds": settings_from_env().total_budget_seconds
        - config.max_runtime_seconds,
        "total_budget_seconds": settings_from_env().total_budget_seconds,
    }

    with db_session_factory() as session:
        experiment = session.scalar(select(ExperimentTable))
        assert experiment is not None
        assert experiment.training_config_unique_id == config.unique_id


def test_final_submission_accepts_training_config_and_predicted_loss(
    client, db_session_factory
):
    with db_session_factory.begin() as session:
        user = UserTable(sunet_id="alice", api_key="test-api-key")
        session.add(user)

    config = training_config(hidden_size=256, max_runtime_seconds=15)
    predicted_final_loss = 2.75
    r_submit = client.post(
        "/final_submission",
        json={
            "training_config": config.model_dump(mode="json"),
            "predicted_final_loss": predicted_final_loss,
        },
        headers={"X-API-Key": user.api_key},
    )

    assert r_submit.status_code == 200
    submitted = r_submit.json()
    assert submitted["training_config"] == config.model_dump(mode="json")
    assert submitted["predicted_final_loss"] == predicted_final_loss
    assert dt.datetime.fromisoformat(submitted["submitted_at"]) < dt.datetime.now(
        dt.timezone.utc
    )

    r_get = client.get("/final_submission", headers={"X-API-Key": user.api_key})

    assert r_get.status_code == 200
    fetched = r_get.json()
    assert fetched["training_config"] == submitted["training_config"]
    assert fetched["predicted_final_loss"] == predicted_final_loss
    assert dt.datetime.fromisoformat(
        fetched["submitted_at"]
    ) == dt.datetime.fromisoformat(submitted["submitted_at"])

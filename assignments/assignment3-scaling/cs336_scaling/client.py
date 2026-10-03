import os

import requests
from pydantic import JsonValue, TypeAdapter

from cs336_scaling.schemas import (
    BudgetSummary,
    ExperimentResponse,
    FinalSubmissionRequest,
    FinalSubmissionResponse,
    SubmitResponse,
)
from cs336_scaling.training.training_config import TrainingConfig

API_BASE_URL = "http://hyperturing.stanford.edu:8000"
API_KEY = os.getenv("A3_API_KEY", "")


def _api_headers() -> dict[str, str]:
    if len(API_KEY) != 8 or not API_KEY.isdigit():
        raise ValueError(
            f"make sure your student id is 8 characters, e.g., 06123456. current: {API_KEY=}"
        )
    return {"X-API-Key": API_KEY}


def _request_json[T](response: requests.Response, response_type: type[T]) -> T:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{exc}\n{response.text}") from exc
    return TypeAdapter(response_type).validate_python(
        None if not response.content else response.json()
    )


def get_budget() -> BudgetSummary:
    response = requests.get(
        f"{API_BASE_URL}/budget",
        headers=_api_headers(),
    )
    return _request_json(response, BudgetSummary)


def submit_experiment(training_config: TrainingConfig | JsonValue) -> SubmitResponse:
    response = requests.post(
        f"{API_BASE_URL}/submit",
        headers=_api_headers(),
        json=TrainingConfig.model_validate(training_config).model_dump(mode="json"),
    )
    return _request_json(response, SubmitResponse)


def list_experiments() -> list[ExperimentResponse]:
    response = requests.get(
        f"{API_BASE_URL}/experiments",
        headers=_api_headers(),
    )
    return _request_json(response, list[ExperimentResponse])


def get_experiment(experiment_id: int) -> ExperimentResponse:
    response = requests.get(
        f"{API_BASE_URL}/experiment/{experiment_id}",
        headers=_api_headers(),
    )
    return _request_json(response, ExperimentResponse)


def save_final_submission(
    training_config: TrainingConfig | JsonValue,
    predicted_final_loss: float,
) -> FinalSubmissionResponse:
    submission = FinalSubmissionRequest(
        training_config=TrainingConfig.model_validate(training_config).model_dump(
            mode="json"
        ),
        predicted_final_loss=predicted_final_loss,
    )
    response = requests.post(
        f"{API_BASE_URL}/final_submission",
        headers=_api_headers(),
        json=submission.model_dump(mode="json"),
    )
    return _request_json(response, FinalSubmissionResponse)


def save_final_submission_json(
    submission: FinalSubmissionRequest | JsonValue,
) -> FinalSubmissionResponse:
    response = requests.post(
        f"{API_BASE_URL}/final_submission",
        headers=_api_headers(),
        json=FinalSubmissionRequest.model_validate(submission).model_dump(mode="json"),
    )
    return _request_json(response, FinalSubmissionResponse)


def get_final_submission() -> FinalSubmissionResponse | None:
    response = requests.get(
        f"{API_BASE_URL}/final_submission",
        headers=_api_headers(),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{exc}\n{response.text}") from exc
    return TypeAdapter(FinalSubmissionResponse | None).validate_python(
        None if not response.content else response.json()
    )

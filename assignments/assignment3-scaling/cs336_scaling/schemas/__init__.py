import datetime as dt
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.schemas.experiment import ExperimentStatus
from cs336_scaling.training.training_config import TrainingConfig

if TYPE_CHECKING:
    from cs336_scaling.db.tables import ExperimentTable


class BudgetSummary(FrozenForbidExtraModel):
    used_seconds: float
    remaining_seconds: float
    total_budget_seconds: float

    def with_reserved_runtime_seconds(
        self, max_runtime_seconds: float
    ) -> "BudgetSummary":
        return BudgetSummary(
            used_seconds=self.used_seconds + max_runtime_seconds,
            remaining_seconds=self.remaining_seconds - max_runtime_seconds,
            total_budget_seconds=self.total_budget_seconds,
        )


class SubmitResponse(FrozenForbidExtraModel):
    experiment_id: int
    budget_summary: BudgetSummary


class ExperimentResponse(FrozenForbidExtraModel):
    experiment_id: int
    training_config: TrainingConfig
    status: ExperimentStatus

    @classmethod
    def from_experiment(cls, experiment: "ExperimentTable") -> "ExperimentResponse":
        return cls(
            experiment_id=experiment.id,
            status=experiment.status,
            training_config=experiment.training_config,
        )


class FinalSubmissionRequest(FrozenForbidExtraModel):
    training_config: TrainingConfig
    predicted_final_loss: float = Field(gt=0.0)


class FinalSubmissionResponse(FrozenForbidExtraModel):
    training_config: TrainingConfig
    predicted_final_loss: float
    submitted_at: dt.datetime


class FinishExperimentRequest(FrozenForbidExtraModel):
    used_runtime_seconds: float = Field(ge=0.0)
    experiment_status: ExperimentStatus
    status: Literal["completed"] = "completed"

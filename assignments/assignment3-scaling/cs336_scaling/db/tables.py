import datetime as dt

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column

from cs336_scaling.db import Base
from cs336_scaling.db.sql_pydantic_types import PydanticJSON
from cs336_scaling.schemas import ExperimentStatus
from cs336_scaling.schemas.event import EventType
from cs336_scaling.training.training_config import TrainingConfig


class UserTable(Base):
    __tablename__ = "users"

    sunet_id: Mapped[str] = mapped_column(Text, primary_key=True)
    api_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class ExperimentTable(Base):
    __tablename__ = "experiments"

    id:                        Mapped[int]              = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_sunet_id:             Mapped[str]              = mapped_column(ForeignKey(UserTable.sunet_id), nullable=False)

    training_config_unique_id: Mapped[str]              = mapped_column(Text, nullable=False)
    training_config:           Mapped[TrainingConfig]   = mapped_column(PydanticJSON(TrainingConfig),   nullable=False)
    status:                    Mapped[ExperimentStatus] = mapped_column(PydanticJSON(ExperimentStatus), nullable=False)

    @hybrid_property
    def queued_at(self) -> dt.datetime:
        return self.status.queued_at

    @queued_at.inplace.expression
    @classmethod
    def _queued_at_expression(cls):
        return cls.status["queued_at"].astext

    __table_args__ = (
        UniqueConstraint(
            "user_sunet_id",
            "training_config_unique_id",
            name="uq_experiments_user_training_config_unique_id",
        ),
        Index("ix_experiments_status_queued_at", status, status["queued_at"].astext),
        Index(
            "ix_experiments_user_training_config",
            user_sunet_id,
            training_config,
        ),
        Index("ix_experiments_user_status", user_sunet_id, status),
    )


class EventLogTable(Base):
    __tablename__ = "experiment_logs"

    id:            Mapped[int]         = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    experiment_id: Mapped[int]         = mapped_column(BigInteger, ForeignKey(ExperimentTable.id), nullable=False)

    event_type:    Mapped[EventType]   = mapped_column(PydanticJSON(EventType), nullable=False)
    created_at:    Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FinalSubmissionTable(Base):
    __tablename__ = "final_submissions"

    user_sunet_id:        Mapped[str]            = mapped_column(ForeignKey(UserTable.sunet_id), primary_key=True)

    training_config:      Mapped[TrainingConfig] = mapped_column(PydanticJSON(TrainingConfig), nullable=False)
    predicted_final_loss: Mapped[float]          = mapped_column(Float, nullable=False)
    submitted_at:         Mapped[dt.datetime]    = mapped_column(DateTime(timezone=True), nullable=False)

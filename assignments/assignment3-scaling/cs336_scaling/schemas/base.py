from pydantic import BaseModel, ConfigDict


class FrozenForbidExtraModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

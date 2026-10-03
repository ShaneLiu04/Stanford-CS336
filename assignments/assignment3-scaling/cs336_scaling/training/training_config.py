from functools import cached_property
from typing import ClassVar, Self, assert_never

from pydantic import Field, model_validator

from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.stable_hash import stable_json_hash
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, OptimizerConfig, SGDConfig
from cs336_scaling.utils import require_int


class TrainingConfig(FrozenForbidExtraModel):
    architecture_config: BasicTransformerConfig
    optimizer_config: OptimizerConfig
    seq_len: ClassVar[int] = 512
    train_batch_size: int = Field(gt=0)
    val_batch_size: int = Field(gt=0)
    n_val_tokens: ClassVar[int] = 2**18
    n_evals: int = Field(default=16, gt=0)
    total_train_tokens: int = Field(gt=0, le=500_000_000_000)
    max_runtime_seconds: float = Field(ge=1)
    model_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_training_config(self) -> Self:
        if self.total_train_tokens % self.tokens_per_optimizer_step != 0:
            raise ValueError(
                "total_train_tokens must be divisible by seq_len * train_batch_size"
            )

        if self.total_optimizer_steps % self.n_evals != 0:
            raise ValueError("total optimizer steps must be divisible by n_evals")

        if self.n_val_tokens % (self.seq_len * self.val_batch_size) != 0:
            raise ValueError(
                "n_val_tokens must be divisible by seq_len * val_batch_size"
            )

        positive_int_fields = (
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
        )
        for field_name in positive_int_fields:
            if getattr(self.architecture_config, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

        if (
            self.architecture_config.hidden_size
            != self.architecture_config.num_attention_heads
            * self.architecture_config.head_dim
        ):
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

        if (
            self.architecture_config.num_attention_heads
            % self.architecture_config.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if (
            self.architecture_config.num_key_value_heads
            != self.architecture_config.num_attention_heads
        ):
            raise ValueError(
                "num_key_value_heads must equal num_attention_heads; GQA is not supported"
            )

        if self.architecture_config.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if (
            self.architecture_config.intermediate_size
            < self.architecture_config.hidden_size
        ):
            raise ValueError("intermediate_size should usually >= hidden_size")

        if self.architecture_config.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

        if self.architecture_config.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        if not 0 <= self.optimizer_config.lr_scheduler.warmup_frac < 1:
            raise ValueError("warmup_frac must be in [0, 1)")

        if not 0 <= self.optimizer_config.lr_scheduler.final_lr_frac <= 1:
            raise ValueError("final_lr_frac must be in [0, 1]")

        if self.optimizer_config.lr_scheduler.init_value < 0:
            raise ValueError("initial learning rate must be nonnegative")

        if self.optimizer_config.lr_scheduler.peak_value <= 0:
            raise ValueError("peak learning rate must be positive")

        if (
            self.optimizer_config.grad_clip_norm is not None
            and self.optimizer_config.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive or None")

        match self.optimizer_config:
            case AdamWConfig() as config:
                if config.weight_decay < 0:
                    raise ValueError("weight_decay must be nonnegative")
                if not 0 <= config.beta1 < 1:
                    raise ValueError("beta1 must be in [0, 1)")
                if not 0 <= config.beta2 < 1:
                    raise ValueError("beta2 must be in [0, 1)")
                if config.eps <= 0:
                    raise ValueError("eps must be positive")
                if config.eps_root < 0:
                    raise ValueError("eps_root must be nonnegative")
            case SGDConfig():
                pass
            case _:
                assert_never(self.optimizer_config)

        return self

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.seq_len * self.train_batch_size

    @property
    def eval_every_tokens(self) -> int:
        return require_int(self.total_train_tokens / self.n_evals)

    @property
    def optimizer_steps_per_eval(self) -> int:
        return require_int(self.eval_every_tokens / self.tokens_per_optimizer_step)

    @property
    def total_optimizer_steps(self) -> int:
        return require_int(self.total_train_tokens / self.tokens_per_optimizer_step)

    @cached_property
    def unique_id(self) -> str:
        return stable_json_hash(self)

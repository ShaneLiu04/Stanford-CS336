from dataclasses import field
from typing import TYPE_CHECKING

from cs336_scaling.schemas.base import FrozenForbidExtraModel

if TYPE_CHECKING:
    import optax

    from cs336_scaling.training.training_config import TrainingConfig


class WarmupCosineDecay(FrozenForbidExtraModel):
    peak_value: float = 3e-4
    final_lr_frac: float = 0.1
    warmup_frac: float = 0.05
    init_value: float = 0.0

    def build(self, n_steps: int) -> "optax.Schedule":
        import optax

        return optax.warmup_cosine_decay_schedule(
            init_value=self.init_value,
            peak_value=self.peak_value,
            warmup_steps=int(n_steps * self.warmup_frac),
            decay_steps=n_steps,
            end_value=self.peak_value * self.final_lr_frac,
        )


def _grad_clip_transforms(
    grad_clip_norm: float | None,
):
    import optax

    if grad_clip_norm is None:
        return []
    return [optax.clip_by_global_norm(grad_clip_norm)]


class SGDConfig(FrozenForbidExtraModel):
    lr_scheduler: WarmupCosineDecay
    grad_clip_norm: float | None = 1.0

    def build(
        self, training_config: "TrainingConfig"
    ) -> "optax.GradientTransformation":
        import optax

        return optax.chain(
            *_grad_clip_transforms(self.grad_clip_norm),
            optax.sgd(self.lr_scheduler.build(training_config.total_optimizer_steps)),
        )


class AdamWConfig(FrozenForbidExtraModel):
    lr_scheduler: WarmupCosineDecay = field(default_factory=WarmupCosineDecay)
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    eps_root: float = 1e-8
    grad_clip_norm: float | None = 1.0

    def build(
        self, training_config: "TrainingConfig"
    ) -> "optax.GradientTransformation":
        import optax

        return optax.chain(
            *_grad_clip_transforms(self.grad_clip_norm),
            optax.adamw(
                learning_rate=self.lr_scheduler.build(
                    training_config.total_optimizer_steps
                ),
                weight_decay=self.weight_decay,
                b1=self.beta1,
                b2=self.beta2,
                eps=self.eps,
                eps_root=self.eps_root,
            ),
        )


type OptimizerConfig = SGDConfig | AdamWConfig

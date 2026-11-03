"""Optimizer configs as tagged unions -- a `match` instead of a `name: str`.

Adding an optimizer means adding a class to the union; the type checker then
flags every `match` that does not handle it.
"""

import attrs
import optax
from attrs import validators as v


@attrs.frozen(kw_only=True)
class Constant:
    value: float = attrs.field(validator=v.gt(0.0))


def _after_warmup(
    instance: "WarmupCosine", attribute: "attrs.Attribute[int]", value: int
) -> None:
    if value <= instance.warmup_steps:
        raise ValueError(
            f"decay_steps={value} must exceed warmup_steps={instance.warmup_steps}: "
            "it counts from step 0, warmup included"
        )


@attrs.frozen(kw_only=True)
class WarmupCosine:
    """Linear warmup from 0 to `peak`, cosine decay to `peak * final_fraction`."""

    peak: float = attrs.field(validator=v.gt(0.0))
    warmup_steps: int = attrs.field(validator=v.ge(0))
    # Total schedule length, warmup included.
    decay_steps: int = attrs.field(validator=[v.gt(0), _after_warmup])
    final_fraction: float = attrs.field(default=0.0, validator=[v.ge(0.0), v.le(1.0)])


@attrs.frozen(kw_only=True)
class LinearDecay:
    """Linear decay from `start` to 0, as used by PPO's learning-rate annealing."""

    start: float = attrs.field(validator=v.gt(0.0))
    steps: int = attrs.field(validator=v.gt(0))


type Schedule = Constant | WarmupCosine | LinearDecay


def make_schedule(schedule: Schedule) -> optax.Schedule:
    match schedule:
        case Constant(value=value):
            return optax.constant_schedule(value)
        case WarmupCosine():
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=schedule.peak,
                warmup_steps=schedule.warmup_steps,
                decay_steps=schedule.decay_steps,
                end_value=schedule.peak * schedule.final_fraction,
            )
        case LinearDecay(start=start, steps=steps):
            return optax.linear_schedule(start, 0.0, steps)


@attrs.frozen(kw_only=True)
class AdamW:
    lr: Schedule
    b1: float = attrs.field(default=0.9, validator=[v.ge(0.0), v.lt(1.0)])
    b2: float = attrs.field(default=0.999, validator=[v.ge(0.0), v.lt(1.0)])
    eps: float = attrs.field(default=1e-8, validator=v.gt(0.0))
    weight_decay: float = attrs.field(default=0.0, validator=v.ge(0.0))


@attrs.frozen(kw_only=True)
class Sgd:
    lr: Schedule
    momentum: float | None = attrs.field(
        default=None, validator=v.optional([v.ge(0.0), v.lt(1.0)])
    )


type Optimizer = AdamW | Sgd


@attrs.frozen(kw_only=True)
class OptimConfig:
    optimizer: Optimizer
    clip_global_norm: float | None = attrs.field(
        default=None, validator=v.optional(v.gt(0.0))
    )

    def schedule(self) -> optax.Schedule:
        """The learning rate as a function of the optimizer's update count."""
        return make_schedule(self.optimizer.lr)

    def make(self) -> optax.GradientTransformation:
        match self.optimizer:
            case AdamW():
                opt = self.optimizer
                tx = optax.adamw(
                    learning_rate=make_schedule(opt.lr),
                    b1=opt.b1,
                    b2=opt.b2,
                    eps=opt.eps,
                    weight_decay=opt.weight_decay,
                )
            case Sgd():
                tx = optax.sgd(
                    learning_rate=make_schedule(self.optimizer.lr),
                    momentum=self.optimizer.momentum,
                )
        if self.clip_global_norm is None:
            return tx
        return optax.chain(optax.clip_by_global_norm(self.clip_global_norm), tx)

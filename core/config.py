"""Configs shared by every task."""

import enum
import pathlib

import attrs
from attrs import validators as v

from core.distributed import MeshConfig
from core.optim import OptimConfig


def _divisible_by_steps_per_call(
    instance: "LoopConfig", attribute: "attrs.Attribute[int]", value: int
) -> None:
    if value % instance.steps_per_call:
        raise ValueError(
            f"num_steps={value} is not a multiple of "
            f"steps_per_call={instance.steps_per_call}"
        )


class InnerLoop(enum.Enum):
    """How the `steps_per_call` steps of one compiled call are strung together.

    SCAN compiles the step once (`lax.scan`); compile time is flat in K. Prefer
    it, except for conv nets on XLA:CPU, where a convolution inside a while
    loop runs ~40x slower than outside one (measured on M5, jax 0.11).

    UNROLL inlines K copies of the step; compile time grows with K, runtime
    matches a plain jitted step.
    """

    SCAN = enum.auto()
    UNROLL = enum.auto()


@attrs.frozen(kw_only=True)
class LoopConfig:
    # Defined first: the num_steps validator reads it.
    steps_per_call: int = attrs.field(default=1, validator=v.gt(0))
    num_steps: int = attrs.field(validator=[v.gt(0), _divisible_by_steps_per_call])
    seed: int = attrs.field(default=0, validator=v.ge(0))
    inner_loop: InnerLoop = InnerLoop.SCAN
    # Over every device in the job; the default is pure data parallelism.
    mesh: MeshConfig = MeshConfig()


@attrs.frozen(kw_only=True)
class ExperimentConfig[TaskConfigT]:
    """One run: the task's own config plus the parts every task shares."""

    task: TaskConfigT
    loop: LoopConfig
    optim: OptimConfig
    workdir: pathlib.Path = attrs.field(converter=pathlib.Path)

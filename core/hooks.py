"""In-jit step hooks: pure post-processing composed into the compiled train step.

A hook sees the state before and after a step plus the step's metrics, and
returns a new state and metrics. Because it runs inside `jit`/`scan`, it must
obey the same rules as `Task.train_step`. Host-side work (I/O) belongs in
`core.callbacks` instead.

Hooks reach into state through *lenses* -- typed getter/setter functions --
never through string paths, so a renamed field is a type error, not a silent
no-op.
"""

import abc
from collections.abc import Callable, Sequence

import attrs
import jax
from attrs import validators as v


@attrs.frozen(kw_only=True)
class StepHook[StateT, MetricsT](abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, before: StateT, after: StateT, metrics: MetricsT
    ) -> tuple[StateT, MetricsT]: ...


def with_hooks[StateT, BatchT, MetricsT](
    step: Callable[[StateT, BatchT], tuple[StateT, MetricsT]],
    hooks: Sequence[StepHook[StateT, MetricsT]],
) -> Callable[[StateT, BatchT], tuple[StateT, MetricsT]]:
    """`step` followed by each hook in order."""
    hooks = tuple(hooks)

    def hooked(state: StateT, batch: BatchT) -> tuple[StateT, MetricsT]:
        after, metrics = step(state, batch)
        for hook in hooks:
            after, metrics = hook(state, after, metrics)
        return after, metrics

    return hooked


@attrs.frozen(kw_only=True)
class EmaHook[StateT, MetricsT, ParamsT](StepHook[StateT, MetricsT]):
    """Exponential moving average of params: `ema <- decay * ema + (1 - decay) * p`."""

    decay: float = attrs.field(validator=[v.ge(0.0), v.lt(1.0)])
    params: Callable[[StateT], ParamsT]
    get_ema: Callable[[StateT], ParamsT]
    set_ema: Callable[[StateT, ParamsT], StateT]

    def __call__(
        self, before: StateT, after: StateT, metrics: MetricsT
    ) -> tuple[StateT, MetricsT]:
        d = self.decay
        ema = jax.tree.map(
            lambda e, p: d * e + (1.0 - d) * p, self.get_ema(after), self.params(after)
        )
        return self.set_ema(after, ema), metrics


@attrs.frozen(kw_only=True)
class NanGuardHook[StateT, MetricsT](StepHook[StateT, MetricsT]):
    """Drop a step whose loss is not finite: keep `before`, report the metrics.

    A dropped step does not advance `Task.step_of`, so the loop's counter and
    the state's can drift apart by the number of dropped steps.
    """

    loss: Callable[[MetricsT], jax.Array]

    def __call__(
        self, before: StateT, after: StateT, metrics: MetricsT
    ) -> tuple[StateT, MetricsT]:
        finite = jax.numpy.isfinite(self.loss(metrics))
        state = jax.lax.cond(finite, lambda: after, lambda: before)
        return state, metrics

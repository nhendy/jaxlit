"""The interface every project implements."""

import abc

import attrs
import jax

from core.types import PRNGKey


@attrs.frozen(kw_only=True)
class Task[StateT, BatchT, MetricsT, EvalMetricsT](abc.ABC):
    """A training problem: how to build state, take a step, and evaluate.

    Subclasses must be `@attrs.frozen(kw_only=True)` and hold whatever they need
    (model, optimizer, env) as typed fields -- there is deliberately no `model`
    on the base, so a subclass's fields keep their concrete types.

    Every method is pure and jit-safe: no I/O, no python side effects, no
    branching on array values. The framework jits and scans them; it never
    calls them eagerly on a hot path.
    """

    @abc.abstractmethod
    def init(self, rng: PRNGKey, example: BatchT) -> StateT:
        """Fresh state. `example` supplies shapes only; its values are unused."""

    @abc.abstractmethod
    def train_step(self, state: StateT, batch: BatchT) -> tuple[StateT, MetricsT]:
        """One step of training."""

    @abc.abstractmethod
    def eval_step(self, state: StateT, batch: BatchT, mask: jax.Array) -> EvalMetricsT:
        """Metrics for one eval batch. Merged across batches by the framework.

        `mask` is `(b,)` bool: False rows are padding the framework added so
        every device and every host sees equal-shaped batches
        (`core.distributed.Partitioner.eval_batches`). Pass it to each clu
        metric -- `Average.from_model_output(loss, mask=mask)` -- and eval
        counts every real example exactly once. A leafless batch (`NoBatch`)
        is never padded and gets a `(0,)` mask.
        """

    def describe(self, example: BatchT) -> str | None:
        """A human-readable model summary, logged once when `fit` starts.

        Typically `self.model.tabulate(<model input built from example>)`. The
        default is nothing; parameter and FLOP totals are logged regardless.
        """
        return None

    @abc.abstractmethod
    def step_of(self, state: StateT) -> jax.Array:
        """The number of `train_step` calls `state` has seen; the loop resumes here."""

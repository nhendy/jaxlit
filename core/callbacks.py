"""Host-side callbacks: everything with a side effect.

A callback is a frozen description of what to do; whatever it accumulates
lives in an explicit `CbStateT` threaded through the loop, so the callback
object itself never mutates. The resources it writes to (metric writers,
checkpoint managers) are passed in as fields: that is the impure edge.

Each loop iteration runs `steps_per_call` train steps and then calls, for each
callback:

* `observe` -- always. Cheap accumulation, e.g. merging metrics.
* `fire` -- when the callback's `Trigger` says so. The expensive part: write
  logs, save, evaluate, decide to stop.
"""

import abc
import enum
import time
from collections.abc import Callable, Iterable
from typing import Any, cast

import attrs
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from attrs import validators as v

from core import metrics as metrics_lib
from core import stats
from core.distributed import EvalBatch, Partitioner
from core.metrics import Metrics
from core.stats import ModelSummary
from core.task import Task
from core.types import jit
from core.writers import MetricWriter


class Control(enum.Enum):
    CONTINUE = enum.auto()
    STOP = enum.auto()


class Split(enum.Enum):
    """Which stream a logged scalar belongs to; its value prefixes the name."""

    TRAIN = "train"
    EVAL = "eval"


class Direction(enum.Enum):
    MINIMIZE = enum.auto()
    MAXIMIZE = enum.auto()


@attrs.frozen(kw_only=True)
class Every:
    """Fire whenever the step crosses a multiple of `n` -- and, with
    `and_at_end`, after the final step too (once, even if it is a multiple)."""

    n: int = attrs.field(validator=v.gt(0))
    and_at_end: bool = False


@attrs.frozen(kw_only=True)
class AtEnd:
    """Fire once, after the final step."""


type Trigger = Every | AtEnd


def fires(trigger: Trigger, *, before: int, after: int, final: bool) -> bool:
    """Whether `trigger` fires for a loop iteration that took steps `before -> after`.

    Crossing rather than divisibility, so `Every(100)` still fires once per
    hundred steps when `steps_per_call` does not divide 100.
    """
    match trigger:
        case Every(n=n, and_at_end=and_at_end):
            return before // n != after // n or (and_at_end and final)
        case AtEnd():
            return final


@attrs.frozen(kw_only=True)
class StepEvent[StateT, MetricsT]:
    """What a callback sees after a loop iteration. Host side, never traced."""

    step: int
    state: StateT
    # Train steps run in this iteration (`steps_per_call`).
    num_steps: int
    # Merged over those steps.
    metrics: MetricsT
    final: bool
    # Params and FLOPs per step, measured once when `fit` starts.
    summary: ModelSummary | None = None
    # Wall time this iteration spent waiting on the train data iterator...
    data_seconds: float = 0.0
    # ...and stacking that chunk and copying it to the devices.
    prep_seconds: float = 0.0


@attrs.frozen(kw_only=True)
class Callback[StateT, MetricsT, CbStateT](abc.ABC):
    trigger: Trigger

    @abc.abstractmethod
    def begin(self, state: StateT) -> CbStateT: ...

    def observe(self, cb: CbStateT, event: StepEvent[StateT, MetricsT]) -> CbStateT:
        return cb

    @abc.abstractmethod
    def fire(
        self, cb: CbStateT, event: StepEvent[StateT, MetricsT]
    ) -> tuple[CbStateT, Control]: ...

    def end(self, cb: CbStateT, state: StateT) -> None:
        return None


# --- Logging -----------------------------------------------------------------


@attrs.frozen(kw_only=True)
class LogState[MetricsT]:
    pending: MetricsT | None
    # Steps and wall time since the previous firing.
    steps: int
    since: float
    # Of the wall time since `since`, how much `fit` spent waiting for data,
    # and preparing it (stack + host-to-device copy).
    data_seconds: float = 0.0
    prep_seconds: float = 0.0
    # Whether the run's constant `model/*` scalars have been written yet.
    reported: bool = False


@attrs.frozen(kw_only=True)
class LogMetrics[StateT, MetricsT: Metrics](
    Callback[StateT, MetricsT, LogState[MetricsT]]
):
    """Merge train metrics between firings, then write their mean."""

    writer: MetricWriter
    # Logged as train/learning_rate when set: `OptimConfig.schedule()`. Read at
    # the step count, so only right where one train step is one optimizer
    # update (not PPO, which updates per minibatch).
    learning_rate: optax.Schedule | None = None

    def begin(self, state: StateT) -> LogState[MetricsT]:
        return LogState(pending=None, steps=0, since=time.perf_counter())

    def observe(
        self, cb: LogState[MetricsT], event: StepEvent[StateT, MetricsT]
    ) -> LogState[MetricsT]:
        pending = (
            event.metrics
            if cb.pending is None
            else metrics_lib.merge(cb.pending, event.metrics)
        )
        return attrs.evolve(
            cb,
            pending=pending,
            steps=cb.steps + event.num_steps,
            data_seconds=cb.data_seconds + event.data_seconds,
            prep_seconds=cb.prep_seconds + event.prep_seconds,
        )

    def fire(
        self, cb: LogState[MetricsT], event: StepEvent[StateT, MetricsT]
    ) -> tuple[LogState[MetricsT], Control]:
        assert cb.pending is not None, "observe runs before fire"
        prefix = f"{Split.TRAIN.value}/"
        now = time.perf_counter()
        # Averaged since the previous firing: with async dispatch, timing a
        # single call measures only how long it took to enqueue.
        elapsed = now - cb.since
        steps_per_sec = cb.steps / elapsed
        scalars = [
            *metrics_lib.scalars(cb.pending, prefix=prefix),
            (f"{prefix}steps_per_sec", steps_per_sec),
            # Device work is dispatched asynchronously, so time the loop spends
            # blocked on the data iterator is time the accelerator can idle.
            # Near 1: input-bound -- more workers, or less per-example work.
            (f"{prefix}input_wait_frac", cb.data_seconds / elapsed),
            # Host time turning the chunk into device arrays. High with low
            # input_wait: the batches are too big to ship, not too slow to make.
            (f"{prefix}host_prep_frac", cb.prep_seconds / elapsed),
            *stats.device_memory(),
        ]
        if self.learning_rate is not None:
            lr = self.learning_rate(event.step)
            scalars.append((f"{prefix}learning_rate", float(np.asarray(lr))))
        if event.summary is not None:
            flops = event.summary.flops.per_step
            # Achieved model FLOP/s, whole job. Divide by a device's peak for MFU.
            scalars.append(
                (f"{prefix}model_tflops_per_sec", flops * steps_per_sec / 1e12)
            )
            if event.summary.bytes_per_step is not None:
                # Achieved memory bandwidth per device; over the device's peak,
                # how close the step is to bandwidth-bound. Low utilization
                # with a busy GPU: launch overhead or latency, not bandwidth.
                gb_per_sec = event.summary.bytes_per_step * steps_per_sec / 1e9
                scalars.append((f"{prefix}hbm_gb_per_sec", gb_per_sec))
                peak = stats.peak_bandwidth(jax.local_devices()[0].device_kind)
                if peak is not None:
                    scalars.append((f"{prefix}hbm_utilization", gb_per_sec / peak))
            if not cb.reported:
                scalars.extend(_summary_scalars(event.summary))
        self.writer.write_scalars(event.step, scalars)
        # Per firing, so file-tailing readers (TensorBoard, mlcompare) see a live run.
        self.writer.flush()
        reported = cb.reported or event.summary is not None
        return (
            LogState(pending=None, steps=0, since=now, reported=reported),
            Control.CONTINUE,
        )

    def end(self, cb: LogState[MetricsT], state: StateT) -> None:
        self.writer.flush()


def _summary_scalars(summary: ModelSummary) -> list[tuple[str, float]]:
    """Constants of the run, written once so they land in W&B and the CSV."""
    out = [("model/flops_per_step", summary.flops.per_step)]
    if summary.flops.per_example is not None:
        out.append(("model/flops_per_example", summary.flops.per_example))
    if summary.params is not None:
        out.append(("model/params", float(summary.params.count)))
    if summary.bytes_per_step is not None:
        out.append(("model/bytes_per_step", summary.bytes_per_step))
    return out


@attrs.frozen(kw_only=True)
class DataProgress[StateT, MetricsT](Callback[StateT, MetricsT, None]):
    """How much of the data the run uses, relative to the whole dataset.

    Per firing:
      data/epochs: passes over the training set so far.
      data/train_coverage: fraction of the training set seen at least once
        (reaches 1 at the end of the first epoch; shuffling is a permutation).
    Constants, written every firing so they sit next to the progress:
      data/train_fraction: training set / training pool (below 1 when eval
        shards are carved out of the training directory).
      data/eval_fraction: eval set / the pool it is drawn from.
    Raw counts ride along under data/*_examples.
    """

    writer: MetricWriter
    # Examples per train step, across every process.
    global_batch_size: int = attrs.field(validator=v.gt(0))
    train_examples: int = attrs.field(validator=v.gt(0))
    # Every example the training set could have been: the full train split.
    train_pool: int = attrs.field(validator=v.gt(0))
    eval_examples: int = attrs.field(validator=v.ge(0))
    # What the eval examples are drawn from (e.g. all of a validation split).
    eval_pool: int = attrs.field(validator=v.gt(0))

    def begin(self, state: StateT) -> None:
        return None

    def fire(
        self, cb: None, event: StepEvent[StateT, MetricsT]
    ) -> tuple[None, Control]:
        seen = event.step * self.global_batch_size
        self.writer.write_scalars(
            event.step,
            [
                ("data/epochs", seen / self.train_examples),
                ("data/train_coverage", min(seen / self.train_examples, 1.0)),
                ("data/train_fraction", self.train_examples / self.train_pool),
                ("data/eval_fraction", self.eval_examples / self.eval_pool),
                ("data/train_examples_seen", float(seen)),
                ("data/train_examples", float(self.train_examples)),
                ("data/eval_examples", float(self.eval_examples)),
            ],
        )
        return None, Control.CONTINUE


# --- Evaluation --------------------------------------------------------------


type EvalStep[StateT, BatchT, EvalMetricsT] = Callable[
    [StateT, EvalBatch[BatchT]], EvalMetricsT
]


def compile_eval[StateT, BatchT, EvalMetricsT: Metrics](
    task: Task[StateT, BatchT, Any, EvalMetricsT], partitioner: Partitioner
) -> EvalStep[StateT, BatchT, EvalMetricsT]:
    """`task.eval_step` jitted, with metrics returned replicated."""

    def step(state: StateT, batch: EvalBatch[BatchT]) -> EvalMetricsT:
        return task.eval_step(state, batch.batch, batch.mask)

    return jit(step, out_shardings=partitioner.replicated)


def evaluate[StateT, BatchT, EvalMetricsT: Metrics](
    task: Task[StateT, BatchT, Any, EvalMetricsT],
    state: StateT,
    batches: Iterable[BatchT],
    *,
    partitioner: Partitioner | None = None,
    eval_step: EvalStep[StateT, BatchT, EvalMetricsT] | None = None,
) -> EvalMetricsT:
    """Merged eval metrics over every batch of a finite iterable.

    `batches` are process-local; see `Partitioner.eval_batches` for how they
    are padded and kept in lockstep across processes.
    """
    partitioner = Partitioner.create() if partitioner is None else partitioner
    step = compile_eval(task, partitioner) if eval_step is None else eval_step
    total: EvalMetricsT | None = None
    for batch in partitioner.eval_batches(batches):
        m = step(state, batch)
        total = m if total is None else metrics_lib.merge(total, m)
    if total is None:
        raise ValueError("evaluate got no batches")
    return total


@attrs.frozen(kw_only=True)
class EvalState[EvalMetricsT]:
    last: EvalMetricsT | None


@attrs.frozen(kw_only=True)
class Evaluate[StateT, BatchT, MetricsT, EvalMetricsT: Metrics](
    Callback[StateT, MetricsT, EvalState[EvalMetricsT]]
):
    """Run `task.eval_step` over `batches` and log the result.

    `batches` is iterated afresh on every firing, so it must be re-iterable (a
    tuple, a Grain dataset) -- not a generator.
    """

    task: Task[StateT, BatchT, MetricsT, EvalMetricsT]
    batches: Iterable[BatchT]
    writer: MetricWriter
    # Must be the one `fit` trains on.
    partitioner: Partitioner = attrs.field(factory=Partitioner.create)
    eval_step: EvalStep[StateT, BatchT, EvalMetricsT] = attrs.field(
        init=False,
        default=attrs.Factory(
            lambda self: compile_eval(self.task, self.partitioner), takes_self=True
        ),
    )

    def begin(self, state: StateT) -> EvalState[EvalMetricsT]:
        return EvalState(last=None)

    def fire(
        self, cb: EvalState[EvalMetricsT], event: StepEvent[StateT, MetricsT]
    ) -> tuple[EvalState[EvalMetricsT], Control]:
        result = evaluate(
            self.task,
            event.state,
            self.batches,
            partitioner=self.partitioner,
            eval_step=self.eval_step,
        )
        self.writer.write_scalars(
            event.step, metrics_lib.scalars(result, prefix=f"{Split.EVAL.value}/")
        )
        self.writer.flush()
        return EvalState(last=result), Control.CONTINUE

    def end(self, cb: EvalState[EvalMetricsT], state: StateT) -> None:
        self.writer.flush()


# --- Early stopping ----------------------------------------------------------


@attrs.frozen(kw_only=True)
class StopState:
    best: float | None
    bad_firings: int


@attrs.frozen(kw_only=True)
class EarlyStop[StateT, MetricsT](Callback[StateT, MetricsT, StopState]):
    """Stop after `patience` firings without improving `monitor`.

    `monitor` is a lens onto the metrics -- `lambda m: m.loss.compute()` -- not a
    metric name.
    """

    monitor: Callable[[MetricsT], jax.Array]
    direction: Direction
    patience: int = attrs.field(validator=v.gt(0))
    min_delta: float = attrs.field(default=0.0, validator=v.ge(0.0))

    def begin(self, state: StateT) -> StopState:
        return StopState(best=None, bad_firings=0)

    def fire(
        self, cb: StopState, event: StepEvent[StateT, MetricsT]
    ) -> tuple[StopState, Control]:
        value = float(self.monitor(event.metrics))
        sign = 1.0 if self.direction is Direction.MINIMIZE else -1.0
        if cb.best is None or sign * (cb.best - value) > self.min_delta:
            return StopState(best=value, bad_firings=0), Control.CONTINUE
        bad = cb.bad_firings + 1
        control = Control.STOP if bad >= self.patience else Control.CONTINUE
        return StopState(best=cb.best, bad_firings=bad), control


# --- Checkpointing -----------------------------------------------------------


@attrs.frozen(kw_only=True)
class Checkpoint[StateT, MetricsT](Callback[StateT, MetricsT, None]):
    """Save the full state with orbax. Restore with `restore_latest`."""

    manager: ocp.CheckpointManager

    def begin(self, state: StateT) -> None:
        return None

    def fire(
        self, cb: None, event: StepEvent[StateT, MetricsT]
    ) -> tuple[None, Control]:
        self.manager.save(event.step, args=ocp.args.StandardSave(event.state))
        return None, Control.CONTINUE

    def end(self, cb: None, state: StateT) -> None:
        self.manager.wait_until_finished()


def restore_latest[StateT](
    manager: ocp.CheckpointManager, abstract: StateT
) -> StateT | None:
    """The latest checkpoint, shaped like `abstract`, or None if there is none.

    `abstract` is typically `Partitioner.abstract_state(task.init, rng, example,
    sharding)`: its leaves carry shardings, so each process restores only its
    own shards, straight into the layout `fit` trains in.
    """
    step = manager.latest_step()
    if step is None:
        return None
    return cast(StateT, manager.restore(step, args=ocp.args.StandardRestore(abstract)))

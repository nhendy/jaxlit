"""The framework end to end on a task small enough to check by hand: y = 3x + 1."""

import dataclasses
import itertools
import pathlib

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from clu import metrics as clu_metrics
from flax import struct

from core import metrics as metrics_lib
from core.callbacks import (
    AtEnd,
    Callback,
    Checkpoint,
    Control,
    DataProgress,
    Direction,
    EarlyStop,
    Evaluate,
    Every,
    LogMetrics,
    StepEvent,
    evaluate,
    fires,
    restore_latest,
)
from core.config import InnerLoop, LoopConfig
from core.hooks import EmaHook, NanGuardHook
from core.loop import fit
from core.metrics import Metrics
from core.optim import Constant, OptimConfig, Sgd
from core.state import TrainState
from core.task import Task
from core.types import HostOrDeviceArray, PRNGKey
from core.writers import InMemoryWriter


@struct.dataclass
class Batch:
    x: HostOrDeviceArray
    y: HostOrDeviceArray


@struct.dataclass
class Params:
    w: jax.Array
    b: jax.Array


@struct.dataclass
class Extra:
    ema: Params


@struct.dataclass
class LinMetrics(Metrics):
    loss: clu_metrics.Average


type LinState = TrainState[Params, Extra]


@attrs.frozen(kw_only=True)
class LinearTask(Task[LinState, Batch, LinMetrics, LinMetrics]):
    tx: optax.GradientTransformation

    def init(self, rng: PRNGKey, example: Batch) -> LinState:
        params = Params(w=jnp.zeros(()), b=jnp.zeros(()))
        return TrainState.create(
            params=params, tx=self.tx, rng=rng, extra=Extra(params)
        )

    def _loss(self, params: Params, batch: Batch) -> jax.Array:
        return jnp.mean((params.w * batch.x + params.b - batch.y) ** 2)

    def train_step(self, state: LinState, batch: Batch) -> tuple[LinState, LinMetrics]:
        loss, grads = jax.value_and_grad(self._loss)(state.params, batch)
        metrics = LinMetrics(loss=clu_metrics.Average.from_model_output(loss))
        return state.apply_gradients(self.tx, grads), metrics

    def eval_step(self, state: LinState, batch: Batch, mask: jax.Array) -> LinMetrics:
        p = state.params
        loss = (p.w * jnp.asarray(batch.x) + p.b - jnp.asarray(batch.y)) ** 2
        return LinMetrics(loss=clu_metrics.Average.from_model_output(loss, mask=mask))

    def step_of(self, state: LinState) -> jax.Array:
        return state.step


def make_task() -> LinearTask:
    return LinearTask(tx=OptimConfig(optimizer=Sgd(lr=Constant(value=0.1))).make())


def batches(seed: int = 0, size: int = 32):
    rng = np.random.default_rng(seed)
    while True:
        x = rng.normal(size=size).astype(np.float32)
        yield Batch(x=x, y=3.0 * x + 1.0)


def finite_batches(n: int) -> tuple[Batch, ...]:
    return tuple(itertools.islice(batches(seed=1), n))


@attrs.frozen(kw_only=True)
class Record[StateT, MetricsT](Callback[StateT, MetricsT, tuple[int, ...]]):
    """Remembers the steps it fired at."""

    def begin(self, state: StateT) -> tuple[int, ...]:
        return ()

    def fire(
        self, cb: tuple[int, ...], event: StepEvent[StateT, MetricsT]
    ) -> tuple[tuple[int, ...], Control]:
        return (*cb, event.step), Control.CONTINUE


@pytest.mark.parametrize("steps_per_call", [1, 5])
def test_fit_learns_the_line(steps_per_call: int) -> None:
    task = make_task()
    state = fit(
        task,
        config=LoopConfig(num_steps=200, steps_per_call=steps_per_call),
        train=batches(),
    )
    assert int(state.step) == 200
    np.testing.assert_allclose(state.params.w, 3.0, atol=1e-3)
    np.testing.assert_allclose(state.params.b, 1.0, atol=1e-3)
    m = evaluate(task, state, finite_batches(3))
    assert float(m.loss.compute()) < 1e-5


@pytest.mark.parametrize("inner_loop", list(InnerLoop))
def test_inner_loop_matches_single_steps(inner_loop: InnerLoop) -> None:
    task = make_task()
    one = fit(task, config=LoopConfig(num_steps=20, steps_per_call=1), train=batches())
    config = LoopConfig(num_steps=20, steps_per_call=10, inner_loop=inner_loop)
    many = fit(task, config=config, train=batches())
    np.testing.assert_allclose(one.params.w, many.params.w, rtol=1e-6)


def test_triggers() -> None:
    assert fires(Every(n=10), before=5, after=10, final=False)
    assert not fires(Every(n=10), before=10, after=15, final=False)
    # K=7 does not divide 10: still fires once per crossing.
    assert fires(Every(n=10), before=7, after=14, final=False)
    assert fires(AtEnd(), before=0, after=1, final=True)
    assert not fires(AtEnd(), before=0, after=1, final=False)


def test_every_and_at_end_fires_once_at_the_end() -> None:
    trigger = Every(n=10, and_at_end=True)
    assert fires(trigger, before=5, after=10, final=False)
    assert not fires(trigger, before=10, after=15, final=False)
    assert fires(trigger, before=15, after=17, final=True)  # final, not a multiple
    assert fires(trigger, before=15, after=20, final=True)  # both: still one firing
    assert not fires(Every(n=10), before=15, after=17, final=True)


def test_callbacks_fire_on_schedule() -> None:
    fired: list[tuple[int, ...]] = []

    @attrs.frozen(kw_only=True)
    class Capture(Record[LinState, LinMetrics]):
        def end(self, cb: tuple[int, ...], state: LinState) -> None:
            fired.append(cb)

    fit(
        make_task(),
        config=LoopConfig(num_steps=30, steps_per_call=5),
        train=batches(),
        callbacks=(Capture(trigger=Every(n=10)), Capture(trigger=AtEnd())),
    )
    assert fired == [(10, 20, 30), (30,)]


def test_data_progress_reports_epochs_and_eval_share() -> None:
    writer = InMemoryWriter()
    fit(
        make_task(),
        config=LoopConfig(num_steps=20, steps_per_call=5),
        train=batches(size=32),
        callbacks=(
            DataProgress(
                trigger=Every(n=1),
                writer=writer,
                global_batch_size=32,
                train_examples=160,
                train_pool=200,
                eval_examples=50,
                eval_pool=200,
            ),
        ),
    )
    values = {(r.step, r.name): r.value for r in writer.scalars}
    assert values[(5, "data/train_coverage")] == 1.0  # 5 * 32 = one epoch
    assert values[(10, "data/epochs")] == pytest.approx(2.0)
    assert values[(20, "data/epochs")] == pytest.approx(4.0)
    assert values[(20, "data/train_examples_seen")] == 640
    assert values[(10, "data/eval_fraction")] == pytest.approx(0.25)
    assert values[(10, "data/train_fraction")] == pytest.approx(0.8)
    assert values[(10, "data/train_coverage")] == 1.0


def test_log_metrics_reports_the_scheduled_learning_rate() -> None:
    writer = InMemoryWriter()
    fit(
        make_task(),
        config=LoopConfig(num_steps=20, steps_per_call=5),
        train=batches(),
        callbacks=(
            LogMetrics(
                trigger=Every(n=10),
                writer=writer,
                learning_rate=lambda step: 0.1 / (1 + step),
            ),
        ),
    )
    lr = {r.step: r.value for r in writer.scalars if r.name == "train/learning_rate"}
    assert lr == {10: pytest.approx(0.1 / 11), 20: pytest.approx(0.1 / 21)}


def test_log_and_eval_write_typed_metrics() -> None:
    writer = InMemoryWriter()
    task = make_task()
    fit(
        task,
        config=LoopConfig(num_steps=20, steps_per_call=5),
        train=batches(),
        callbacks=(
            LogMetrics(trigger=Every(n=10), writer=writer),
            Evaluate(
                trigger=AtEnd(), task=task, batches=finite_batches(2), writer=writer
            ),
        ),
    )
    assert [(r.step, r.name) for r in writer.scalars] == [
        (10, "train/loss"),
        (10, "train/steps_per_sec"),
        (10, "train/input_wait_frac"),
        (10, "train/host_prep_frac"),
        (10, "train/model_tflops_per_sec"),
        # Memory traffic: XLA reports bytes on CPU too; CPU has no peak in
        # `PEAK_BANDWIDTH`, so no utilization.
        (10, "train/hbm_gb_per_sec"),
        # The run's constants, once, at the first firing.
        (10, "model/flops_per_step"),
        (10, "model/flops_per_example"),
        (10, "model/params"),
        (10, "model/bytes_per_step"),
        (20, "train/loss"),
        (20, "train/steps_per_sec"),
        (20, "train/input_wait_frac"),
        (20, "train/host_prep_frac"),
        (20, "train/model_tflops_per_sec"),
        (20, "train/hbm_gb_per_sec"),
        (20, "eval/loss"),
    ]
    values = {r.name: r.value for r in writer.scalars}
    # y = w x + b with scalar w, b: two params and no matmuls, so the
    # matmul+conv FLOP count is zero.
    assert values["model/params"] == 2
    assert values["model/flops_per_step"] == 0
    assert values["model/bytes_per_step"] > 0
    assert 0.0 <= values["train/input_wait_frac"] <= 1.0


def test_early_stop() -> None:
    state = fit(
        make_task(),
        config=LoopConfig(num_steps=1000, steps_per_call=1),
        train=batches(),
        callbacks=(
            EarlyStop(
                trigger=Every(n=1),
                # A constant never improves: first firing sets the best, the
                # next three are bad, and the third of those stops.
                monitor=lambda m: jnp.zeros(()),
                direction=Direction.MAXIMIZE,
                patience=3,
            ),
        ),
    )
    assert int(state.step) == 4


def test_ema_hook() -> None:
    hook = EmaHook[LinState, LinMetrics, Params](
        decay=0.5,
        params=lambda s: s.params,
        get_ema=lambda s: s.extra.ema,
        set_ema=lambda s, e: dataclasses.replace(s, extra=Extra(e)),
    )
    state = fit(
        make_task(),
        config=LoopConfig(num_steps=1),
        train=batches(),
        hooks=(hook,),
    )
    np.testing.assert_allclose(state.extra.ema.w, 0.5 * state.params.w)


def test_nan_guard_drops_bad_steps() -> None:
    def poisoned():
        for i, b in enumerate(batches()):
            yield b if i != 2 else Batch(x=b.x, y=b.y * np.nan)

    guard = NanGuardHook[LinState, LinMetrics](loss=lambda m: m.loss.compute())
    state = fit(
        make_task(),
        config=LoopConfig(num_steps=5),
        train=poisoned(),
        hooks=(guard,),
    )
    assert np.isfinite(state.params.w)
    assert int(state.step) == 4  # one of five steps dropped


def test_checkpoint_roundtrip(tmp_path: pathlib.Path) -> None:
    task = make_task()
    manager = ocp.CheckpointManager(tmp_path)
    saved = fit(
        task,
        config=LoopConfig(num_steps=10),
        train=batches(),
        callbacks=(Checkpoint(trigger=AtEnd(), manager=manager),),
    )
    abstract = jax.eval_shape(task.init, jax.random.key(0), next(batches()))
    restored = restore_latest(manager, abstract)
    assert restored is not None
    np.testing.assert_array_equal(restored.params.w, saved.params.w)
    assert (
        jax.random.key_data(restored.rng).tolist()
        == jax.random.key_data(saved.rng).tolist()
    )

    # Resuming continues from the restored step.
    resumed = fit(
        task, config=LoopConfig(num_steps=15), train=batches(), state=restored
    )
    assert int(resumed.step) == 15


def test_metrics_scalars_are_field_names() -> None:
    m = LinMetrics(loss=clu_metrics.Average.from_model_output(jnp.array([1.0, 3.0])))
    assert list(metrics_lib.scalars(m)) == [("loss", 2.0)]

"""A small MLP regression task whose runs are cheap to compare across layouts.

Shared by the in-process reference run and the fake-device worker processes,
so both sides train the *same* task on the *same* global batches.
"""

import itertools
from collections.abc import Iterator

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from clu import metrics as clu_metrics
from flax import struct

from core.callbacks import evaluate
from core.config import LoopConfig
from core.distributed import Fsdp, MeshConfig, Partitioner, Replicated, StateSharding
from core.loop import fit
from core.metrics import Metrics
from core.model import LinenModel
from core.state import TrainState
from core.task import Task
from core.types import HostOrDeviceArray, NoExtra, PRNGKey, Variables

IN, HIDDEN, OUT = 16, 256, 4
GLOBAL_BATCH = 16
EVAL_EXAMPLES = 37


@attrs.frozen(kw_only=True)
class ToyConfig:
    hidden: int = HIDDEN


@struct.dataclass
class ToyBatch:
    x: HostOrDeviceArray
    y: HostOrDeviceArray


@struct.dataclass
class ToyMetrics(Metrics):
    loss: clu_metrics.Average


type ToyState = TrainState[Variables, NoExtra]


class Mlp(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        return nn.Dense(OUT)(nn.tanh(nn.Dense(HIDDEN)(x)))


@attrs.frozen(kw_only=True)
class ToyTask(Task[ToyState, ToyBatch, ToyMetrics, ToyMetrics]):
    model: LinenModel[jax.Array, jax.Array] = LinenModel(module=Mlp())
    tx: optax.GradientTransformation = attrs.field(
        factory=lambda: optax.adam(1e-2), eq=False
    )

    def init(self, rng: PRNGKey, example: ToyBatch) -> ToyState:
        params = self.model.init(rng, jnp.asarray(example.x))
        return TrainState.create(params=params, tx=self.tx, rng=rng, extra=NoExtra())

    def _sq(self, params: Variables, batch: ToyBatch) -> jax.Array:
        pred = self.model.apply(params, jnp.asarray(batch.x), train=True)
        return jnp.mean((pred - jnp.asarray(batch.y)) ** 2, axis=-1)

    def train_step(
        self, state: ToyState, batch: ToyBatch
    ) -> tuple[ToyState, ToyMetrics]:
        loss, grads = jax.value_and_grad(lambda p: self._sq(p, batch).mean())(
            state.params
        )
        metrics = ToyMetrics(loss=clu_metrics.Average.from_model_output(loss))
        return state.apply_gradients(self.tx, grads), metrics

    def eval_step(
        self, state: ToyState, batch: ToyBatch, mask: jax.Array
    ) -> ToyMetrics:
        loss = self._sq(state.params, batch)
        return ToyMetrics(loss=clu_metrics.Average.from_model_output(loss, mask=mask))

    def step_of(self, state: ToyState) -> jax.Array:
        return state.step


def _data(seed: int, n: int) -> ToyBatch:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, IN)).astype(np.float32)
    w = np.random.default_rng(1234).normal(size=(IN, OUT)).astype(np.float32)
    return ToyBatch(x=x, y=np.sin(x @ w))


def global_batches() -> Iterator[ToyBatch]:
    """The same infinite sequence of global batches on every process."""
    for i in itertools.count():
        yield _data(i, GLOBAL_BATCH)


def local_slice(batch: ToyBatch, index: int, count: int) -> ToyBatch:
    """Process `index`'s contiguous rows of a global batch."""
    n = np.shape(batch.x)[0] // count
    return jax.tree.map(lambda a: a[index * n : (index + 1) * n], batch)


def eval_set() -> ToyBatch:
    return _data(10_000, EVAL_EXAMPLES)


def eval_batches(set_: ToyBatch, size: int) -> tuple[ToyBatch, ...]:
    n = np.shape(set_.x)[0]
    return tuple(
        jax.tree.map(lambda a: a[i : i + size], set_) for i in range(0, n, size)
    )


@attrs.frozen(kw_only=True)
class ToyResult:
    leaves: tuple[np.ndarray, ...] = attrs.field(eq=False)
    eval_loss: float
    eval_count: int
    specs: tuple[str, ...]


def flat_params(state: ToyState) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(x) for x in jax.tree.leaves(state.params))


def run(
    mesh: MeshConfig, sharding: StateSharding[ToyState], num_steps: int = 20
) -> ToyResult:
    """Train single-process on whatever devices exist; evaluate on the full set."""
    partitioner = Partitioner.create(mesh)
    task = ToyTask()
    state = fit(
        task,
        config=LoopConfig(num_steps=num_steps, steps_per_call=5, mesh=mesh),
        train=global_batches(),
        partitioner=partitioner,
        sharding=sharding,
    )
    result = evaluate(task, state, eval_batches(eval_set(), 8), partitioner=partitioner)
    return ToyResult(
        leaves=flat_params(state),
        eval_loss=float(result.loss.compute()),
        eval_count=int(result.loss.count),
        specs=tuple(str(x.sharding.spec) for x in jax.tree.leaves(state.params)),
    )


def _pack(result: ToyResult) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(len(jax.devices())),
        np.asarray(result.eval_loss),
        np.asarray(result.eval_count),
        np.asarray(result.specs),
        *result.leaves,
    )


def unpack(arrays: tuple[np.ndarray, ...]) -> tuple[int, ToyResult]:
    devices, loss, count, specs, *leaves = arrays
    return int(devices), ToyResult(
        leaves=tuple(leaves),
        eval_loss=float(loss),
        eval_count=int(count),
        specs=tuple(str(s) for s in specs),
    )


def dp() -> tuple[np.ndarray, ...]:
    """Pure data parallelism over every device."""
    return _pack(run(MeshConfig(), Replicated()))


def fsdp() -> tuple[np.ndarray, ...]:
    """2-way data x 4-way FSDP (needs 8 devices)."""
    return _pack(run(MeshConfig(data=2, fsdp=4), Fsdp(min_size=1024)))

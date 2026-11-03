"""MNIST classification: the smallest complete `Task`."""

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics
from flax import struct

from commmon.mlp import MLPConfig
from core.metrics import Metrics
from core.model import LinenModel
from core.state import TrainState
from core.task import Task
from core.types import NoExtra, PRNGKey, Variables
from tasks.mnist.data import MnistBatch


class Classifier(nn.Module):
    """Flattens images and runs `commmon.mlp.MLP`."""

    mlp: MLPConfig

    @nn.compact
    def __call__(self, image: jax.Array, *, train: bool) -> jax.Array:
        return self.mlp.make()(image.reshape(image.shape[0], -1), train=train)


@struct.dataclass
class MnistMetrics(Metrics):
    loss: clu_metrics.Average
    accuracy: clu_metrics.Accuracy


type MnistState = TrainState[Variables, NoExtra]


@attrs.frozen(kw_only=True)
class MnistTask(Task[MnistState, MnistBatch, MnistMetrics, MnistMetrics]):
    model: LinenModel[jax.Array, jax.Array]
    tx: optax.GradientTransformation

    def init(self, rng: PRNGKey, example: MnistBatch) -> MnistState:
        init_rng, rng = jax.random.split(rng)
        params = self.model.init(init_rng, jnp.asarray(example.image))
        return TrainState.create(params=params, tx=self.tx, rng=rng, extra=NoExtra())

    def _metrics(
        self, logits: jax.Array, batch: MnistBatch, mask: jax.Array | None = None
    ) -> MnistMetrics:
        labels = jnp.asarray(batch.label)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return MnistMetrics(
            loss=clu_metrics.Average.from_model_output(loss, mask=mask),
            accuracy=clu_metrics.Accuracy.from_model_output(
                logits=logits, labels=labels, mask=mask
            ),
        )

    def train_step(
        self, state: MnistState, batch: MnistBatch
    ) -> tuple[MnistState, MnistMetrics]:
        def loss_fn(params: Variables) -> tuple[jax.Array, jax.Array]:
            logits = self.model.apply(params, jnp.asarray(batch.image), train=True)
            labels = jnp.asarray(batch.label)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            return loss.mean(), logits

        grads, logits = jax.grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(self.tx, grads), self._metrics(logits, batch)

    def eval_step(
        self, state: MnistState, batch: MnistBatch, mask: jax.Array
    ) -> MnistMetrics:
        logits = self.model.apply(state.params, jnp.asarray(batch.image), train=False)
        return self._metrics(logits, batch, mask)

    def describe(self, example: MnistBatch) -> str:
        return self.model.tabulate(jnp.asarray(example.image))

    def step_of(self, state: MnistState) -> jax.Array:
        return state.step

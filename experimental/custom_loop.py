from collections.abc import Iterator, Mapping
from typing import Any, Literal, SupportsFloat, TypedDict, cast, override

from absl import logging
from clu import metric_writers
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxloop import actions
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import train_loop as train_loop_lib
from jaxloop import types as jl_types
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

Variables = Mapping[str, Any]
MutableUpdates = Mapping[str, Any]
LossAux = tuple[jax.Array, MutableUpdates]


class MnistBatch(TypedDict):
    image: jax.Array
    label: jax.Array


class MnistOutput(TypedDict):
    loss: jax.Array


class MLP(nn.Module):
    """A simple MLP model."""

    units: tuple[int, ...]
    norm: Literal['batch', 'layer']
    activation: Literal['relu', 'gelu', 'linear']

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        for i, unit in enumerate(self.units):
            x = nn.Dense(features=unit, use_bias=self.norm != 'batch')(x)
            if i != len(self.units) - 1:
                match self.norm:
                    case 'batch':
                        x = nn.BatchNorm(use_running_average=not train)(x)
                    case 'layer':
                        x = nn.LayerNorm()(x)
                    case _:
                        raise ValueError(f'Unknown norm {self.norm}')

                match self.activation:
                    case 'relu':
                        x = nn.relu(x)
                    case 'gelu':
                        x = nn.gelu(x)
                    case 'linear':
                        pass
                    case _:
                        raise ValueError(
                            f'Unknown activation {self.activation}')
        return x


class MNISTStep(step_lib.Step):
    """A single step for MNIST."""

    # Names of the model's mutable variable collections (everything except
    # `params`), discovered from the model at initialization. Empty for models
    # that only carry parameters (e.g. LayerNorm), populated for models with
    # extra state (e.g. `batch_stats` for BatchNorm).
    _mutable_collections: tuple[str, ...] = ()

    @override
    def initialize_model(
        self,
        spec: jl_types.BatchSpec,
        log_num_params: bool = False,
    ) -> jl_types.TrainState:
        """Initializes the model state on the image portion of the batch."""

        def init_fn(batch: MnistBatch) -> jl_types.TrainState:
            variables: Variables = self._model.init(
                self._base_prng, batch['image'], train=self.train)
            return step_lib.State.create(
                apply_fn=self._model.apply,
                tx=self._optimizer,
                **{k: v for k, v in variables.items() if k in self._STATE_KEYS},
            )

        batch: MnistBatch = cast(MnistBatch, step_lib.get_zeroed_batch(spec))
        if self._should_shard_batch:
            batch = cast(MnistBatch, self.shard_batch(batch))
        abstract_variables: Variables = jax.eval_shape(
            lambda b: self._model.init(
                self._base_prng, b['image'], train=self.train),
            batch,
        )
        self._mutable_collections = tuple(
            k for k in abstract_variables if k != 'params')
        state: jl_types.TrainState = self._partitioner.shard_init_fn(init_fn)(
            batch)
        if log_num_params:
            self.compute_num_params(state)
        return state

    @override
    def run(
        self,
        state: jl_types.TrainState,
        batch: MnistBatch,
    ) -> tuple[jl_types.TrainState, MnistOutput | None]:
        """A single step for MNIST."""
        x: jax.Array = batch['image']
        y: jax.Array = batch['label']

        def loss_fn(
            params: optax.Params,
            state: jl_types.TrainState,
            x: jax.Array,
            y: jax.Array,
        ) -> tuple[jax.Array, LossAux]:
            variables: Variables = {
                'params': params,
                **{name: getattr(state, name)
                   for name in self._mutable_collections},
            }
            rngs: Mapping[str, jax.Array] = (
                {'dropout': self.prng_key(state.step)} if self.train else {})
            logits: jax.Array
            updates: MutableUpdates
            logits, updates = state.apply_fn(
                variables,
                x,
                train=self.train,
                mutable=list(self._mutable_collections),
                rngs=rngs,
            )
            loss: jax.Array = optax.softmax_cross_entropy_with_integer_labels(
                logits, y).mean()
            return loss, (logits, updates)

        loss: jax.Array
        new_state: jl_types.TrainState
        if self.train:
            grads: optax.Params
            updates: MutableUpdates
            (loss, (_, updates)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params, state, x, y)
            new_state = state.apply_gradients(grads=grads, **updates)
        else:
            loss, (_, _) = loss_fn(state.params, state, x, y)
            new_state = state

        output: MnistOutput = {'loss': loss}
        return new_state, output


def mnist_dataset(batch_size: int) -> Iterator[MnistBatch]:
    """Returns an infinite iterator of flattened, normalized MNIST batches."""

    def preprocess(sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        image: tf.Tensor = tf.reshape(
            tf.cast(sample['image'], tf.float32) / 255.0, [-1])
        return {'image': image, 'label': tf.cast(sample['label'], tf.int32)}

    ds: tf.data.Dataset = (
        tfds.load(
            'mnist', split='train', as_supervised=False, shuffle_files=True)
        .map(preprocess)
        .shuffle(10_000)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    return cast(Iterator[MnistBatch], iter(tfds.as_numpy(ds)))


def main() -> None:
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    batch_size: int = 128
    num_steps: int = 10_000
    log_interval: int = 100
    model: MLP = MLP(units=(256, 128, 10), norm='layer', activation='relu')
    writer: metric_writers.MetricWriter = metric_writers.LoggingWriter()
    end_actions: list[actions.Action] = [
        actions.SummaryAction(writer, interval=log_interval)]
    train_step: MNISTStep = MNISTStep(
        base_prng=jax.random.PRNGKey(0),
        model=model,
        optimizer=optax.adam(1e-3),
        train=True,
        end_actions=end_actions,
    )
    batch_spec: jl_types.BatchSpec = {
        'image': ((batch_size, 784), jnp.float32),
        'label': ((batch_size,), jnp.int32),
    }
    state: jl_types.TrainState = train_step.initialize_model(
        batch_spec, log_num_params=True)
    stat_names: list[str] = [
        stat_loop.STAT_NUM_PARAMS,
        stat_loop.STAT_NUM_FLOPS,
        stat_loop.STAT_STEPS_PER_SEC,
        stat_loop.STAT_EXAMPLES_PER_SEC,
    ]
    train_loop: train_loop_lib.TrainLoop = train_loop_lib.TrainLoop(
        train_step,
        stat_names=stat_names,
        global_batch_size=batch_size,
    )
    outputs: jl_types.Output | None
    state, outputs = train_loop(
        state, mnist_dataset(batch_size), num_steps=num_steps)

    stat_values: dict[str, float] = {}
    if outputs is not None:
        for name in stat_names:
            if name in outputs:
                stat_values[name] = float(cast(SupportsFloat, outputs[name]))
    if stat_values:
        writer.write_scalars(int(state.step), stat_values)
        writer.flush()


if __name__ == '__main__':
    main()

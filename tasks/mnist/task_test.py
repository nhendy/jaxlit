"""MNIST task on synthetic data: no download, seconds on CPU."""

import numpy as np
import pytest

from commmon.mlp import MLPConfig
from core.callbacks import evaluate
from core.config import LoopConfig
from core.loop import fit
from core.optim import AdamW, Constant, OptimConfig
from tasks.mnist import data
from tasks.mnist.config import MnistConfig


def synthetic(n: int = 64) -> data.MnistArrays:
    rng = np.random.default_rng(0)
    return data.MnistArrays(
        images=rng.uniform(size=(n, 28, 28, 1)).astype(np.float32),
        labels=rng.integers(0, 10, size=n).astype(np.int32),
    )


def test_pipeline_yields_typed_batches() -> None:
    train = data.batches(synthetic(), batch_size=16, seed=0, training=True)
    batch = next(iter(train))
    assert isinstance(batch, data.MnistBatch)
    assert batch.image.shape == (16, 28, 28, 1)
    test = data.batches(synthetic(40), batch_size=16, seed=0, training=False)
    assert [b.label.shape[0] for b in test] == [16, 16, 8]


def test_batch_shapes_are_checked() -> None:
    with pytest.raises(ValueError):
        data.MnistBatch(image=np.zeros((4, 28, 28, 1)), label=np.zeros(3))


def test_memorizes_a_small_set() -> None:
    arrays = synthetic()
    task = MnistConfig(
        mlp=MLPConfig(units=(64, 10), norm="layer", activation="relu")
    ).make(OptimConfig(optimizer=AdamW(lr=Constant(value=1e-2))).make())
    train = data.batches(arrays, batch_size=32, seed=0, training=True)
    state = fit(task, config=LoopConfig(num_steps=100, steps_per_call=10), train=train)

    test = data.batches(arrays, batch_size=32, seed=0, training=False)
    result = evaluate(task, state, test)
    assert float(result.accuracy.compute()) == 1.0
    # Eval saw every example exactly once.
    assert int(result.accuracy.count) == len(arrays)

"""Distributed training on fake CPU devices and fake hosts.

Unit tests run in-process on the one default CPU device. The layout tests run
`core.testing.toy` in fresh processes -- one faking 8 devices, or two faking
2 devices each and talking over gloo -- and hold them to the in-process
single-device run of the same task on the same global batches.
"""

import enum
import pathlib

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

from core.callbacks import evaluate
from core.distributed import (
    Axis,
    Coordinator,
    Logical,
    MeshConfig,
    Partitioner,
    Replicated,
    Rule,
    Shard,
    batch_size_of,
    fsdp_spec,
    pad_batch,
    partitioned,
)
from core.testing import multidevice, multiprocess, toy
from tasks.mnist import data


@pytest.fixture(scope="module")
def reference() -> toy.ToyResult:
    assert len(jax.devices()) == 1, "the reference must be single-device"
    return toy.run(MeshConfig(), Replicated())


# --- unit ---------------------------------------------------------------------


def test_mesh_fills_one_axis() -> None:
    assert MeshConfig().shape(8) == (8, 1, 1)
    assert MeshConfig(data=2, fsdp=-1).shape(8) == (2, 4, 1)
    assert MeshConfig(data=2, fsdp=2, model=2).shape(8) == (2, 2, 2)


@pytest.mark.parametrize(
    "config, devices",
    [(MeshConfig(data=3), 8), (MeshConfig(data=2, fsdp=2), 8), (MeshConfig(fsdp=3), 8)],
)
def test_mesh_rejects_wrong_device_counts(config: MeshConfig, devices: int) -> None:
    with pytest.raises(ValueError):
        config.shape(devices)


def test_mesh_allows_one_wildcard() -> None:
    with pytest.raises(ValueError):
        MeshConfig(data=-1, fsdp=-1)


def test_coordinator_validates_process_id() -> None:
    with pytest.raises(ValueError):
        Coordinator(address="localhost:1", num_processes=2, process_id=2)


def test_fsdp_rule() -> None:
    assert fsdp_spec((256, 4), 4, 1024) == P("fsdp", None)
    assert fsdp_spec((16, 256), 4, 1024) == P(None, "fsdp")
    assert fsdp_spec((256,), 4, 1024) == P()  # small: replicated
    assert fsdp_spec((1023, 3), 4, 16) == P()  # nothing divides
    assert fsdp_spec((), 4, 1) == P()
    assert fsdp_spec((256, 4), 1, 1) == P()  # no FSDP axis


def test_pad_batch() -> None:
    batch = toy.ToyBatch(x=np.ones((5, 2)), y=np.ones((5,)))
    padded = pad_batch(batch, 8)
    assert batch_size_of(padded) == 8
    assert np.asarray(padded.x)[5:].sum() == 0
    assert batch_size_of(toy.ToyBatch(x=np.ones((0, 2)), y=np.ones((0,)))) == 0
    with pytest.raises(ValueError):
        batch_size_of(toy.ToyBatch(x=np.ones((5, 2)), y=np.ones((4,))))


def test_single_process_shard_is_everything() -> None:
    assert Shard.current() == Shard(index=0, count=1)


def test_mnist_shards_are_disjoint() -> None:

    arrays = data.MnistArrays(
        images=np.zeros((10, 28, 28, 1), np.float32),
        labels=np.arange(10, dtype=np.int32),
    )
    seen = [
        np.concatenate(
            [
                np.asarray(b.label)
                for b in data.batches(
                    arrays,
                    batch_size=4,
                    seed=0,
                    training=False,
                    shard=Shard(index=i, count=2),
                )
            ]
        )
        for i in range(2)
    ]
    assert set(seen[0]).isdisjoint(seen[1])
    assert sorted(np.concatenate(seen)) == list(range(10))


def test_eval_pads_and_counts_every_example(reference: toy.ToyResult) -> None:
    """37 examples in batches of 8: the short last batch is padded and masked."""
    assert reference.eval_count == toy.EVAL_EXAMPLES


def test_eval_rejects_a_batch_larger_than_the_first() -> None:
    task = toy.ToyTask()
    set_ = toy.eval_set()
    small, large = toy.eval_batches(set_, 4)[0], toy.eval_batches(set_, 8)[0]
    state = jax.jit(task.init)(jax.random.key(0), small)
    with pytest.raises(ValueError, match="largest"):
        evaluate(task, state, (small, large), partitioner=Partitioner.create())


# --- fake devices, one process -------------------------------------------------


def _assert_matches(got: toy.ToyResult, want: toy.ToyResult) -> None:
    assert got.eval_count == want.eval_count
    assert got.eval_loss == pytest.approx(want.eval_loss, rel=1e-5)
    for a, b in zip(got.leaves, want.leaves, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_data_parallel_matches_one_device(
    tmp_path: pathlib.Path, reference: toy.ToyResult
) -> None:
    out = str(tmp_path / "dp.npz")
    multidevice.run_with_devices("core.testing.toy:dp", out, devices=8)
    devices, got = toy.unpack(multidevice.load(out))
    assert devices == 8
    assert set(got.specs) == {"P()"}
    _assert_matches(got, reference)


def test_fsdp_matches_one_device(
    tmp_path: pathlib.Path, reference: toy.ToyResult
) -> None:
    out = str(tmp_path / "fsdp.npz")
    multidevice.run_with_devices("core.testing.toy:fsdp", out, devices=8)
    devices, got = toy.unpack(multidevice.load(out))
    assert devices == 8
    # Both kernels are split; the biases are below `min_size`.
    assert sorted(got.specs) == sorted(
        ["P()", "P()", "P('fsdp', None)", "P(None, 'fsdp')"]
    )
    _assert_matches(got, reference)


# --- fake hosts ------------------------------------------------------------------


def test_two_processes(tmp_path: pathlib.Path, reference: toy.ToyResult) -> None:
    outs = multiprocess.launch(2, tmp_path)
    results = [multidevice.load(o) for o in outs]
    for i, (index, loss, count, restored, stopped_at, _writers, *leaves) in enumerate(
        results
    ):
        assert int(index) == i
        # Uneven eval shards (24 + 13 rows) still count each example once.
        assert int(count) == toy.EVAL_EXAMPLES
        assert float(loss) == pytest.approx(reference.eval_loss, rel=1e-5)
        # One checkpoint, written by both, restores to the trained state.
        assert bool(restored)
        # Only process 1 asked to stop; both did, at the same step.
        assert int(stopped_at) == multiprocess.STOP_AT
        for a, b in zip(leaves, reference.leaves, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # Files come from the primary only: process 0 gets logging, TensorBoard and
    # CSV writers, process 1 logging alone.
    assert [int(r[5]) for r in results] == [3, 1]
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "config.yaml").exists()


# --- logical rules -------------------------------------------------------------


class _Feature(enum.Enum):
    EMBED = "embed"
    HIDDEN = "hidden"
    HEADS = "heads"


class _Annotated(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        init = nn.initializers.lecun_normal()
        h = nn.Dense(8, kernel_init=partitioned(init, _Feature.EMBED, _Feature.HIDDEN))(
            x
        )
        return nn.Dense(
            4, kernel_init=partitioned(init, _Feature.HIDDEN, _Feature.EMBED)
        )(nn.LayerNorm()(h))


def _abstract() -> object:
    tx = optax.adam(1e-3)

    def init(rng: jax.Array) -> object:
        params = _Annotated().init(rng, jnp.ones((1, 4)))
        return params, tx.init(params)

    return jax.eval_shape(init, jax.random.key(0))


def test_logical_rules_place_annotated_dims() -> None:
    strategy = Logical(rules=(Rule(logical=_Feature.HIDDEN, mesh=Axis.MODEL),))
    params, opt_state = nn.unbox(Partitioner.create().specs(_abstract(), strategy))
    assert params["params"]["Dense_0"]["kernel"] == P(None, "model")
    assert params["params"]["Dense_1"]["kernel"] == P("model", None)
    # Unannotated leaves are replicated.
    assert params["params"]["LayerNorm_0"]["scale"] == P()
    # Adam's moments inherit the annotations.
    assert opt_state[0].mu["params"]["Dense_1"]["kernel"] == P("model", None)


def test_logical_rules_without_a_match_replicate() -> None:
    specs = Partitioner.create().specs(_abstract(), Logical())
    assert set(
        jax.tree.leaves(nn.unbox(specs), is_leaf=lambda x: isinstance(x, P))
    ) == {
        P(),
        P(None, None),
    }


def test_logical_rules_validate() -> None:
    with pytest.raises(ValueError, match="more than one rule"):
        Logical(
            rules=(
                Rule(logical=_Feature.HIDDEN, mesh=Axis.MODEL),
                Rule(logical=_Feature.HIDDEN, mesh=Axis.FSDP),
            )
        )
    # Sharing a mesh axis is fine across arrays, not within one.
    clash = Logical(
        rules=(
            Rule(logical=_Feature.HIDDEN, mesh=Axis.MODEL),
            Rule(logical=_Feature.EMBED, mesh=Axis.MODEL),
        )
    )
    with pytest.raises(ValueError, match="two dims of one array"):
        Partitioner.create().specs(_abstract(), clash)

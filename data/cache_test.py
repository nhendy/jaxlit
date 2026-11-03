"""A cache must hold exactly what the pipeline would have produced."""

import itertools
import pathlib

import numpy as np
import pytest

from core.distributed import Shard
from data import cache
from data.ingest.writer import write_shards
from data.pipeline import Stages, build
from data.schema import Scenario, decode_scenario_id
from data.specs import AV2, Budget, TimeWindow
from data.testing import synthetic_scenario

NUM_SCENARIOS = 20
BUDGET = Budget(max_agents=8, max_polylines=1)
STAGES = Stages(
    ego_frame=False,
    augment=False,
    keep_map=False,
    window=TimeWindow(stride=2, history=15),
)


@pytest.fixture
def shards(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    rng = np.random.default_rng(0)
    scenarios = [
        synthetic_scenario(
            AV2,
            num_agents=int(rng.integers(3, 12)),
            num_polylines=int(rng.integers(5, 20)),
            rng=rng,
            scenario_id=f"av2-{i:04d}",
        )
        for i in range(NUM_SCENARIOS)
    ]
    return write_shards(scenarios, tmp_path / "shards", num_shards=4, prefix="av2")


def _materialize(shards: list[pathlib.Path], root: pathlib.Path) -> Scenario:
    directory = cache.materialize(
        shards, AV2, budget=BUDGET, stages=STAGES, root=root, batch_size=3
    )
    like = cache.template(shards, AV2, budget=BUDGET, stages=STAGES)
    return cache.load(directory, like)


def _ids(scenes: Scenario) -> list[str]:
    return [decode_scenario_id(row) for row in np.asarray(scenes.scenario_id)]


def test_cache_matches_the_pipeline(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    scenes = _materialize(shards, tmp_path / "cache")
    expected = next(
        iter(
            build(
                shards,
                AV2,
                batch_size=NUM_SCENARIOS,
                seed=0,
                budget=BUDGET,
                training=False,
                stages=STAGES,
            )
        )
    )
    assert _ids(scenes) == _ids(expected)
    np.testing.assert_array_equal(scenes.agents.position, expected.agents.position)
    np.testing.assert_array_equal(scenes.agents.valid, expected.agents.valid)
    assert scenes.spec == expected.spec


def test_materialize_is_idempotent(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    first = cache.materialize(shards, AV2, budget=BUDGET, stages=STAGES, root=tmp_path)
    stamp = (first / "agents.position.npy").stat().st_mtime_ns
    again = cache.materialize(shards, AV2, budget=BUDGET, stages=STAGES, root=tmp_path)
    assert again == first
    assert (again / "agents.position.npy").stat().st_mtime_ns == stamp


def test_a_different_budget_is_a_different_cache(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    other = Budget(max_agents=4, max_polylines=1)
    assert cache.materialize(
        shards, AV2, budget=BUDGET, stages=STAGES, root=tmp_path
    ) != cache.materialize(shards, AV2, budget=other, stages=STAGES, root=tmp_path)


def test_augmentation_cannot_be_cached(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(ValueError, match="augment=False"):
        cache.materialize(
            shards, AV2, budget=BUDGET, stages=Stages(augment=True), root=tmp_path
        )


def test_load_refuses_an_incomplete_cache(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        cache.load(tmp_path, synthetic_scenario(AV2, 3, 3, np.random.default_rng(0)))


def test_batches_cover_each_epoch_once(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    scenes = _materialize(shards, tmp_path)
    batches = cache.dataset(scenes, batch_size=5, seed=0)
    epoch = list(itertools.islice(iter(batches), NUM_SCENARIOS // 5))
    assert all(np.asarray(b.agents.position).shape[0] == 5 for b in epoch)
    assert sorted(i for b in epoch for i in _ids(b)) == sorted(_ids(scenes))


def test_batches_reshuffle_per_epoch_and_follow_the_seed(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    scenes = _materialize(shards, tmp_path)

    def ids(seed: int) -> list[list[str]]:
        it = iter(cache.dataset(scenes, batch_size=10, seed=seed))
        return [_ids(next(it)) for _ in range(4)]

    assert ids(0) == ids(0)
    assert ids(0) != ids(1)
    first, second = ids(0)[:2], ids(0)[2:]
    assert first != second


def test_shards_split_the_rows(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    scenes = _materialize(shards, tmp_path)
    seen = [
        {
            i
            for b in itertools.islice(
                iter(
                    cache.dataset(
                        scenes,
                        batch_size=5,
                        seed=0,
                        shard=Shard(index=k, count=2),
                    )
                ),
                2,
            )
            for i in _ids(b)
        }
        for k in range(2)
    ]
    assert not seen[0] & seen[1]
    assert len(seen[0] | seen[1]) == NUM_SCENARIOS


def test_eval_dataset_is_every_scene_once_in_order(
    shards: list[pathlib.Path], tmp_path: pathlib.Path
) -> None:
    scenes = _materialize(shards, tmp_path)
    ds = cache.eval_dataset(scenes, batch_size=6)
    for _ in range(2):  # re-iterable: eval sweeps it at every firing
        batches = list(ds)
        assert [np.asarray(b.agents.position).shape[0] for b in batches] == [6, 6, 6, 2]
        assert [i for b in batches for i in _ids(b)] == _ids(scenes)

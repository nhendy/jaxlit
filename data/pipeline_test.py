"""Transform and pipeline tests.

These run entirely on synthetic scenarios, so the Grain half of the library is
proven before any dataset adapter exists. `num_workers=0` throughout: Grain
spawns worker processes, and an exception raised inside a transform is far
easier to read in-process.
"""

import pathlib

import attrs
import grain
import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from data.ingest.writer import read_shards, write_shards
from data.pipeline import Stages, build
from data.schema import Scenario, decode_scenario_id
from data.specs import ALL_SPECS, AV2, DEFAULT_BUDGETS, Budget, DatasetSpec
from data.testing import synthetic_scenario
from data.transforms import AgentCentric, PadAndMask, RandomRotate

NUM_SCENARIOS = 20
BATCH = 4


@parameterized_fixture
def spec() -> dict[str, DatasetSpec]:
    return dict(ALL_SPECS)


@pytest.fixture
def budget(spec: DatasetSpec) -> Budget:
    return DEFAULT_BUDGETS[spec.name]


@pytest.fixture
def scenario(spec: DatasetSpec) -> Scenario:
    return synthetic_scenario(spec, 7, 13, np.random.default_rng(0))


@pytest.fixture
def shards(spec: DatasetSpec, tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """Ragged scenarios on disk -- agent and polyline counts deliberately vary."""
    rng = np.random.default_rng(0)
    scenarios = [
        synthetic_scenario(
            spec,
            num_agents=int(rng.integers(3, 12)),
            num_polylines=int(rng.integers(5, 20)),
            rng=rng,
            scenario_id=f"{spec.name}-{i:04d}",
        )
        for i in range(NUM_SCENARIOS)
    ]
    return write_shards(scenarios, tmp_path, num_shards=4, prefix=spec.name)


# --- transforms ---------------------------------------------------------------


def test_agent_centric_puts_ego_at_origin(scenario: Scenario) -> None:
    out = AgentCentric().map(scenario)
    ego = int(np.asarray(out.ego_index))
    t_ref = scenario.spec.num_history - 1
    np.testing.assert_allclose(
        np.asarray(out.agents.position)[ego, t_ref], [0.0, 0.0], atol=1e-4
    )
    assert abs(float(np.asarray(out.agents.heading)[ego, t_ref])) < 1e-4


def test_agent_centric_preserves_relative_geometry(scenario: Scenario) -> None:
    """SE(2) is rigid: pairwise distances at the reference step must not move."""
    t_ref = scenario.spec.num_history - 1
    before = np.asarray(scenario.agents.position)[:, t_ref]
    after = np.asarray(AgentCentric().map(scenario).agents.position)[:, t_ref]
    valid = np.asarray(scenario.agents.valid)[:, t_ref]
    idx = np.nonzero(valid)[0]
    d_before = np.linalg.norm(before[idx, None] - before[None, idx], axis=-1)
    d_after = np.linalg.norm(after[idx, None] - after[None, idx], axis=-1)
    np.testing.assert_allclose(d_after, d_before, atol=1e-3)


def test_agent_centric_zeroes_invalid_steps(scenario: Scenario) -> None:
    """Invalid entries must not carry plausible-looking coordinates."""
    out = AgentCentric().map(scenario)
    invalid = ~np.asarray(out.agents.valid)
    assert invalid.any(), "fixture should contain invalid steps"
    np.testing.assert_array_equal(np.asarray(out.agents.position)[invalid], 0.0)


def test_agent_centric_rejects_invalid_ego(scenario: Scenario) -> None:
    valid = np.asarray(scenario.agents.valid).copy()
    ego = int(np.asarray(scenario.ego_index))
    valid[ego, scenario.spec.num_history - 1] = False
    bad = scenario.replace(agents=scenario.agents.replace(valid=valid))
    with pytest.raises(ValueError, match="ego agent"):
        AgentCentric().map(bad)


def test_random_rotate_is_rigid(scenario: Scenario) -> None:
    centred = AgentCentric().map(scenario)
    rotated = RandomRotate().random_map(centred, np.random.default_rng(0))
    valid = np.asarray(centred.agents.valid)
    before = np.linalg.norm(np.asarray(centred.agents.position), axis=-1)
    after = np.linalg.norm(np.asarray(rotated.agents.position), axis=-1)
    np.testing.assert_allclose(after[valid], before[valid], atol=1e-3)


def test_random_rotate_is_seed_determined(scenario: Scenario) -> None:
    centred = AgentCentric().map(scenario)
    a = RandomRotate().random_map(centred, np.random.default_rng(7))
    b = RandomRotate().random_map(centred, np.random.default_rng(7))
    np.testing.assert_array_equal(
        np.asarray(a.agents.position), np.asarray(b.agents.position)
    )


def test_headings_stay_wrapped(scenario: Scenario) -> None:
    out = RandomRotate().random_map(
        AgentCentric().map(scenario), np.random.default_rng(3)
    )
    heading = np.asarray(out.agents.heading)
    assert np.all(np.abs(heading) <= np.pi + 1e-6)


# --- padding ------------------------------------------------------------------


def test_pad_produces_static_shapes(spec: DatasetSpec, budget: Budget) -> None:
    """Output shape must not depend on the input agent count."""
    pad = PadAndMask(DEFAULT_BUDGETS[spec.name])
    shapes = {
        np.asarray(pad.map(synthetic_scenario(spec, n, 5)).agents.position).shape
        for n in (2, 5, 11)
    }
    assert shapes == {(budget.max_agents, spec.num_steps, 2)}


def test_pad_masks_are_consistent_with_valid(spec: DatasetSpec) -> None:
    out = PadAndMask(DEFAULT_BUDGETS[spec.name]).map(synthetic_scenario(spec, 5, 7))
    mask = np.asarray(out.agent_mask)
    assert mask.sum() == 5
    # Padding rows are invalid at every step, by both routes.
    assert not np.asarray(out.agents.valid)[~mask].any()
    assert not np.asarray(out.vector_map.valid)[~np.asarray(out.polyline_mask)].any()


def test_pad_truncation_keeps_the_ego(spec: DatasetSpec, budget: Budget) -> None:
    """Over-budget scenarios rank by distance to ego, so the ego always survives."""
    tiny = attrs.evolve(budget, max_agents=3)
    scenario = synthetic_scenario(spec, num_agents=20, num_polylines=5)
    out = PadAndMask(tiny).map(scenario)

    assert np.asarray(out.agents.position).shape[0] == 3
    assert np.asarray(out.agent_mask).sum() == 3
    ego = int(np.asarray(out.ego_index))
    assert 0 <= ego < 3
    # The ego row must still be the ego's own trajectory.
    original_ego = int(np.asarray(scenario.ego_index))
    np.testing.assert_array_equal(
        np.asarray(out.agents.position)[ego],
        np.asarray(scenario.agents.position)[original_ego],
    )


def test_pad_truncates_polylines(spec: DatasetSpec, budget: Budget) -> None:
    tiny = attrs.evolve(budget, max_polylines=4)
    out = PadAndMask(tiny).map(synthetic_scenario(spec, 3, num_polylines=30))
    assert np.asarray(out.vector_map.points).shape[0] == 4
    assert np.asarray(out.polyline_mask).all()


def test_strict_budget_raises_instead_of_truncating(spec: DatasetSpec) -> None:
    """Eval wants an error: a silently truncated validation set flatters metrics."""
    tiny = Budget(max_agents=3, max_polylines=4, strict=True)
    with pytest.raises(ValueError, match="budget binds"):
        PadAndMask(tiny).map(synthetic_scenario(spec, 20, 30))


def test_non_strict_budget_truncates_quietly(spec: DatasetSpec) -> None:
    tiny = Budget(max_agents=3, max_polylines=4)
    out = PadAndMask(tiny).map(synthetic_scenario(spec, 20, 30))
    assert np.asarray(out.agents.position).shape[0] == 3


def test_a_budget_is_not_stored_on_the_scenario(spec: DatasetSpec) -> None:
    """Two budgets, same treedef -- so they still batch together."""
    a = PadAndMask(Budget(max_agents=8, max_polylines=8)).map(
        synthetic_scenario(spec, 5, 5)
    )
    assert not hasattr(a.spec, "max_agents")
    assert a.spec is spec


# --- ingest round trip --------------------------------------------------------


def test_shards_round_trip(shards: list[pathlib.Path], spec: DatasetSpec) -> None:
    restored = list(read_shards(shards))
    assert len(restored) == NUM_SCENARIOS
    ids = {decode_scenario_id(s.scenario_id) for s in restored}
    assert ids == {f"{spec.name}-{i:04d}" for i in range(NUM_SCENARIOS)}
    assert all(s.spec is spec for s in restored)


def test_write_shards_rejects_zero_shards(
    spec: DatasetSpec, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(ValueError, match="num_shards"):
        write_shards([synthetic_scenario(spec, 3, 5)], tmp_path, num_shards=0)


# --- pipeline -----------------------------------------------------------------


def test_pipeline_yields_static_batches(
    shards: list[pathlib.Path], spec: DatasetSpec, budget: Budget
) -> None:
    batch = next(iter(build(shards, spec, batch_size=BATCH, seed=0)))
    assert isinstance(batch, Scenario), "batching must not degrade to an object array"
    assert np.asarray(batch.agents.position).shape == (
        BATCH,
        budget.max_agents,
        spec.num_steps,
        2,
    )
    assert np.asarray(batch.vector_map.points).shape == (
        BATCH,
        budget.max_polylines,
        spec.max_points,
        2,
    )
    assert batch.spec is spec


def test_pipeline_is_finite_everywhere(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    """A NaN reaching jit becomes a silent NaN gradient, so check every leaf."""
    batch = next(iter(build(shards, spec, batch_size=BATCH, seed=0)))
    for name in ("position", "heading", "velocity", "extent"):
        array = np.asarray(getattr(batch.agents, name))
        assert np.isfinite(array).all(), name
    assert np.isfinite(np.asarray(batch.vector_map.points)).all()


def test_pipeline_is_deterministic_for_a_seed(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    def positions(seed: int) -> np.ndarray:
        ds = build(shards, spec, batch_size=BATCH, seed=seed)
        return np.stack([np.asarray(b.agents.position) for b in ds])

    np.testing.assert_array_equal(positions(0), positions(0))


def test_different_seeds_differ(shards: list[pathlib.Path], spec: DatasetSpec) -> None:
    def ids(seed: int) -> list[str]:
        ds = build(shards, spec, batch_size=BATCH, seed=seed)
        return [
            decode_scenario_id(sid) for b in ds for sid in np.asarray(b.scenario_id)
        ]

    assert ids(0) != ids(1)


def test_eval_mode_keeps_every_example(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    """drop_remainder=False in eval: every example must be seen exactly once."""
    ds = build(shards, spec, batch_size=3, seed=0, training=False)
    seen = [decode_scenario_id(sid) for b in ds for sid in np.asarray(b.scenario_id)]
    assert sorted(seen) == sorted(f"{spec.name}-{i:04d}" for i in range(NUM_SCENARIOS))


def test_training_mode_drops_the_short_batch(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    ds = build(shards, spec, batch_size=3, seed=0, training=True)
    counts = [np.asarray(b.agents.position).shape[0] for b in ds]
    assert set(counts) == {3}
    assert len(counts) == NUM_SCENARIOS // 3


def test_eval_mode_does_not_shuffle(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    a = [
        decode_scenario_id(sid)
        for b in build(shards, spec, batch_size=BATCH, seed=0, training=False)
        for sid in np.asarray(b.scenario_id)
    ]
    b = [
        decode_scenario_id(sid)
        for b_ in build(shards, spec, batch_size=BATCH, seed=99, training=False)
        for sid in np.asarray(b_.scenario_id)
    ]
    assert a == b


def test_pipeline_rejects_bad_batch_size(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build(shards, spec, batch_size=0, seed=0)


def test_single_path_is_accepted(shards: list[pathlib.Path], spec: DatasetSpec) -> None:
    """A bare path, not just a sequence."""
    batch = next(iter(build(shards[0], spec, batch_size=1, seed=0, training=False)))
    assert np.asarray(batch.agents.position).shape[0] == 1


def test_multiprocess_prefetch_matches_inprocess(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    """Grain spawns workers, so every transform must survive pickling.

    This is the production path and the one place an unpicklable closure or a
    module-global would show up, so it asserts equality rather than merely
    running.
    """
    if spec is not AV2:
        pytest.skip("one dataset is enough to exercise the worker round trip")

    def ids(num_workers: int) -> list[str]:
        ds = build(
            shards,
            spec,
            batch_size=BATCH,
            seed=0,
            training=False,
            num_workers=num_workers,
        )
        return [
            decode_scenario_id(sid) for b in ds for sid in np.asarray(b.scenario_id)
        ]

    assert ids(2) == ids(0)


def test_unpadded_scenarios_cannot_be_batched(spec: DatasetSpec) -> None:
    """Forgetting PadAndMask is a loud failure, not silent corruption."""

    a = synthetic_scenario(spec, 3, 5)
    b = synthetic_scenario(spec, 8, 5)
    with pytest.raises(Exception):
        grain.MapDataset.source([a, b]).batch(2)[0]


# --- lean pipelines -------------------------------------------------------------


def test_drop_map_keeps_shapes_and_masks_every_polyline(
    shards: list[pathlib.Path], spec: DatasetSpec, budget: Budget
) -> None:
    batch = next(
        iter(
            build(
                shards,
                spec,
                batch_size=BATCH,
                seed=0,
                budget=budget,
                stages=Stages(keep_map=False),
            )
        )
    )
    assert np.asarray(batch.vector_map.points).shape == (
        BATCH,
        budget.max_polylines,
        spec.max_points,
        2,
    )
    assert not np.asarray(batch.polyline_mask).any()
    assert not np.asarray(batch.vector_map.valid).any()
    # Agents are untouched.
    assert np.asarray(batch.agent_mask).any()


def test_source_frame_leaves_positions_unmoved(
    shards: list[pathlib.Path], spec: DatasetSpec, budget: Budget
) -> None:
    def first(ego_frame: bool) -> np.ndarray:
        ds = build(
            shards,
            spec,
            batch_size=1,
            seed=0,
            budget=budget,
            training=False,
            stages=Stages(ego_frame=ego_frame),
        )
        return np.asarray(next(iter(ds)).agents.position)

    centred, source = first(True), first(False)
    t_ref = spec.num_history - 1
    ego_now = source[0, 0, t_ref]
    # Synthetic agent 0 is the ego: at the origin only when re-centred.
    np.testing.assert_allclose(centred[0, 0, t_ref], 0.0, atol=1e-3)
    assert np.linalg.norm(ego_now) > 1.0


def test_augment_requires_the_ego_frame(
    shards: list[pathlib.Path], spec: DatasetSpec
) -> None:
    with pytest.raises(ValueError, match="ego_frame"):
        Stages(ego_frame=False, augment=True)
    with pytest.raises(ValueError, match="ego_frame"):
        # augment=None means "when training": caught at build time instead.
        build(shards, spec, batch_size=BATCH, seed=0, stages=Stages(ego_frame=False))


def test_checks_off_in_spawned_workers(
    shards: list[pathlib.Path], spec: DatasetSpec, budget: Budget
) -> None:
    """The setting reaches Grain's spawned workers and batches still arrive."""
    ds = build(
        shards,
        spec,
        batch_size=BATCH,
        seed=0,
        budget=budget,
        shape_checks=False,
        num_workers=2,
    )
    batch = next(iter(ds))
    assert np.asarray(batch.agents.position).shape[0] == BATCH

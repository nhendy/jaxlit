"""Argoverse 2 adapter tests, against three real scenarios committed under
`data/testdata/argoverse/`.

Real fixtures rather than synthetic ones because the whole job of this module is
to be faithful to a format we do not control -- a synthetic parquet would only
test our own assumptions back at us.
"""

import json
import pathlib

import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from data.ingest.writer import write_shards
from data.pipeline import build
from data.schema import Scenario, decode_scenario_id
from data.sources.argoverse import _OBJECT_TYPE, iter_scenarios, load_scenario
from data.specs import AV2, DEFAULT_BUDGETS, AgentType, PolylineType
from data.transforms import AgentCentric, PadAndMask

TESTDATA = pathlib.Path(__file__).parent.parent / "testdata" / "argoverse"


def _scenario_dirs() -> list[pathlib.Path]:
    return sorted(p for p in TESTDATA.iterdir() if p.is_dir())


pytestmark = pytest.mark.skipif(
    not TESTDATA.exists() or not _scenario_dirs(),
    reason="AV2 fixtures missing; run scripts/fetch_samples.py --dataset argoverse",
)


@parameterized_fixture
def scenario_dir() -> dict[str, pathlib.Path]:
    """One case per committed scenario, id'd by its (truncated) UUID."""
    return {p.name[:8]: p for p in _scenario_dirs()}


@pytest.fixture
def scenario(scenario_dir: pathlib.Path) -> Scenario:
    return load_scenario(scenario_dir)


# --- agents -------------------------------------------------------------------


def test_agent_shapes_match_spec(scenario: Scenario) -> None:
    position = np.asarray(scenario.agents.position)
    assert position.ndim == 3 and position.shape[1:] == (AV2.num_steps, 2)
    assert np.asarray(scenario.agents.valid).shape == position.shape[:2]


def test_tracks_are_ragged(scenario: Scenario) -> None:
    """The parquet is long-format; densifying it is what creates `valid`.

    If this ever becomes all-True the densification has silently stopped doing
    anything.
    """
    valid = np.asarray(scenario.agents.valid)
    assert not valid.all()
    assert valid.any()
    per_agent = valid.sum(axis=1)
    assert per_agent.min() < AV2.num_steps


def test_focal_agent_is_valid_at_reference_step(scenario: Scenario) -> None:
    """AgentCentric needs this, and it is the one thing AV2 does not guarantee."""
    ego = int(np.asarray(scenario.ego_index))
    assert bool(np.asarray(scenario.agents.valid)[ego, AV2.num_history - 1])


def test_agent_types_are_in_range(scenario: Scenario) -> None:
    types = np.asarray(scenario.agents.agent_type)
    assert types.min() >= 0 and types.max() < len(AgentType)


def test_object_type_mapping_covers_the_av2_taxonomy() -> None:
    """Every documented AV2 class must map; an unmapped one silently becomes UNKNOWN."""
    av2_classes = {
        "vehicle",
        "pedestrian",
        "motorcyclist",
        "cyclist",
        "bus",
        "static",
        "background",
        "construction",
        "riderless_bicycle",
        "unknown",
    }
    assert av2_classes <= set(_OBJECT_TYPE)


def test_positions_are_finite(scenario: Scenario) -> None:
    for name in ("position", "heading", "velocity", "extent"):
        assert np.isfinite(np.asarray(getattr(scenario.agents, name))).all(), name


def test_invalid_steps_are_zero_not_garbage(scenario: Scenario) -> None:
    invalid = ~np.asarray(scenario.agents.valid)
    np.testing.assert_array_equal(np.asarray(scenario.agents.position)[invalid], 0.0)


# --- map ----------------------------------------------------------------------


def test_polylines_have_fixed_length(scenario: Scenario) -> None:
    points = np.asarray(scenario.vector_map.points)
    assert points.shape[1:] == (AV2.max_points, 2)
    assert np.asarray(scenario.vector_map.valid).all(), "resampling leaves no gaps"


def test_polyline_types_are_in_range(scenario: Scenario) -> None:
    types = np.asarray(scenario.vector_map.types)
    assert types.min() >= 0 and types.max() < len(PolylineType)


def test_lane_centerlines_are_present(scenario: Scenario) -> None:
    """Every AV2 map has lane segments; missing them means the parse broke."""
    types = np.asarray(scenario.vector_map.types)
    assert (types == int(PolylineType.LANE_CENTER)).sum() > 0


def test_unmarked_boundaries_are_skipped(scenario_dir: pathlib.Path) -> None:
    """`NONE` markings are not physical features and must not become polylines."""
    raw = json.loads(next(scenario_dir.glob("log_map_archive_*.json")).read_text())
    lanes = raw["lane_segments"].values()
    marked = sum(
        1
        for lane in lanes
        for side in ("left", "right")
        if lane.get(f"{side}_lane_mark_type", "NONE") != "NONE"
    )
    types = np.asarray(load_scenario(scenario_dir).vector_map.types)
    boundaries = np.isin(
        types,
        [
            int(PolylineType.LANE_BOUNDARY_SOLID),
            int(PolylineType.LANE_BOUNDARY_DASHED),
        ],
    ).sum()
    assert boundaries == marked


# --- identity and errors ------------------------------------------------------


def test_scenario_id_matches_directory(
    scenario: Scenario, scenario_dir: pathlib.Path
) -> None:
    assert decode_scenario_id(scenario.scenario_id) == scenario_dir.name


def test_missing_files_raise(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="AV2 scenario directory"):
        load_scenario(tmp_path)


def test_iter_scenarios_yields_all() -> None:
    assert len(list(iter_scenarios(TESTDATA))) == len(_scenario_dirs())


# --- end to end ---------------------------------------------------------------


def test_real_scenarios_survive_the_full_pipeline(tmp_path: pathlib.Path) -> None:
    """Real AV2 -> ingest -> ArrayRecord -> Grain -> batched Scenario."""
    shards = write_shards(
        iter_scenarios(TESTDATA), tmp_path, num_shards=2, prefix="av2"
    )
    batch = next(iter(build(shards, AV2, batch_size=2, seed=0, training=False)))
    assert isinstance(batch, Scenario)
    assert np.asarray(batch.agents.position).shape == (
        2,
        DEFAULT_BUDGETS["av2"].max_agents,
        AV2.num_steps,
        2,
    )
    assert np.asarray(batch.vector_map.points).shape == (
        2,
        DEFAULT_BUDGETS["av2"].max_polylines,
        AV2.max_points,
        2,
    )
    assert np.isfinite(np.asarray(batch.agents.position)).all()
    assert batch.spec is AV2


def test_agent_centric_on_real_data(scenario: Scenario) -> None:
    out = AgentCentric().map(scenario)
    ego = int(np.asarray(out.ego_index))
    np.testing.assert_allclose(
        np.asarray(out.agents.position)[ego, AV2.num_history - 1], [0.0, 0.0], atol=1e-3
    )


def test_pad_keeps_the_focal_agent_on_real_data(scenario: Scenario) -> None:
    """Real scenarios exceed max_agents, so truncation actually fires here."""
    padded = PadAndMask(DEFAULT_BUDGETS["av2"]).map(AgentCentric().map(scenario))
    num_agents = np.asarray(scenario.agents.position).shape[0]
    assert (
        np.asarray(padded.agents.position).shape[0] == DEFAULT_BUDGETS["av2"].max_agents
    )
    kept = int(np.asarray(padded.agent_mask).sum())
    assert kept == min(num_agents, DEFAULT_BUDGETS["av2"].max_agents)
    ego = int(np.asarray(padded.ego_index))
    assert bool(np.asarray(padded.agent_mask)[ego])


def test_budgets_cover_the_real_distribution() -> None:
    """The committed fixtures must fit the budgets in `specs.py`.

    Those budgets are measured, not guessed. Over 40 real val scenarios:
        agents     p50  54  p90  96  p95 114  p99 121  max 122
        polylines  p50 146  p90 276  p95 287  p99 335  max 354
    An earlier 64/256 silently truncated 32% and 12.5% of scenarios. If a future
    change to the polyline extraction inflates the count, this catches it.
    """
    for scenario_dir in _scenario_dirs():
        s = load_scenario(scenario_dir)
        num_agents = np.asarray(s.agents.position).shape[0]
        num_polylines = np.asarray(s.vector_map.points).shape[0]
        assert num_agents <= DEFAULT_BUDGETS["av2"].max_agents, (
            f"{scenario_dir.name}: {num_agents} agents"
        )
        assert num_polylines <= DEFAULT_BUDGETS["av2"].max_polylines, (
            f"{scenario_dir.name}: {num_polylines} polylines"
        )


def test_polyline_truncation_keeps_the_nearest(scenario: Scenario) -> None:
    """Positional truncation would delete whole map categories, not far geometry."""
    import attrs as _attrs

    from data.transforms import AgentCentric, PadAndMask

    tiny = _attrs.evolve(DEFAULT_BUDGETS["av2"], max_polylines=8)
    # The scenario carries its spec, so it has to be evolved too -- PadAndMask
    # refuses a scenario whose spec differs from its own.
    centred = AgentCentric().map(scenario)
    padded = PadAndMask(tiny).map(centred)

    kept = np.asarray(padded.vector_map.points)[
        : int(np.asarray(padded.polyline_mask).sum())
    ]
    all_points = np.asarray(centred.vector_map.points)
    all_valid = np.asarray(centred.vector_map.valid)

    origin = np.asarray(centred.agents.position)[
        int(np.asarray(centred.ego_index)), AV2.num_history - 1
    ]
    all_dist = np.sort(
        np.where(all_valid, np.linalg.norm(all_points - origin, axis=-1), np.inf).min(
            -1
        )
    )
    kept_dist = np.linalg.norm(kept - origin, axis=-1).min(-1)
    # Every kept polyline is among the 8 closest.
    assert kept_dist.max() <= all_dist[7] + 1e-4

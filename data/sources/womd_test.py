"""WOMD adapter tests.

Two tiers, because the licence forbids redistributing real records:

* `data/testdata/womd/fixture.tfrecord-*` is synthesised through the same proto
  and TFRecord writer the real path reads, so it exercises the identical parse
  code and runs in CI with no download.
* Anything marked `realdata` runs only when a real shard is present, fetched by
  hand per the licence. Those are the tests that would catch an assumption about
  the data that a fixture we wrote ourselves cannot.
"""

import pathlib

import numpy as np
import pytest

from data.schema import Scenario, decode_scenario_id
from data.sources._tfrecord import read_records, write_records
from data.sources._womd_proto import scenario_pb2
from data.sources.womd import (
    _DASHED_ROAD_LINES,
    _OBJECT_TYPE,
    iter_scenarios,
    iter_shard,
    to_scenario,
)
from data.specs import DEFAULT_BUDGETS, WOMD, AgentType, PolylineType
from data.transforms import AgentCentric, PadAndMask

TESTDATA = pathlib.Path(__file__).parent.parent / "testdata" / "womd"
FIXTURE = TESTDATA / "fixture.tfrecord-00000-of-00001"

# A real shard, if one has been fetched by hand. Never committed.
REAL_SHARD = next(TESTDATA.glob("*validation.tfrecord*"), None)
needs_real = pytest.mark.skipif(
    REAL_SHARD is None,
    reason="no real WOMD shard; see data/sources/womd.py for the gcloud command",
)


@pytest.fixture
def scenario() -> Scenario:
    return next(iter_shard(FIXTURE))


# --- TFRecord framing ---------------------------------------------------------


def test_round_trip(tmp_path: pathlib.Path) -> None:
    payloads = [b"", b"a", b"x" * 1000, bytes(range(256))]
    path = tmp_path / "t.tfrecord"
    assert write_records(path, iter(payloads)) == len(payloads)
    assert list(read_records(path)) == payloads


def test_limit_stops_early(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "t.tfrecord"
    write_records(path, iter([b"a", b"b", b"c"]))
    assert list(read_records(path, limit=2)) == [b"a", b"b"]


def test_truncation_yields_whole_records_only(tmp_path: pathlib.Path) -> None:
    """The property that makes a byte-range prefix of a remote shard readable."""
    path = tmp_path / "t.tfrecord"
    payloads = [b"record-%03d" % i for i in range(8)]
    write_records(path, iter(payloads))
    raw = path.read_bytes()

    for cut in (len(raw) // 3, len(raw) // 2, len(raw) - 5):
        partial = tmp_path / "cut.tfrecord"
        partial.write_bytes(raw[:cut])
        got = list(read_records(partial))
        assert got == payloads[: len(got)], "a partial record leaked through"


def test_no_tensorflow_is_imported() -> None:
    """The whole point: TF cannot share a process with array_record."""
    import sys

    list(read_records(FIXTURE, limit=1))
    assert "tensorflow" not in sys.modules


# --- adapter ------------------------------------------------------------------


def test_shapes_match_spec(scenario: Scenario) -> None:
    position = np.asarray(scenario.agents.position)
    assert position.ndim == 3 and position.shape[1:] == (WOMD.num_steps, 2)
    assert np.asarray(scenario.vector_map.points).shape[1:] == (WOMD.max_points, 2)


def test_valid_mask_is_exercised(scenario: Scenario) -> None:
    """A late-entering agent, so masking is tested rather than assumed."""
    valid = np.asarray(scenario.agents.valid)
    assert not valid.all() and valid.any()


def test_invalid_steps_are_zero(scenario: Scenario) -> None:
    invalid = ~np.asarray(scenario.agents.valid)
    np.testing.assert_array_equal(np.asarray(scenario.agents.position)[invalid], 0.0)


def test_extents_come_from_the_data(scenario: Scenario) -> None:
    """WOMD ships measured boxes; unlike AV2 these are not a per-class prior."""
    extent = np.asarray(scenario.agents.extent)
    assert (extent > 0).all()


def test_types_are_in_range(scenario: Scenario) -> None:
    assert np.asarray(scenario.agents.agent_type).max() < len(AgentType)
    assert np.asarray(scenario.vector_map.types).max() < len(PolylineType)


def test_object_type_mapping_is_total() -> None:
    """WOMD's enum is 0..4; an unmapped value would silently become UNKNOWN."""
    assert set(_OBJECT_TYPE) == {0, 1, 2, 3, 4}


def test_dashed_road_lines_are_the_broken_enum_values() -> None:
    """1/4/5 are BROKEN_*; everything else is SOLID_*. Getting this backwards
    would mislabel every lane boundary in the corpus."""
    assert _DASHED_ROAD_LINES == {1, 4, 5}


def test_scenario_id_round_trips(scenario: Scenario) -> None:
    assert decode_scenario_id(scenario.scenario_id).startswith("fixture")


def test_ego_is_the_sdc(scenario: Scenario) -> None:
    ego = int(np.asarray(scenario.ego_index))
    assert bool(np.asarray(scenario.agents.valid)[ego, WOMD.num_history - 1])


def test_wrong_step_count_is_rejected() -> None:
    """A future format revision should fail loudly, not silently mis-shape."""
    proto = scenario_pb2.Scenario()
    proto.scenario_id = "short"
    proto.timestamps_seconds.extend([0.1] * 10)
    proto.current_time_index = WOMD.num_history - 1
    with pytest.raises(ValueError, match="steps, expected"):
        to_scenario(proto)


def test_wrong_history_index_is_rejected() -> None:
    proto = scenario_pb2.Scenario()
    proto.scenario_id = "shifted"
    proto.timestamps_seconds.extend([i * 0.1 for i in range(WOMD.num_steps)])
    proto.current_time_index = 42
    with pytest.raises(ValueError, match="current_time_index"):
        to_scenario(proto)


def test_missing_shards_raise(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="no TFRecord shards"):
        list(iter_scenarios(tmp_path))


def test_iter_scenarios_reads_the_directory() -> None:
    assert len(list(iter_scenarios(TESTDATA))) >= 3


def test_survives_the_transform_chain(scenario: Scenario) -> None:
    centred = AgentCentric().map(scenario)
    ego = int(np.asarray(centred.ego_index))
    np.testing.assert_allclose(
        np.asarray(centred.agents.position)[ego, WOMD.num_history - 1],
        [0.0, 0.0],
        atol=1e-3,
    )
    budget = DEFAULT_BUDGETS["womd"]
    padded = PadAndMask(budget).map(centred)
    assert np.asarray(padded.agents.position).shape == (
        budget.max_agents,
        WOMD.num_steps,
        2,
    )


# --- real data ----------------------------------------------------------------


@needs_real
def test_real_shard_parses() -> None:
    scenarios = list(iter_shard(REAL_SHARD))
    assert len(scenarios) > 100
    for s in scenarios[:20]:
        assert np.isfinite(np.asarray(s.agents.position)).all()
        assert np.asarray(s.agents.valid).any()


@needs_real
def test_real_budgets_are_adequate() -> None:
    """Budgets are measured, not guessed. If the extraction changes shape, this
    is what notices."""
    budget = DEFAULT_BUDGETS["womd"]
    over_agents = over_map = 0
    total = 0
    for s in iter_shard(REAL_SHARD):
        total += 1
        over_agents += np.asarray(s.agents.position).shape[0] > budget.max_agents
        over_map += np.asarray(s.vector_map.points).shape[0] > budget.max_polylines
    assert over_agents / total < 0.02, f"{over_agents}/{total} exceed the agent budget"
    assert over_map / total < 0.02, f"{over_map}/{total} exceed the map budget"

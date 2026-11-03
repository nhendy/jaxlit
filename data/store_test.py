"""ScenarioStore tests.

Uses the committed AV2 fixtures ingested on the fly, so this runs in CI without
the 1.6 GB val corpus present.
"""

import pathlib

import numpy as np
import pytest

from data.ingest.writer import write_shards
from data.schema import Scenario
from data.sources.argoverse import iter_scenarios
from data.store import ScenarioStore, agent_track

TESTDATA = pathlib.Path(__file__).parent / "testdata" / "argoverse"


@pytest.fixture
def store(tmp_path: pathlib.Path) -> ScenarioStore:
    write_shards(iter_scenarios(TESTDATA), tmp_path, num_shards=2, prefix="av2")
    return ScenarioStore.open(tmp_path)


def test_len_matches_the_corpus(store: ScenarioStore) -> None:
    assert len(store) == 3


def test_indexing_returns_scenarios(store: ScenarioStore) -> None:
    assert all(isinstance(store[i], Scenario) for i in range(len(store)))


def test_negative_indices(store: ScenarioStore) -> None:
    assert store.scenario_id(-1) == store.scenario_id(len(store) - 1)


def test_out_of_range_raises(store: ScenarioStore) -> None:
    with pytest.raises(IndexError, match="out of range"):
        store[len(store)]


def test_iteration_covers_everything(store: ScenarioStore) -> None:
    assert len(list(store)) == len(store)


def test_find_by_scenario_id(store: ScenarioStore) -> None:
    target = store.scenario_id(1)
    assert store.find(target) == 1


def test_find_rejects_an_unknown_id(store: ScenarioStore) -> None:
    with pytest.raises(KeyError):
        store.find("not-a-scenario")


def test_index_is_stable_across_opens(tmp_path: pathlib.Path) -> None:
    """A visualiser deep-linking to 'scenario 2' must not drift between runs."""
    write_shards(iter_scenarios(TESTDATA), tmp_path, num_shards=2, prefix="av2")
    first = [ScenarioStore.open(tmp_path).scenario_id(i) for i in range(3)]
    second = [ScenarioStore.open(tmp_path).scenario_id(i) for i in range(3)]
    assert first == second


def test_missing_shards_raise(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="no .arrayrecord shards"):
        ScenarioStore.open(tmp_path / "nothing")


def test_agent_track_drops_invalid_steps(store: ScenarioStore) -> None:
    """Raw arrays zero invalid steps, so plotting them draws a line to the
    origin. The helper exists to keep that out of every call site."""
    scenario = store[0]
    valid = np.asarray(scenario.agents.valid)
    partial = int(np.argmin(valid.all(axis=1)))  # an agent with gaps

    xs, ys = agent_track(scenario, partial)
    assert len(xs) == int(valid[partial].sum())
    assert len(xs) < valid.shape[1], "fixture should have an agent with gaps"
    assert np.isfinite(xs).all() and np.isfinite(ys).all()

"""TimeWindow and Resample: 5 Hz / 3 s views of AV2 without re-ingest."""

import numpy as np
import pytest

from data import codec
from data.specs import AV2, TimeWindow
from data.testing import synthetic_scenario
from data.transforms import Resample

FIVE_HZ = TimeWindow(stride=2, history=15)


def test_indices_keep_the_present_and_the_full_future() -> None:
    idx = FIVE_HZ.indices(AV2)
    now = AV2.num_history - 1  # 49
    assert idx[FIVE_HZ.history - 1] == now  # the present is kept, last of history
    assert idx[0] == now - 2 * 14  # 2.8 s back at 0.2 s
    assert idx[-1] == AV2.num_steps - 1  # 6.0 s ahead
    assert np.all(np.diff(idx) == 2)


def test_apply_describes_the_window() -> None:
    spec = FIVE_HZ.apply(AV2)
    assert spec.dt == pytest.approx(0.2)
    assert spec.num_history == 15
    assert spec.num_future == 30
    assert spec.future_seconds == pytest.approx(AV2.future_seconds)
    assert spec.name != AV2.name


def test_window_longer_than_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="observed steps"):
        TimeWindow(stride=2, history=26).indices(AV2)  # needs 51 of 50


def test_resample_keeps_exactly_the_windowed_steps() -> None:
    scenario = synthetic_scenario(AV2, num_agents=5, num_polylines=3)
    out = Resample(window=FIVE_HZ).map(scenario)
    idx = FIVE_HZ.indices(AV2)
    np.testing.assert_array_equal(
        np.asarray(out.agents.position), np.asarray(scenario.agents.position)[:, idx]
    )
    np.testing.assert_array_equal(
        np.asarray(out.agents.valid), np.asarray(scenario.agents.valid)[:, idx]
    )
    assert out.spec == FIVE_HZ.apply(AV2)
    # Per-agent constants and the map are untouched.
    np.testing.assert_array_equal(out.agents.extent, scenario.agents.extent)
    assert out.vector_map is scenario.vector_map


def test_resampled_scenarios_cannot_be_encoded_as_the_full_dataset() -> None:
    out = Resample(window=FIVE_HZ).map(synthetic_scenario(AV2))
    with pytest.raises((KeyError, ValueError)):
        codec.decode(codec.encode(out))

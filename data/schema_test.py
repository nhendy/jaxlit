"""Schema and codec tests.

The centrepiece is `test_each_declared_dim_is_load_bearing`, which derives one
case per (field, axis) from the spec dicts themselves. Adding a field to
AGENT_SPEC or MAP_SPEC automatically adds its cases, so the suite cannot quietly
fall behind the schema.
"""

import jax
import numpy as np
import pytest
from flax import serialization

from commmon.fixtures import parameterized_fixture
from data.codec import decode, encode
from data.schema import (
    AGENT_SPEC,
    MAP_SPEC,
    SCENARIO_ID_BYTES,
    Scenario,
    decode_scenario_id,
    encode_scenario_id,
)
from data.specs import ALL_SPECS, AV2, DatasetSpec
from data.testing import synthetic_scenario

NUM_AGENTS = 7
NUM_POLYLINES = 13


@parameterized_fixture
def spec() -> dict[str, DatasetSpec]:
    """Fans every test below out over the three datasets."""
    return dict(ALL_SPECS)


@pytest.fixture
def scenario(spec: DatasetSpec) -> Scenario:
    """Ragged (pre-pad) scenario. Inherits the per-dataset fan-out."""
    return synthetic_scenario(spec, NUM_AGENTS, NUM_POLYLINES, np.random.default_rng(0))


@parameterized_fixture
def perturbation() -> dict[str, tuple[str, str, int]]:
    """Every (field, axis) pair a spec claims to constrain.

    The leading `*n` is dropped: it matches zero dims on an unbatched record, so
    there is no axis to perturb.
    """
    return {
        f"{obj}.{field}.axis{axis}": (obj, field, axis)
        for obj, specs in (("agents", AGENT_SPEC), ("vector_map", MAP_SPEC))
        for field, spec_str in specs.items()
        for axis in range(len(spec_str.split()) - 1)
    }


def test_each_declared_dim_is_load_bearing(
    scenario: Scenario, perturbation: tuple[str, str, int]
) -> None:
    """Grow one axis by one; construction must reject it.

    If a spec is wrong or missing, exactly one of these stops failing.
    """
    obj_name, field, axis = perturbation
    sub = getattr(scenario, obj_name)
    original = np.asarray(getattr(sub, field))
    grown = np.concatenate([original, np.take(original, [0], axis=axis)], axis=axis)
    with pytest.raises(ValueError):
        scenario.replace(**{obj_name: sub.replace(**{field: grown})})


def test_outer_anchor_ties_inner_to_outer(scenario: Scenario) -> None:
    """agent_mask is only checked against `a` because of the dotted anchor."""
    bad = np.ones((NUM_AGENTS + 2,), dtype=bool)
    with pytest.raises(ValueError, match="agent_mask"):
        scenario.replace(agent_mask=bad)


def test_polyline_mask_tied_to_map(scenario: Scenario) -> None:
    bad = np.ones((NUM_POLYLINES + 2,), dtype=bool)
    with pytest.raises(ValueError, match="polyline_mask"):
        scenario.replace(polyline_mask=bad)


def test_star_n_accepts_unbatched_and_batched(scenario: Scenario) -> None:
    """One class serves both stages; `*n` binds () or (b,)."""
    batched = jax.tree.map(lambda x: np.asarray(x)[None], scenario)
    assert np.asarray(batched.agents.position).shape[0] == 1
    assert np.asarray(scenario.agents.position).ndim == 3


def test_star_n_couples_batch_dims_across_fields(scenario: Scenario) -> None:
    """A named variadic is what makes this an error; `...` would accept it."""
    agents = scenario.agents
    with pytest.raises(ValueError):
        agents.replace(
            position=np.asarray(agents.position)[None],
            heading=np.asarray(agents.heading),
        )


def test_is_registered_pytree(scenario: Scenario) -> None:
    """Guards the silent-object-array failure if @struct.dataclass is dropped."""
    leaves = jax.tree_util.tree_leaves(scenario)
    assert len(leaves) == len(AGENT_SPEC) + len(MAP_SPEC) + 4
    assert all(not isinstance(leaf, Scenario) for leaf in leaves)


def test_spec_is_static_not_a_leaf(scenario: Scenario, spec: DatasetSpec) -> None:
    flat, treedef = jax.tree_util.tree_flatten(scenario)
    assert jax.tree_util.tree_unflatten(treedef, flat).spec is spec


def test_every_leaf_is_device_transferable(scenario: Scenario) -> None:
    """A `str` leaf would poison device_put for the whole tree."""
    on_device = jax.device_put(scenario)
    assert isinstance(on_device.agents.position, jax.Array)
    assert isinstance(on_device.scenario_id, jax.Array)


def test_scenario_id_roundtrip() -> None:
    packed = encode_scenario_id("abc-123")
    assert packed.shape == (SCENARIO_ID_BYTES,)
    assert packed.dtype == np.uint8
    assert decode_scenario_id(packed) == "abc-123"


def test_scenario_id_too_long_is_rejected() -> None:
    with pytest.raises(ValueError, match="max"):
        encode_scenario_id("x" * (SCENARIO_ID_BYTES + 1))


def test_codec_roundtrip_preserves_dtype(scenario: Scenario) -> None:
    restored = decode(encode(scenario))
    assert restored.spec is scenario.spec
    for obj, fields in (("agents", AGENT_SPEC), ("vector_map", MAP_SPEC)):
        for field in fields:
            want = np.asarray(getattr(getattr(scenario, obj), field))
            got = np.asarray(getattr(getattr(restored, obj), field))
            assert got.dtype == want.dtype, f"{obj}.{field} dtype"
            np.testing.assert_array_equal(got, want, err_msg=f"{obj}.{field}")


def test_codec_roundtrip_after_device_put(scenario: Scenario) -> None:
    """Encoding must not depend on whether the scenario has been to a device."""
    assert encode(jax.device_put(scenario)) == encode(scenario)


def test_codec_rejects_unknown_spec(scenario: Scenario) -> None:

    raw = serialization.msgpack_restore(encode(scenario))
    raw["spec"] = "not-a-dataset"
    with pytest.raises(ValueError, match="unknown dataset spec"):
        decode(serialization.msgpack_serialize(raw))


def test_synthetic_has_invalid_steps(scenario: Scenario) -> None:
    """The fixture must exercise masking, not pristine data."""
    assert not np.asarray(scenario.agents.valid).all()


def test_horizons_match_spec(scenario: Scenario, spec: DatasetSpec) -> None:
    assert np.asarray(scenario.agents.position).shape[1] == spec.num_steps
    assert spec.num_history + spec.num_future == spec.num_steps


def test_dataset_specs_are_hashable() -> None:
    """Required for `struct.field(pytree_node=False)` to sit in the treedef."""
    assert len({hash(s) for s in ALL_SPECS.values()}) == len(ALL_SPECS)
    assert AV2 in {AV2}

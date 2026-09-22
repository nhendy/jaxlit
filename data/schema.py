"""The canonical scenario schema. One shape, three datasets.

Shape checking is eincheck's job, not jaxtyping's: fields are annotated
`jt.ArrayLike` (bare -- arrays are numpy before `device_put` and `jax.Array`
after) and every dimension lives in a `check_data` spec. Keeping shapes in one
place means the annotation and the spec can never drift apart.

Two conventions run through every spec:

* Dimension names are lower case: `a` agents, `t` timesteps, `p` polylines,
  `s` points per polyline.
* Every spec leads with `*n`, a *named* variadic capturing zero or more batch
  dims. That is what lets one class serve the per-example stage (`n=()`), the
  post-`.batch()` stage (`n=(b,)`), and any `vmap`/`scan` that adds or strips a
  leading axis. The named form couples those dims across fields; anonymous
  `...` would not.

Classes are `flax.struct.dataclass`, which registers them as JAX pytrees. That
is not optional: grain's `.batch()` stacks with `jax.tree_util.tree_map`, and an
unregistered class is a *leaf*, so batching one silently yields an object array
instead of raising. Registration also means `tree_unflatten` goes through
`__init__`, so eincheck re-validates batched records for free.
"""

import eincheck
import jaxtyping as jt
import numpy as np
from flax import struct

from data.specs import DatasetSpec

SCENARIO_ID_BYTES = 64
"""Scenario ids are fixed-width ascii in a uint8 array, never `str`.

64 rather than 32 because AV2 ids are hyphenated UUIDs -- 36 characters, e.g.
`00010486-9a07-48ae-b493-cf4545855937`. nuScenes tokens are 32 hex characters
and WOMD ids are shorter still, so 64 clears all three with room to spare.

A `str` leaf makes `jax.device_put(scenario)` fail for the *whole* tree with
`TypeError: Dtype <U9 is not a valid JAX array type`, so every leaf stays
numeric.
"""

# Exported so the test suite can derive one perturbation case per (field, axis)
# and stay honest as the schema grows.
AGENT_SPEC = {
    "position": "*n a t 2",
    "heading": "*n a t",
    "velocity": "*n a t 2",
    "extent": "*n a 2",
    "agent_type": "*n a",
    "valid": "*n a t",
}

MAP_SPEC = {
    "points": "*n p s 2",
    "types": "*n p",
    "valid": "*n p s",
}


@eincheck.check_data(AGENT_SPEC)
@struct.dataclass
class AgentTracks:
    """Trajectories for every agent in a scenario.

    `a` is ragged before `PadAndMask` and equal to `budget.max_agents` after.
    """

    position: jt.ArrayLike
    """(x, y) in the scenario frame, metres."""
    heading: jt.ArrayLike
    """Yaw in radians."""
    velocity: jt.ArrayLike
    """(vx, vy) in metres/second."""
    extent: jt.ArrayLike
    """(length, width) in metres."""
    agent_type: jt.ArrayLike
    """`data.specs.AgentType`."""
    valid: jt.ArrayLike
    """Bool. False where the agent is unobserved at that step *or* is padding.

    Not optional: all three datasets have agents entering and leaving mid-window,
    and a NaN that reaches `jax.jit` becomes a silent NaN gradient.
    """


@eincheck.check_data(MAP_SPEC)
@struct.dataclass
class VectorMap:
    """Vector map as fixed-length polylines."""

    points: jt.ArrayLike
    """(x, y) per point, same frame as `AgentTracks.position`."""
    types: jt.ArrayLike
    """`data.specs.PolylineType`."""
    valid: jt.ArrayLike
    """Bool. False where the point is padding."""


@eincheck.check_data(
    {
        # The leaf classes already pin a/t and p/s inside their own scope; what
        # they cannot see is each other. These two anchors pull those names into
        # one shared scope so the outer's own fields are checked against them --
        # five lines instead of restating all nine inner specs.
        "agents.position": "*n a t 2",
        "vector_map.points": "*n p s 2",
        "agent_mask": "*n a",
        "polyline_mask": "*n p",
        "ego_index": "*n",
        "scenario_id": f"*n {SCENARIO_ID_BYTES}",
    }
)
@struct.dataclass
class Scenario:
    """One scenario: agents, map, and which agent is the ego vehicle."""

    agents: AgentTracks
    vector_map: VectorMap
    agent_mask: jt.ArrayLike
    """Bool, per agent row. False for padding rows."""
    polyline_mask: jt.ArrayLike
    """Bool, per polyline row. False for padding rows."""
    ego_index: jt.ArrayLike
    """int32 index into `a`. The SDC / ego vehicle."""
    scenario_id: jt.ArrayLike
    """uint8 ascii, NUL-padded to SCENARIO_ID_BYTES."""
    spec: DatasetSpec = struct.field(pytree_node=False)
    """Static: rides in the treedef, not the leaves.

    Consequence: scenarios from different datasets have different treedefs, so
    grain refuses to batch them together. That is the intended guardrail.
    """


def encode_scenario_id(scenario_id: str) -> np.ndarray:
    """Pack an id into fixed-width ascii so every leaf of the tree is numeric."""
    raw = scenario_id.encode("ascii")
    if len(raw) > SCENARIO_ID_BYTES:
        raise ValueError(
            f"scenario_id {scenario_id!r} is {len(raw)} bytes, max {SCENARIO_ID_BYTES}"
        )
    out = np.zeros(SCENARIO_ID_BYTES, dtype=np.uint8)
    out[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def decode_scenario_id(packed: jt.ArrayLike) -> str:
    """Inverse of `encode_scenario_id`. Accepts numpy or jax arrays."""
    return bytes(np.asarray(packed, dtype=np.uint8)).rstrip(b"\x00").decode("ascii")

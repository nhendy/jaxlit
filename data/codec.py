"""`Scenario` <-> bytes, for the ArrayRecord shards the ingest stage writes.

Uses `flax.serialization`'s msgpack, which round-trips nested dicts of ndarrays
preserving shape and dtype. flax is already a dependency, so this adds nothing
new -- protobuf or a hand-rolled npz wrapper would both be more machinery for
the same result.

`spec` is not serialized as an object: it is a static pytree field and a closed
set, so only its name goes on the wire and `decode` looks it up. That keeps the
payload numeric and makes an unknown dataset a loud failure at decode time.
"""

import numpy as np
from flax import serialization

from data.schema import AGENT_SPEC, MAP_SPEC, AgentTracks, Scenario, VectorMap
from data.specs import ALL_SPECS

_OUTER_FIELDS = ("agent_mask", "polyline_mask", "ego_index", "scenario_id")


def encode(scenario: Scenario) -> bytes:
    """Serialize a `Scenario` to msgpack bytes.

    Arrays are forced through `np.asarray` so a scenario that has been through
    `device_put` encodes identically to one that has not.
    """
    return serialization.msgpack_serialize(
        {
            "spec": scenario.spec.name,
            "agents": {
                name: np.asarray(getattr(scenario.agents, name)) for name in AGENT_SPEC
            },
            "vector_map": {
                name: np.asarray(getattr(scenario.vector_map, name))
                for name in MAP_SPEC
            },
            **{name: np.asarray(getattr(scenario, name)) for name in _OUTER_FIELDS},
        }
    )


def decode(payload: bytes) -> Scenario:
    """Inverse of `encode`.

    Construction re-runs every eincheck spec, so a corrupt or truncated record
    fails here rather than somewhere downstream in a training step.
    """
    raw = serialization.msgpack_restore(payload)
    spec_name = raw["spec"]
    if spec_name not in ALL_SPECS:
        raise ValueError(
            f"unknown dataset spec {spec_name!r}; expected one of {sorted(ALL_SPECS)}"
        )
    return Scenario(
        agents=AgentTracks(**raw["agents"]),
        vector_map=VectorMap(**raw["vector_map"]),
        spec=ALL_SPECS[spec_name],
        **{name: raw[name] for name in _OUTER_FIELDS},
    )

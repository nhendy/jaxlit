"""Synthetic `Scenario` construction, for tests and for exercising the pipeline
before any raw dataset is on disk.

Deliberately not in a test file: step 2 of the build (the Grain pipeline) runs
against synthetic scenarios so the pipeline half can be proven independently of
the three dataset adapters.
"""

import numpy as np

from data.schema import AgentTracks, Scenario, VectorMap, encode_scenario_id
from data.specs import AgentType, DatasetSpec, PolylineType


def synthetic_agents(
    spec: DatasetSpec, num_agents: int, rng: np.random.Generator
) -> AgentTracks:
    """Agents on smooth constant-velocity paths, with a realistic validity gap.

    One agent is left partially invalid on purpose: every dataset has agents
    entering or leaving mid-window, and a transform that ignores `valid` should
    fail against this rather than against pristine data.
    """
    t = spec.num_steps
    start = rng.uniform(-50.0, 50.0, size=(num_agents, 1, 2))
    vel = rng.uniform(-15.0, 15.0, size=(num_agents, 1, 2))
    steps = np.arange(t, dtype=np.float32).reshape(1, t, 1)

    position = (start + vel * steps * spec.dt).astype(np.float32)
    velocity = np.broadcast_to(vel, (num_agents, t, 2)).astype(np.float32)
    heading = np.arctan2(velocity[..., 1], velocity[..., 0]).astype(np.float32)

    valid = np.ones((num_agents, t), dtype=bool)
    if num_agents > 1:
        valid[-1, : t // 2] = False  # a late entrant

    return AgentTracks(
        position=position,
        heading=heading,
        velocity=velocity,
        extent=rng.uniform(1.0, 5.0, size=(num_agents, 2)).astype(np.float32),
        agent_type=rng.integers(0, len(AgentType), size=(num_agents,), dtype=np.int32),
        valid=valid,
    )


def synthetic_map(
    spec: DatasetSpec, num_polylines: int, rng: np.random.Generator
) -> VectorMap:
    """Straight polylines at random positions and orientations."""
    s = spec.max_points
    origin = rng.uniform(-80.0, 80.0, size=(num_polylines, 1, 2))
    angle = rng.uniform(-np.pi, np.pi, size=(num_polylines, 1))
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    offsets = np.arange(s, dtype=np.float32).reshape(1, s, 1) * 2.0

    return VectorMap(
        points=(origin + direction * offsets).astype(np.float32),
        types=rng.integers(0, len(PolylineType), size=(num_polylines,), dtype=np.int32),
        valid=np.ones((num_polylines, s), dtype=bool),
    )


def synthetic_scenario(
    spec: DatasetSpec,
    num_agents: int = 7,
    num_polylines: int = 13,
    rng: np.random.Generator | None = None,
    scenario_id: str | None = None,
) -> Scenario:
    """A ragged (pre-`PadAndMask`) scenario with `spec`'s rate and horizon."""
    rng = np.random.default_rng(0) if rng is None else rng
    return Scenario(
        agents=synthetic_agents(spec, num_agents, rng),
        vector_map=synthetic_map(spec, num_polylines, rng),
        agent_mask=np.ones((num_agents,), dtype=bool),
        polyline_mask=np.ones((num_polylines,), dtype=bool),
        ego_index=np.int32(0),
        scenario_id=encode_scenario_id(
            scenario_id if scenario_id is not None else f"{spec.name}-synthetic"
        ),
        spec=spec,
    )

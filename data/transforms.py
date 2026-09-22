"""Per-example transforms for the Grain pipeline.

All geometry and array shaping comes from `commmon.geometry` and
`commmon.arrays` -- this module contains no math of its own. It is only
responsible for *which* transform applies to *which* field, and for the schema
invariants that follow.

Ordering matters and is not arbitrary:

    DecodeScenario -> AgentCentric -> RandomRotate -> PadAndMask -> .batch()

`PadAndMask` must be last. `.batch()` stacks, and stacking requires shapes to
already match; it is also the step that turns the ragged `a`/`p` into the static
shapes JAX needs. Augmentation uses `grain.transforms.RandomMap` rather than
`Map` so Grain owns the per-record RNG and the whole pipeline stays reproducible
from a single seed.

Every transform returns a new `Scenario` via `.replace()`, which re-runs the
eincheck specs -- so each stage is shape-validated on the way out with no extra
code.
"""

import attrs
import eincheck
import grain
import numpy as np

from commmon import arrays, geometry
from data import codec
from data.schema import AgentTracks, Scenario, VectorMap
from data.specs import Budget


@attrs.frozen
class DecodeScenario(grain.transforms.Map):
    """ArrayRecord bytes -> `Scenario`."""

    def map(self, element: bytes) -> Scenario:
        return codec.decode(element)


def _transform(
    scenario: Scenario, translation: np.ndarray, angle: np.ndarray
) -> Scenario:
    """Apply one SE(2) to every spatial field, agents and map alike.

    Positions and map points are *points* (translated and rotated); velocities
    are *directions* (rotated only). `commmon.geometry` names that distinction
    explicitly because getting it wrong yields plausible-looking output.
    """
    agents, vmap = scenario.agents, scenario.vector_map
    agent_valid = np.asarray(agents.valid)
    map_valid = np.asarray(vmap.valid)

    return scenario.replace(
        agents=agents.replace(
            position=arrays.zero_invalid_points(
                geometry.transform_se2(agents.position, translation, angle),
                agent_valid,
            ),
            velocity=arrays.zero_invalid_points(
                geometry.rotate_direction(agents.velocity, angle), agent_valid
            ),
            heading=arrays.zero_invalid(
                geometry.wrap_angle(np.asarray(agents.heading) + angle), agent_valid
            ),
        ),
        vector_map=vmap.replace(
            points=arrays.zero_invalid_points(
                geometry.transform_se2(vmap.points, translation, angle), map_valid
            ),
        ),
    )


@attrs.frozen
class AgentCentric(grain.transforms.Map):
    """Re-express a scenario in the ego's frame at the last observed step.

    That step -- `num_history - 1` -- is the last one a model is allowed to see,
    so it is the only frame choice that does not leak future information into
    the inputs.
    """

    def map(self, scenario: Scenario) -> Scenario:
        ego = int(np.asarray(scenario.ego_index))
        t_ref = scenario.spec.num_history - 1

        if not bool(np.asarray(scenario.agents.valid)[ego, t_ref]):
            raise ValueError(
                f"ego agent {ego} is invalid at the reference step {t_ref}; "
                "cannot build an agent-centric frame"
            )

        translation = np.asarray(scenario.agents.position)[ego, t_ref]
        # A 0-d array, not a float: commmon.geometry takes array angles so it
        # can shape-check them and dispatch numpy/jax on them.
        angle = -np.asarray(scenario.agents.heading)[ego, t_ref]
        return _transform(scenario, translation, angle)


@attrs.frozen
class RandomRotate(grain.transforms.RandomMap):
    """Rotate about the origin by a uniform angle.

    Only meaningful after `AgentCentric`, where the origin is the ego.
    """

    max_angle: float = np.pi

    def random_map(self, scenario: Scenario, rng: np.random.Generator) -> Scenario:
        angle = np.float32(rng.uniform(-self.max_angle, self.max_angle))
        return _transform(scenario, np.zeros(2, dtype=np.float32), angle)


def _select_agents(scenario: Scenario, max_agents: int) -> np.ndarray:
    """Indices of the agents to keep, nearest-to-ego first.

    When a scenario has more agents than the budget, dropping the tail would be
    arbitrary and could drop the ego itself. Ranking by distance to the ego at
    the reference step keeps the ego (distance -inf) and the agents that
    actually matter for its prediction.
    """
    agents = scenario.agents
    num_agents = np.asarray(agents.position).shape[0]
    if num_agents <= max_agents:
        return np.arange(num_agents)

    t_ref = scenario.spec.num_history - 1
    ego = int(np.asarray(scenario.ego_index))
    position = np.asarray(agents.position)[:, t_ref]
    valid = np.asarray(agents.valid)[:, t_ref]

    distance = np.asarray(geometry.distance_to(position, position[ego]))
    distance = np.where(valid, distance, np.inf)  # invalid agents go last
    distance[ego] = -np.inf
    return np.argsort(distance, kind="stable")[:max_agents]


def _select_polylines(scenario: Scenario, max_polylines: int) -> np.ndarray:
    """Indices of the polylines to keep, nearest-to-ego first.

    Taking the first N instead would be far worse than arbitrary: the source
    adapters emit polylines grouped by kind (all lane centerlines, then
    boundaries, then crosswalks, then drivable areas), so a positional cut
    systematically deletes whole *categories* of map -- crosswalks first --
    rather than distant geometry. Ranking by distance to the ego drops what is
    actually irrelevant.
    """
    vmap = scenario.vector_map
    num_polylines = np.asarray(vmap.points).shape[0]
    if num_polylines <= max_polylines:
        return np.arange(num_polylines)

    t_ref = scenario.spec.num_history - 1
    ego = int(np.asarray(scenario.ego_index))
    origin = np.asarray(scenario.agents.position)[ego, t_ref]

    points = np.asarray(vmap.points)
    valid = np.asarray(vmap.valid)
    # Closest approach of each polyline to the ego; invalid points are pushed out
    # of contention rather than dragging the minimum down.
    per_point = np.linalg.norm(points - origin, axis=-1)
    distance = np.where(valid, per_point, np.inf).min(axis=-1)
    return np.argsort(distance, kind="stable")[:max_polylines]


@attrs.frozen
class PadAndMask(grain.transforms.Map):
    """Ragged `a`/`p` -> the fixed sizes JAX needs.

    Takes a `Budget`, not a `DatasetSpec`: how many agents and polylines to keep
    is a model decision, and nothing about it is stored with the scenario. The
    horizon and polyline resolution it needs come from `scenario.spec`, which is
    where dataset facts live. Sweeping a budget therefore needs no re-ingest and
    no surgery on stored data.

    Must be the last per-example transform: `.batch()` stacks, so shapes have to
    match beforehand. Padding rows carry `valid=False` *and* a False entry in the
    corresponding mask, so a consumer can ignore them by either route.
    """

    budget: Budget

    def map(self, scenario: Scenario) -> Scenario:
        budget, spec = self.budget, scenario.spec
        agents, vmap = scenario.agents, scenario.vector_map

        num_agents = np.asarray(agents.position).shape[0]
        num_polylines = np.asarray(vmap.points).shape[0]
        if budget.strict and (
            num_agents > budget.max_agents or num_polylines > budget.max_polylines
        ):
            raise ValueError(
                f"budget binds on this scenario: {num_agents} agents (max "
                f"{budget.max_agents}), {num_polylines} polylines (max "
                f"{budget.max_polylines}). Raise the budget or unset strict."
            )

        keep = _select_agents(scenario, budget.max_agents)
        keep_map = _select_polylines(scenario, budget.max_polylines)

        # The ego moves when agents are reordered; carry its index along.
        ego = int(np.asarray(scenario.ego_index))
        (new_ego,) = np.nonzero(keep == ego)

        padded_agents = AgentTracks(
            **{
                name: arrays.pad_to(
                    np.asarray(getattr(agents, name))[keep], budget.max_agents
                )
                for name in (
                    "position",
                    "heading",
                    "velocity",
                    "extent",
                    "agent_type",
                    "valid",
                )
            }
        )
        padded_map = VectorMap(
            **{
                name: arrays.pad_to(
                    np.asarray(getattr(vmap, name))[keep_map], budget.max_polylines
                )
                for name in ("points", "types", "valid")
            }
        )

        out = scenario.replace(
            agents=padded_agents,
            vector_map=padded_map,
            agent_mask=arrays.prefix_mask(len(keep), budget.max_agents),
            polyline_mask=arrays.prefix_mask(len(keep_map), budget.max_polylines),
            ego_index=np.int32(new_ego[0]),
        )

        # The one invariant check_data cannot state: `a` and `p` are literals
        # now. eincheck parses specs at decoration time, so this is asserted
        # here, against the budget rather than re-derived from the data.
        eincheck.check_shapes(
            agents=(out.agents, "$"), a=budget.max_agents, t=spec.num_steps
        )
        eincheck.check_shapes(
            vector_map=(out.vector_map, "$"),
            p=budget.max_polylines,
            s=spec.max_points,
        )
        return out

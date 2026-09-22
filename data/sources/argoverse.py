"""Argoverse 2 Motion Forecasting -> canonical `Scenario`.

Reads the raw layout directly with pyarrow and stdlib json rather than going
through the `av2` package: `av2` pulls in `torch`, which has no business in a
JAX project, and the formats here are simple enough that the SDK buys nothing.

One scenario directory holds exactly two files::

    <scenario_id>/
        scenario_<scenario_id>.parquet      agent tracks, long format
        log_map_archive_<scenario_id>.json  vector map

The parquet is *long* -- one row per (track, timestep) that actually exists, so
110 timesteps x N tracks with gaps. Densifying it is what produces the `valid`
mask; tracks in a real scenario run anywhere from 10 to 110 steps.
"""

import json
import pathlib
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq

from commmon import geometry
from data.schema import (
    AgentTracks,
    Scenario,
    VectorMap,
    encode_scenario_id,
)
from data.specs import AV2, AgentType, DatasetSpec, PolylineType

# AV2's 10-way taxonomy onto ours. `construction` (cones, barriers) and
# `riderless_bicycle` are mapped to STATIC rather than to a vehicle class: they
# do not move, and their behaviour is what a downstream model keys off.
_OBJECT_TYPE = {
    "vehicle": AgentType.VEHICLE,
    "pedestrian": AgentType.PEDESTRIAN,
    "motorcyclist": AgentType.MOTORCYCLIST,
    "cyclist": AgentType.CYCLIST,
    "bus": AgentType.BUS,
    "static": AgentType.STATIC,
    "construction": AgentType.STATIC,
    "riderless_bicycle": AgentType.STATIC,
    "background": AgentType.UNKNOWN,
    "unknown": AgentType.UNKNOWN,
}

# Lane-marking strings are open-ended (DASHED_WHITE, SOLID_YELLOW,
# DOUBLE_SOLID_WHITE, ...), so classify on the substring rather than enumerate.
_MARK_NONE = "NONE"


def _xy(vertices: list[dict]) -> np.ndarray:
    """(x, y) from AV2's list-of-dicts vertices, dropping z."""
    return np.array([[v["x"], v["y"]] for v in vertices], dtype=np.float64)


def _polylines(map_json: dict, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the map into (points, types) arrays of fixed-length polylines."""
    points: list[np.ndarray] = []
    types: list[int] = []

    def add(vertices: list[dict], kind: PolylineType) -> None:
        if len(vertices) < 1:
            return
        points.append(geometry.resample_polyline(_xy(vertices), spec.max_points))
        types.append(int(kind))

    for lane in map_json.get("lane_segments", {}).values():
        add(lane["centerline"], PolylineType.LANE_CENTER)
        for side in ("left", "right"):
            mark = lane.get(f"{side}_lane_mark_type", _MARK_NONE)
            if mark == _MARK_NONE:
                continue  # no physical marking to encode
            kind = (
                PolylineType.LANE_BOUNDARY_DASHED
                if "DASHED" in mark
                else PolylineType.LANE_BOUNDARY_SOLID
            )
            add(lane[f"{side}_lane_boundary"], kind)

    for crossing in map_json.get("pedestrian_crossings", {}).values():
        add(crossing["edge1"], PolylineType.CROSSWALK)
        add(crossing["edge2"], PolylineType.CROSSWALK)

    for area in map_json.get("drivable_areas", {}).values():
        add(area["area_boundary"], PolylineType.ROAD_EDGE)

    if not points:
        return (
            np.zeros((0, spec.max_points, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.stack(points).astype(np.float32),
        np.array(types, dtype=np.int32),
    )


def _agents(table_columns: dict, spec: DatasetSpec) -> tuple[AgentTracks, int]:
    """Densify the long-format parquet into (a, t) arrays plus a valid mask."""
    track_ids = np.asarray(table_columns["track_id"])
    timesteps = np.asarray(table_columns["timestep"], dtype=np.int64)

    unique_ids, row_to_agent = np.unique(track_ids, return_inverse=True)
    num_agents, num_steps = len(unique_ids), spec.num_steps

    if timesteps.max(initial=-1) >= num_steps:
        raise ValueError(
            f"timestep {timesteps.max()} exceeds {spec.name} num_steps={num_steps}"
        )

    position = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    velocity = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    heading = np.zeros((num_agents, num_steps), dtype=np.float32)
    valid = np.zeros((num_agents, num_steps), dtype=bool)

    position[row_to_agent, timesteps, 0] = table_columns["position_x"]
    position[row_to_agent, timesteps, 1] = table_columns["position_y"]
    velocity[row_to_agent, timesteps, 0] = table_columns["velocity_x"]
    velocity[row_to_agent, timesteps, 1] = table_columns["velocity_y"]
    heading[row_to_agent, timesteps] = table_columns["heading"]
    valid[row_to_agent, timesteps] = True

    # object_type is constant per track, so the last write per agent wins and is
    # the same as the first.
    agent_type = np.zeros((num_agents,), dtype=np.int32)
    raw_types = table_columns["object_type"]
    for row, agent in enumerate(row_to_agent):
        agent_type[agent] = int(_OBJECT_TYPE.get(raw_types[row], AgentType.UNKNOWN))

    # AV2 ships no box dimensions in the forecasting split, so extents are a
    # per-class prior rather than a measurement. Flagged here because a model
    # that leans on extent will be learning this table, not the data.
    extent = _EXTENT_PRIOR[agent_type]

    focal = str(table_columns["focal_track_id"][0])
    (focal_index,) = np.nonzero(unique_ids.astype(str) == focal)
    if len(focal_index) == 0:
        raise ValueError(f"focal_track_id {focal!r} is absent from the tracks")

    return (
        AgentTracks(
            position=position,
            heading=heading.astype(np.float32),
            velocity=velocity,
            extent=extent,
            agent_type=agent_type,
            valid=valid,
        ),
        int(focal_index[0]),
    )


# (length, width) in metres, indexed by AgentType.
_EXTENT_PRIOR = np.array(
    [
        [1.0, 1.0],  # UNKNOWN
        [4.5, 2.0],  # VEHICLE
        [0.7, 0.7],  # PEDESTRIAN
        [1.8, 0.7],  # CYCLIST
        [2.2, 0.9],  # MOTORCYCLIST
        [12.0, 2.6],  # BUS
        [1.0, 1.0],  # STATIC
    ],
    dtype=np.float32,
)


def load_scenario(scenario_dir: pathlib.Path, spec: DatasetSpec = AV2) -> Scenario:
    """Read one AV2 scenario directory into a canonical `Scenario`."""
    parquets = sorted(scenario_dir.glob("scenario_*.parquet"))
    maps = sorted(scenario_dir.glob("log_map_archive_*.json"))
    if not parquets or not maps:
        raise FileNotFoundError(
            f"{scenario_dir} is not an AV2 scenario directory "
            "(need scenario_*.parquet and log_map_archive_*.json)"
        )

    columns = pq.read_table(parquets[0]).to_pydict()
    agents, focal_index = _agents(columns, spec)
    points, types = _polylines(json.loads(maps[0].read_text()), spec)

    num_agents = np.asarray(agents.position).shape[0]
    return Scenario(
        agents=agents,
        vector_map=VectorMap(
            points=points,
            types=types,
            valid=np.ones(points.shape[:2], dtype=bool),
        ),
        agent_mask=np.ones((num_agents,), dtype=bool),
        polyline_mask=np.ones((len(types),), dtype=bool),
        # AV2's focal track, not the sensor vehicle: it is the agent the
        # scenario exists to forecast, so it is the right origin for the
        # agent-centric frame. The "AV" track is present but is just another
        # agent here.
        ego_index=np.int32(focal_index),
        scenario_id=encode_scenario_id(str(columns["scenario_id"][0])),
        spec=spec,
    )


def iter_scenarios(root: pathlib.Path, spec: DatasetSpec = AV2) -> Iterator[Scenario]:
    """Yield every scenario under an AV2 split directory, in sorted order."""
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield load_scenario(scenario_dir, spec)

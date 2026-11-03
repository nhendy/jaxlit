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
from typing import Any

import eincheck
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


def _polylines(
    map_json: dict[str, Any], spec: DatasetSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the map into (points, types) arrays of fixed-length polylines."""
    points: list[np.ndarray] = []
    types: list[int] = []

    def add(vertices: list[dict] | np.ndarray, kind: PolylineType) -> None:
        if len(vertices) < 1:
            return
        xy = vertices if isinstance(vertices, np.ndarray) else _xy(vertices)
        # One feature can become several polylines: `spec.max_spacing` caps the
        # resulting point spacing, so a long boundary is split rather than
        # stretched over a fixed 20 points. Lanes are already finer than the cap
        # and come back as a single chunk.
        chunks = geometry.chunk_polyline(xy, spec.max_points, spec.max_spacing)
        points.extend(chunks)
        types.extend([int(kind)] * len(chunks))

    # Shape checks off for the per-polyline geometry: a scene has ~150 map
    # features, and eincheck's per-call overhead was ~70% of ingest time. The
    # assembled `Scenario` is still checked, and ingest re-validates every
    # record it writes.
    with eincheck.disable_checks():
        _add_features(map_json, add)
    return _stack(points, types, spec)


def _add_features(map_json: dict[str, Any], add: Any) -> None:
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

    # AV2 tiles the drivable area into adjacent polygons, so each polygon's
    # boundary includes seams that run straight across the road. The road edge
    # is the outline of their union, where those seams cancel.
    rings = [
        _xy(area["area_boundary"])
        for area in map_json.get("drivable_areas", {}).values()
    ]
    for edge in geometry.union_outline(rings):
        add(edge, PolylineType.ROAD_EDGE)


def _stack(
    points: list[np.ndarray], types: list[int], spec: DatasetSpec
) -> tuple[np.ndarray, np.ndarray]:
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


def _sole(scenario_dir: pathlib.Path, pattern: str) -> pathlib.Path:
    """The one file matching `pattern`, or a message saying what is wrong.

    A bare `[x] = sorted(...)` is the right strictness -- exactly one match, not
    "the first of several" -- but its ValueError ("not enough values to unpack")
    says nothing about which directory or which file is missing, which is
    useless 200k scenarios into an ingest.
    """
    matches = sorted(scenario_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{scenario_dir} is not an AV2 scenario directory: expected exactly "
            f"one {pattern}, found {len(matches)}"
        )
    return matches[0]


def load_scenario(scenario_dir: pathlib.Path, spec: DatasetSpec = AV2) -> Scenario:
    """Read one AV2 scenario directory into a canonical `Scenario`."""
    parquet = _sole(scenario_dir, "scenario_*.parquet")
    map_json = _sole(scenario_dir, "log_map_archive_*.json")
    columns = pq.read_table(parquet).to_pydict()
    agents, focal_index = _agents(columns, spec)
    points, types = _polylines(json.loads(map_json.read_text()), spec)

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


def iter_scenarios(
    root: pathlib.Path,
    spec: DatasetSpec = AV2,
    *,
    worker_index: int = 0,
    num_workers: int = 1,
) -> Iterator[Scenario]:
    """Yield the scenarios under an AV2 split directory, in sorted order.

    With `num_workers > 1`, only every `num_workers`-th directory from
    `worker_index` -- selected from the *listing*, before anything is decoded.
    Filtering decoded scenarios instead costs each of N workers the whole
    split's decode time, for 1/N of the output.
    """
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for scenario_dir in dirs[worker_index::num_workers]:
        yield load_scenario(scenario_dir, spec)

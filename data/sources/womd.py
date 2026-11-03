"""Waymo Open Motion Dataset -> canonical `Scenario`.

Reads `Scenario` protos out of TFRecord shards using vendored protobuf stubs and
a hand-rolled TFRecord reader, so this module depends on nothing beyond
`protobuf`. The `waymo-open-dataset` package is avoided for two independent
reasons: it publishes no arm64 macOS wheel, and it would pull in TensorFlow,
which segfaults when loaded alongside `array_record`.

Layout: one flat directory of shards per split::

    validation/
        validation.tfrecord-00000-of-00150
        validation.tfrecord-00001-of-00150
        ...

Shards must be copied down first -- object-store paths are not supported, see
`data.ingest.__main__` for why. Note `gcloud storage cp` needs
`CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0`; its sliced download
otherwise produces zero-byte files that fail their own checksum.
"""

import pathlib
from collections.abc import Iterator

import numpy as np

from commmon import geometry
from data.schema import AgentTracks, Scenario, VectorMap, encode_scenario_id
from data.sources._tfrecord import read_records
from data.sources._womd_proto import scenario_pb2
from data.specs import WOMD, AgentType, DatasetSpec, PolylineType

# WOMD's 5-way taxonomy. TYPE_OTHER covers everything from debris to animals,
# which is closer to UNKNOWN than to any concrete class.
_OBJECT_TYPE = {
    0: AgentType.UNKNOWN,  # TYPE_UNSET
    1: AgentType.VEHICLE,
    2: AgentType.PEDESTRIAN,
    3: AgentType.CYCLIST,
    4: AgentType.UNKNOWN,  # TYPE_OTHER
}

# RoadLine enum: 1/4/5 are BROKEN_* (dashed), the rest SOLID_*.
_DASHED_ROAD_LINES = frozenset({1, 4, 5})


def _polyline_xy(points) -> np.ndarray:
    """(x, y) from a repeated MapPoint, dropping z."""
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def _map_features(scenario, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Flatten MapFeatures into fixed-length polylines.

    Every feature is resampled to `spec.max_points`, which is genuinely lossy
    here: WOMD polylines are dense (measured p50 37 points, p90 158, max 1578)
    where AV2's are sparse. Splitting long features into fixed-length segments
    preserves the geometry but multiplies the polyline count several-fold, and
    the map budget is already the binding constraint (p99 840 features per
    scenario) -- so WOMD leaves `spec.max_spacing` at its default `inf`, which
    is one polyline per feature. AV2 sets it, because there a single feature can
    be a half-kilometre ring. Set it here too, or raise `max_points`, if the
    fidelity turns out to matter more than the budget.
    """
    points: list[np.ndarray] = []
    types: list[int] = []

    def add(vertices, kind: PolylineType) -> None:
        if len(vertices) < 1:
            return
        points.append(
            geometry.resample_polyline(_polyline_xy(vertices), spec.max_points)
        )
        types.append(int(kind))

    for feature in scenario.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind == "lane":
            add(feature.lane.polyline, PolylineType.LANE_CENTER)
        elif kind == "road_line":
            add(
                feature.road_line.polyline,
                PolylineType.LANE_BOUNDARY_DASHED
                if feature.road_line.type in _DASHED_ROAD_LINES
                else PolylineType.LANE_BOUNDARY_SOLID,
            )
        elif kind == "road_edge":
            add(feature.road_edge.polyline, PolylineType.ROAD_EDGE)
        elif kind == "crosswalk":
            add(feature.crosswalk.polygon, PolylineType.CROSSWALK)
        elif kind == "speed_bump":
            add(feature.speed_bump.polygon, PolylineType.SPEED_BUMP)
        elif kind == "driveway":
            add(feature.driveway.polygon, PolylineType.ROAD_EDGE)
        elif kind == "stop_sign":
            # A stop sign is a single position, not a polyline. resample_polyline
            # turns a one-point line into that point repeated, which is the right
            # degenerate answer -- the feature has a location and no extent.
            add([feature.stop_sign.position], PolylineType.STOP_LINE)

    if not points:
        return (
            np.zeros((0, spec.max_points, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return np.stack(points).astype(np.float32), np.array(types, dtype=np.int32)


def _agents(scenario, spec: DatasetSpec) -> AgentTracks:
    """Track states are already dense over all 91 steps, with a per-step valid flag."""
    num_agents, num_steps = len(scenario.tracks), spec.num_steps

    position = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    velocity = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    heading = np.zeros((num_agents, num_steps), dtype=np.float32)
    valid = np.zeros((num_agents, num_steps), dtype=bool)
    # Unlike AV2, WOMD ships measured box dimensions per state. They are constant
    # per track in practice, so the value at the reference step is taken.
    extent = np.zeros((num_agents, 2), dtype=np.float32)
    agent_type = np.zeros((num_agents,), dtype=np.int32)

    t_ref = spec.num_history - 1
    for i, track in enumerate(scenario.tracks):
        agent_type[i] = int(_OBJECT_TYPE.get(track.object_type, AgentType.UNKNOWN))
        for t, state in enumerate(track.states[:num_steps]):
            if not state.valid:
                continue
            position[i, t] = (state.center_x, state.center_y)
            velocity[i, t] = (state.velocity_x, state.velocity_y)
            heading[i, t] = state.heading
            valid[i, t] = True
        ref = track.states[t_ref] if len(track.states) > t_ref else None
        if ref is not None and ref.valid:
            extent[i] = (ref.length, ref.width)
        else:  # fall back to any valid state rather than leaving a zero box
            for state in track.states:
                if state.valid:
                    extent[i] = (state.length, state.width)
                    break

    return AgentTracks(
        position=position,
        heading=heading,
        velocity=velocity,
        extent=extent,
        agent_type=agent_type,
        valid=valid,
    )


def to_scenario(proto, spec: DatasetSpec = WOMD) -> Scenario:
    """Convert one parsed `Scenario` proto into the canonical form."""
    if len(proto.timestamps_seconds) != spec.num_steps:
        raise ValueError(
            f"scenario {proto.scenario_id!r} has {len(proto.timestamps_seconds)} "
            f"steps, expected {spec.num_steps} for {spec.name}"
        )
    if proto.current_time_index != spec.num_history - 1:
        raise ValueError(
            f"scenario {proto.scenario_id!r} has current_time_index="
            f"{proto.current_time_index}, expected {spec.num_history - 1}"
        )

    agents = _agents(proto, spec)
    points, types = _map_features(proto, spec)
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
        # The self-driving car. WOMD also marks `tracks_to_predict`, but the SDC
        # is the one agent guaranteed present and valid throughout, which is what
        # AgentCentric needs for its reference frame.
        ego_index=np.int32(proto.sdc_track_index),
        scenario_id=encode_scenario_id(proto.scenario_id),
        spec=spec,
    )


def iter_shard(path: pathlib.Path, spec: DatasetSpec = WOMD) -> Iterator[Scenario]:
    """Yield every scenario in one TFRecord shard."""
    for payload in read_records(path):
        yield to_scenario(scenario_pb2.Scenario.FromString(payload), spec)


def iter_scenarios(root: pathlib.Path, spec: DatasetSpec = WOMD) -> Iterator[Scenario]:
    """Yield every scenario across all shards in a split directory, in order."""
    shards = sorted(root.glob("*tfrecord*"))
    if not shards:
        raise FileNotFoundError(
            f"no TFRecord shards under {root} (expected files like "
            "'validation.tfrecord-00000-of-00150')"
        )
    for shard in shards:
        yield from iter_shard(shard, spec)

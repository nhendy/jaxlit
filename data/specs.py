"""Dataset facts and model budgets -- deliberately two different things.

`DatasetSpec` describes the *data*: its sample rate, horizon, and the polyline
resolution baked in at ingest. It rides on every `Scenario` as a static pytree
field and never changes for a given ingested corpus.

`Budget` describes the *model's appetite*: how many agents and polylines to pad
to. It is pipeline configuration, is not stored with the data, and is expected to
change every time someone sweeps it.

Keeping them apart matters because they have completely different lifetimes:

    dt, num_steps, num_history   never change -- properties of the dataset
    max_points                   changing it requires a full re-ingest
                                 (polylines are resampled to it on the way in)
    max_agents, max_polylines    changed freely, no re-ingest -- ingest stores
                                 scenarios ragged and nothing is clipped until
                                 PadAndMask

Conflating them meant a budget sweep had to surgically rewrite the spec carried
by every stored scenario, and two scenarios differing only in budget had
different pytree treedefs and so could not be batched together. Neither is a
real constraint.
"""

import enum

import attrs


class AgentType(enum.IntEnum):
    """Normalized agent taxonomy.

    The three datasets disagree sharply -- WOMD has 3 classes, AV2 has 10 and
    nuScenes 23 -- so each source module maps its own labels onto this set and
    anything unrecognized becomes UNKNOWN rather than being dropped.
    """

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    MOTORCYCLIST = 4
    BUS = 5
    STATIC = 6


class PolylineType(enum.IntEnum):
    """Normalized vector-map element taxonomy."""

    UNKNOWN = 0
    LANE_CENTER = 1
    ROAD_EDGE = 2
    LANE_BOUNDARY_SOLID = 3
    LANE_BOUNDARY_DASHED = 4
    CROSSWALK = 5
    STOP_LINE = 6
    SPEED_BUMP = 7


@attrs.frozen(kw_only=True)
class DatasetSpec:
    """What is true of a dataset, independent of any model.

    Frozen and hashable so it can ride in the pytree treedef via
    `struct.field(pytree_node=False)`. Scenarios from different datasets
    therefore have different treedefs and cannot be batched together -- the
    intended guardrail. Scenarios differing only in *budget* are unaffected,
    because budgets are not stored here.
    """

    name: str
    dt: float
    """Seconds between consecutive timesteps."""
    num_steps: int
    """Total timesteps per scenario: history + future."""
    num_history: int
    """Observed prefix length. `[:num_history]` is input, `[num_history:]` target."""
    max_points: int
    """Points per polyline after arc-length resampling.

    Applied at *ingest*, so changing it means re-converting the corpus. It lives
    here rather than in `Budget` for exactly that reason.
    """

    @property
    def num_future(self) -> int:
        return self.num_steps - self.num_history

    @property
    def history_seconds(self) -> float:
        return self.num_history * self.dt

    @property
    def future_seconds(self) -> float:
        return self.num_future * self.dt


@attrs.frozen(kw_only=True)
class Budget:
    """How many agents and polylines to pad to. A model knob, not a dataset fact.

    Storage barely notices a generous budget -- 512 polylines is 5 MB per batch
    of 64 -- but attention over map elements is quadratic in it, so this trades
    against model cost, and is the sort of thing you sweep.

    Set it from the measured distribution, not intuition. Over 40 real AV2 val
    scenarios::

        agents     p50  54  p90  96  p95 114  p99 121  max 122
        polylines  p50 146  p90 276  p95 287  p99 335  max 354

    An earlier 64/256 silently truncated 32% and 12.5% of scenarios.
    """

    max_agents: int
    max_polylines: int
    strict: bool = False
    """Raise instead of truncating when the budget binds.

    Worth enabling for eval: a silently truncated validation set quietly
    flatters the metrics, and you would rather find out. Training generally
    wants truncation, since the alternative is dropping the scenario.
    """


# 10 Hz, 1.1 s history + 8.0 s future. The 91st step exists because WOMD counts
# the current step in both halves.
WOMD = DatasetSpec(name="womd", dt=0.1, num_steps=91, num_history=11, max_points=20)

# 10 Hz, 5.0 s observed + 6.0 s future.
AV2 = DatasetSpec(name="av2", dt=0.1, num_steps=110, num_history=50, max_points=20)

# 2 Hz, 2.0 s history + 6.0 s future. Far coarser than the other two, which is
# why `dt` is a per-scenario constant rather than a global.
NUSCENES = DatasetSpec(
    name="nuscenes", dt=0.5, num_steps=16, num_history=4, max_points=20
)

ALL_SPECS: dict[str, DatasetSpec] = {s.name: s for s in (WOMD, AV2, NUSCENES)}

# Starting points, not gospel -- override per experiment.
DEFAULT_BUDGETS: dict[str, Budget] = {
    # Measured: covers the observed max of 122 agents / 354 polylines.
    "av2": Budget(max_agents=128, max_polylines=512),
    # Measured over 120 real validation scenarios:
    #     agents        p50  59  p90 141  p99 267  max  288
    #     map_features  p50 304  p90 582  p99 840  max 1015
    # The earlier 128/512 placeholder truncated 14.2% and 15.8% of scenarios.
    # Note 1024 polylines is 4x the map self-attention cost of 512 -- if the
    # model turns out attention-bound, trading back to 512 loses the tail of
    # ~16% of scenarios, which is a real but bounded cost.
    "womd": Budget(max_agents=320, max_polylines=1024),
    # Also unmeasured; nuScenes scenes are denser in agents but the prediction
    # split is sparser. Re-derive once v1.0-trainval_meta is available.
    "nuscenes": Budget(max_agents=128, max_polylines=512),
}

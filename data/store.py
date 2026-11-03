"""Random access over ingested scenarios.

The adapters and `read_shards` are iterators, which is right for ingest and for
training. A visualiser wants the opposite: jump straight to scenario 14,203
because someone dragged a slider. That is exactly what ArrayRecord buys and why
the pipeline stores shards rather than a stream -- a decode is ~0.4 ms, so a UI
can seek anywhere without a loading state.

    store = ScenarioStore.open("~/datasets/av2/ingested/val")
    scenario = store[1234]

Deliberately not a `grain` dataset: this is for interactive use, with no
shuffling, batching or worker pool. Mixing the two would mean pulling grain's
machinery into a Streamlit process for no benefit.
"""

import dataclasses
from collections.abc import Iterator, Sequence

import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from etils import epath

from data import codec
from data.schema import Scenario, decode_scenario_id


@dataclasses.dataclass(frozen=True)
class ScenarioStore:
    """Indexable, read-only view over a directory of ArrayRecord shards."""

    source: ArrayRecordDataSource

    @classmethod
    def open(cls, root: epath.PathLike) -> "ScenarioStore":
        """Open every `*.arrayrecord` shard under `root`, in sorted order.

        Sorted so an index means the same scenario across runs -- a visualiser
        that deep-links to "scenario 1234" should not silently point somewhere
        else because the filesystem returned shards in a different order.
        """
        shards = sorted(epath.Path(root).expanduser().glob("*.arrayrecord"))
        if not shards:
            raise FileNotFoundError(f"no .arrayrecord shards under {root}")
        return cls(ArrayRecordDataSource([str(p) for p in shards]))

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Scenario:
        """Decode one scenario. Negative indices work as they do for a list."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} scenarios")
        return codec.decode(self.source[index])

    def __iter__(self) -> Iterator[Scenario]:
        for i in range(len(self)):
            yield self[i]

    def scenario_id(self, index: int) -> str:
        """The dataset's own id for a scenario, e.g. to cross-reference raw files.

        Decodes the whole record, so it is not cheaper than `__getitem__` -- it
        exists to spare callers the two-step dance.
        """
        return decode_scenario_id(self[index].scenario_id)

    def find(self, scenario_id: str) -> int:
        """Index of a scenario by its dataset id.

        A linear scan, because nothing in the shard layout indexes by id. Fine
        for a one-off lookup; build a dict if you need it in a loop.
        """
        for i in range(len(self)):
            if self.scenario_id(i) == scenario_id:
                return i
        raise KeyError(f"no scenario with id {scenario_id!r}")


def agent_track(
    scenario: Scenario, index: int
) -> tuple[Sequence[float], Sequence[float]]:
    """One agent's (x, y) over time, with invalid steps dropped.

    Plotting the raw arrays would draw a line through the origin wherever an
    agent is unobserved, since invalid entries are zeroed. Masking here keeps
    that out of every call site.
    """

    valid = np.asarray(scenario.agents.valid)[index]
    position = np.asarray(scenario.agents.position)[index][valid]
    return position[:, 0], position[:, 1]

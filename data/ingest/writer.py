"""Sharded ArrayRecord writer -- the output side of the one-time ingest.

ArrayRecord is what makes the training pipeline a `grain.MapDataset` rather than
an `IterDataset`: it is randomly addressable, which buys global shuffling and
exact checkpoint restore. None of the three raw formats offers that, which is the
whole reason this conversion step exists.
"""

import pathlib
from collections.abc import Iterable, Iterator

from array_record.python.array_record_module import ArrayRecordWriter

from data import codec
from data.schema import Scenario

WRITER_OPTIONS = "group_size:1"
"""One record per compression group.

The default groups many records together, so reading any one of them decompresses
the whole group -- fine for sequential reads, ruinous for the random access this
pipeline is built on.
"""


def shard_paths(
    out_dir: pathlib.Path, num_shards: int, prefix: str
) -> list[pathlib.Path]:
    """Canonical `<prefix>-00000-of-00064.arrayrecord` names."""
    return [
        out_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.arrayrecord"
        for i in range(num_shards)
    ]


def write_shards(
    scenarios: Iterable[Scenario],
    out_dir: pathlib.Path,
    *,
    num_shards: int = 64,
    prefix: str = "data",
    progress_every: int = 0,
) -> list[pathlib.Path]:
    """Encode `scenarios` round-robin across `num_shards` ArrayRecord files.

    Round-robin rather than contiguous chunks because the raw datasets arrive in
    a meaningful order (WOMD shards are grouped by segment), and contiguous
    chunking would preserve that correlation inside each output shard.

    Returns the shard paths that were written.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = shard_paths(out_dir, num_shards, prefix)
    writers = [ArrayRecordWriter(str(p), WRITER_OPTIONS) for p in paths]

    count = 0
    try:
        for count, scenario in enumerate(scenarios, start=1):
            writers[(count - 1) % num_shards].write(codec.encode(scenario))
            if progress_every and count % progress_every == 0:
                print(f"  wrote {count} scenarios", flush=True)
    finally:
        # Unclosed writers leave truncated files that fail confusingly at read
        # time, so close them even if the source iterator raised.
        for writer in writers:
            writer.close()

    if count < num_shards:
        # Empty shards are legal but a symptom: ArrayRecordDataSource happily
        # opens them, and the user probably wanted fewer, fuller shards.
        print(
            f"warning: {count} scenarios across {num_shards} shards "
            f"leaves {num_shards - count} empty",
            flush=True,
        )
    return paths


def read_shards(paths: Iterable[pathlib.Path]) -> Iterator[Scenario]:
    """Decode every record back, for verifying an ingest run."""
    import grain

    source = grain.sources.ArrayRecordDataSource([str(p) for p in paths])
    for i in range(len(source)):
        yield codec.decode(source[i])

"""The single dataset-agnostic Grain pipeline.

Every dataset reaches this the same way: ingest normalized it into ArrayRecord
shards of encoded `Scenario`s, so nothing below this line knows whether the bytes
came from a WOMD proto, an AV2 parquet, or the nuScenes JSON database.

`MapDataset` (random access) rather than `IterDataset` (sequential) is the
load-bearing choice -- it is what gives global shuffling and exact checkpoint
restore. `IterDataset` offers neither, and the raw shards are *ordered*, so
window shuffling would leave strong correlation inside every batch.
"""

import pathlib
from collections.abc import Sequence

import grain

from data.specs import DEFAULT_BUDGETS, Budget, DatasetSpec
from data.transforms import AgentCentric, DecodeScenario, PadAndMask, RandomRotate


def build(
    paths: str | pathlib.Path | Sequence[str | pathlib.Path],
    spec: DatasetSpec,
    *,
    batch_size: int,
    seed: int,
    budget: Budget | None = None,
    training: bool = True,
    shuffle: bool | None = None,
    num_workers: int = 0,
    max_rotation: float = 3.141592653589793,
) -> grain.IterDataset:
    """Build the training/eval input pipeline over ArrayRecord shards.

    Args:
      paths: ArrayRecord shard path(s) written by `data.ingest.writer`.
      spec: dataset constants; must match the scenarios in `paths`.
      batch_size: examples per batch.
      budget: how many agents/polylines to pad to. A model knob, independent of
        the data, so changing it needs no re-ingest. Defaults to
        `DEFAULT_BUDGETS[spec.name]`; pass `Budget(..., strict=True)` for eval
        if a bound budget should be an error rather than a silent truncation.
      seed: seeds both shuffling and augmentation, so a run is reproducible
        from this alone.
      training: enables augmentation and drops the short final batch. Set False
        for eval, where every example must be seen exactly once.
      shuffle: defaults to `training`.
      num_workers: multiprocessing workers. 0 keeps everything in-process,
        which is what you want in tests -- Grain spawns workers, so an
        exception in a transform is much easier to read without them.
      max_rotation: augmentation range in radians; ignored unless `training`.

    Returns:
      An `IterDataset` yielding batched `Scenario`s with `n=(batch_size,)`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    shuffle = training if shuffle is None else shuffle
    budget = DEFAULT_BUDGETS[spec.name] if budget is None else budget

    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    source = grain.sources.ArrayRecordDataSource([str(p) for p in paths])

    # .seed() must precede any random transform -- grain raises otherwise. It
    # also makes the whole pipeline reproducible from `seed` alone.
    ds = grain.MapDataset.source(source).seed(seed)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    ds = ds.map(DecodeScenario())
    ds = ds.map(AgentCentric())
    if training:
        ds = ds.random_map(RandomRotate(max_angle=max_rotation))

    # Last per-example step: .batch() stacks, so shapes must already match.
    ds = ds.map(PadAndMask(budget))
    ds = ds.batch(batch_size, drop_remainder=training)

    ds = ds.to_iter_dataset()
    if num_workers > 0:
        ds = ds.mp_prefetch(
            grain.multiprocessing.MultiprocessingOptions(num_workers=num_workers)
        )
    return ds

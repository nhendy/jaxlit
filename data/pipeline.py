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

import attrs
import grain

from core.distributed import Shard
from data.specs import DEFAULT_BUDGETS, Budget, DatasetSpec, TimeWindow
from data.transforms import (
    AgentCentric,
    DecodeScenario,
    DropMap,
    PadAndMask,
    RandomRotate,
    Resample,
    ShapeChecks,
)


@attrs.frozen(kw_only=True)
class Stages:
    """Which optional per-scene transforms run -- a model-side choice.

    The defaults are the full canonical pipeline. Turn off what a model does
    not consume: each stage is per-scene Python on the CPU, and the input
    pipeline is what usually starves the accelerator.
    """

    # Re-express each scene in the ego's frame (`AgentCentric`). A model that
    # builds its own frames on device (the forecaster: every agent in its own)
    # makes this a no-op.
    ego_frame: bool = True
    # Random rotation about the ego. None: when training. Needs `ego_frame`.
    augment: bool | None = None
    # False drops the vector map before any transform touches it, for models
    # that never read it; batches keep the schema's shapes, every polyline
    # masked out. Pair True with a polyline budget the map encoder can use.
    keep_map: bool = True
    # A coarser, shorter timeline (`data.specs.TimeWindow`); None keeps the
    # dataset's own. Scenarios -- and `output_spec` -- then describe the window.
    window: TimeWindow | None = None

    def output_spec(self, spec: DatasetSpec) -> DatasetSpec:
        """The spec of the scenarios this pipeline yields."""
        return spec if self.window is None else self.window.apply(spec)

    def __attrs_post_init__(self) -> None:
        if self.augment and not self.ego_frame:
            raise ValueError("augment rotates about the ego, so it needs ego_frame")


def build(
    paths: str | pathlib.Path | Sequence[str | pathlib.Path],
    spec: DatasetSpec,
    *,
    batch_size: int,
    seed: int,
    budget: Budget | None = None,
    training: bool = True,
    shuffle: bool | None = None,
    repeat: bool = False,
    shard: Shard | None = None,
    num_workers: int = 0,
    max_rotation: float = 3.141592653589793,
    stages: Stages = Stages(),
    shape_checks: bool = True,
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
      shard: which rows this process reads. Defaults to one slice per JAX
        process, so under multi-host training `batch_size` is per process.
      repeat: cycle forever, for step-budgeted training. Each epoch is
        reshuffled and re-augmented, since both are keyed by the global index.
      num_workers: multiprocessing workers. 0 keeps everything in-process,
        which is what you want in tests -- Grain spawns workers, so an
        exception in a transform is much easier to read without them.
      max_rotation: augmentation range in radians; ignored unless augmenting.
      stages: which optional transforms run; see `Stages`.
      shape_checks: run eincheck's per-example shape checks. Ingest already
        validated every record, so turning them off loses no safety and
        removes most of the pipeline's CPU cost.

    Returns:
      An `IterDataset` yielding batched `Scenario`s with `n=(batch_size,)`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    shuffle = training if shuffle is None else shuffle
    augment = training if stages.augment is None else stages.augment
    if augment and not stages.ego_frame:
        raise ValueError("augment rotates about the ego, so it needs ego_frame")
    budget = DEFAULT_BUDGETS[spec.name] if budget is None else budget

    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    source = grain.sources.ArrayRecordDataSource([str(p) for p in paths])

    # .seed() must precede any random transform -- grain raises otherwise. It
    # also makes the whole pipeline reproducible from `seed` alone.
    shard = Shard.current() if shard is None else shard
    ds = grain.MapDataset.source(source)[shard.index :: shard.count].seed(seed)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if repeat:
        ds = ds.repeat()

    ds = ds.map(ShapeChecks(enabled=shape_checks))
    ds = ds.map(DecodeScenario())
    if stages.window is not None:
        ds = ds.map(Resample(window=stages.window))
    if not stages.keep_map:
        ds = ds.map(DropMap())
    if stages.ego_frame:
        ds = ds.map(AgentCentric())
    if augment:
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

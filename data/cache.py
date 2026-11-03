"""Train from a materialized pipeline: run the per-scene transforms once, then
just gather rows.

The Grain pipeline pays decode, resample and pad in Python for every scene of
every epoch. When those stages are deterministic -- no augmentation -- their
output never changes, so `materialize` runs them once and writes every padded
scene as one row of a per-leaf `.npy` file; `dataset` serves them back
through plain Grain (shuffle, repeat, batch), with nothing left to compute.

Why not `grain.experimental.CacheIterDataset`: it caches in memory only, so
every run and every process rebuilds it, and it replays its first pass, so
every epoch has the same order. This cache persists (on the data volume) and
shuffles each epoch.

    directory = cache.materialize(paths, spec, budget=..., stages=..., root=...)
    scenes = cache.load(directory, cache.template(paths, spec, ...))
    for batch in cache.dataset(scenes, batch_size=512, seed=0): ...

The cache directory is named by a fingerprint of everything that shapes its
contents (shards, spec, budget, stages), so changing any of them builds a new
cache instead of silently reusing a stale one.

Costs memory, not CPU: rows are stored padded (AV2 at 64 agents, 5 Hz, no
map: ~62 KB a scene, 12 GB for all of train), and `load` reads them into RAM.
"""

import contextlib
import hashlib
import os
import pathlib
import re
import time
from collections.abc import Sequence
from typing import BinaryIO

import attrs
import grain
import jax
import numpy as np
from absl import logging
from array_record.python.array_record_data_source import ArrayRecordDataSource

from core.distributed import Shard
from data import pipeline
from data.pipeline import Stages
from data.schema import Scenario
from data.specs import Budget, DatasetSpec

_DONE = ".done"


def _paths(paths: Sequence[pathlib.Path]) -> list[str]:
    return [str(pathlib.Path(p).expanduser()) for p in paths]


def fingerprint(
    paths: Sequence[pathlib.Path], spec: DatasetSpec, budget: Budget, stages: Stages
) -> str:
    """Names the cache: every input that changes what its rows hold."""
    shards = sorted(
        (p.name, p.stat().st_size) for p in map(pathlib.Path, _paths(paths))
    )
    key = repr((shards, spec, budget, stages))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _check_deterministic(stages: Stages) -> None:
    if stages.augment is not False:
        raise ValueError(
            "a cache freezes the pipeline's output, so it needs "
            f"Stages(augment=False); got {stages}"
        )


def template(
    paths: Sequence[pathlib.Path], spec: DatasetSpec, *, budget: Budget, stages: Stages
) -> Scenario:
    """One scene through the same pipeline: the cache's tree structure and row
    shapes, without reading the cache."""
    return next(
        iter(
            pipeline.build(
                _paths(paths)[:1],
                spec,
                batch_size=1,
                seed=0,
                budget=budget,
                training=False,
                shard=Shard(index=0, count=1),
                stages=stages,
                shape_checks=False,
            )
        )
    )


def _leaf_names(tree: Scenario) -> list[str]:
    """File names for the leaves, in flatten order: `agents.position` etc."""
    return [
        re.sub(r"[^A-Za-z0-9_.]", "", jax.tree_util.keystr(path)).strip(".")
        for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]
    ]


def _drop_from_page_cache(f: BinaryIO) -> None:
    """Write `f` out and let the kernel forget it (Linux; elsewhere just sync)."""
    f.flush()
    os.fsync(f.fileno())
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is not None:
        fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def materialize(
    paths: Sequence[pathlib.Path],
    spec: DatasetSpec,
    *,
    budget: Budget,
    stages: Stages,
    root: pathlib.Path,
    num_workers: int = 0,
    batch_size: int = 256,
) -> pathlib.Path:
    """Write every scene in `paths`, transformed and padded, under `root`.

    Idempotent: returns at once if this exact cache is already complete.
    Single-process: run it as a setup step, not from every training process.

    Returns:
      The cache directory, `root/<fingerprint>`.
    """
    _check_deterministic(stages)
    directory = root.expanduser() / fingerprint(paths, spec, budget, stages)
    if (directory / _DONE).exists():
        logging.info("cache: %s already complete", directory)
        return directory
    directory.mkdir(parents=True, exist_ok=True)

    example = template(paths, spec, budget=budget, stages=stages)
    names = _leaf_names(example)
    count = len(ArrayRecordDataSource(_paths(paths)))
    batches = pipeline.build(
        _paths(paths),
        spec,
        batch_size=batch_size,
        seed=0,
        budget=budget,
        training=False,
        shard=Shard(index=0, count=1),
        num_workers=num_workers,
        stages=stages,
        shape_checks=False,
    )
    started, row = time.perf_counter(), 0
    # Streamed, not memory-mapped: pages written through a memmap stay resident
    # (and count against the container's memory) until it is closed, which for
    # a cache bigger than RAM is an OOM kill. Plain writes, flushed and dropped
    # from the page cache as they go, hold ~nothing.
    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(open(directory / f"{name}.npy", "wb")) for name in names
        ]
        for f, leaf in zip(files, jax.tree.leaves(example), strict=True):
            header = np.lib.format.header_data_from_array_1_0(np.asarray(leaf))
            header["shape"] = (count, *np.shape(leaf)[1:])
            np.lib.format.write_array_header_1_0(f, header)
        for batch in batches:
            leaves = jax.tree.leaves(batch)
            for f, leaf in zip(files, leaves, strict=True):
                f.write(np.ascontiguousarray(leaf).tobytes())
            row += np.shape(leaves[0])[0]
            if row % (100 * batch_size) == 0:
                for f in files:
                    _drop_from_page_cache(f)
                logging.info(
                    "cache: %d / %d scenes (%.0f/s)",
                    row,
                    count,
                    row / (time.perf_counter() - started),
                )
        for f in files:
            _drop_from_page_cache(f)
    if row != count:
        raise RuntimeError(f"cache: wrote {row} scenes, expected {count}")
    (directory / _DONE).touch()
    logging.info(
        "cache: %d scenes -> %s in %.0fs",
        count,
        directory,
        time.perf_counter() - started,
    )
    return directory


_CHUNK = 256 << 20


def _read(path: pathlib.Path) -> np.ndarray:
    """`np.load`, minus the page cache.

    A plain read leaves a copy of every byte in the page cache, and container
    memory limits (and Ray's OOM killer) count it: loading a 22 GB cache looks
    like 44 GB. Reading in chunks straight into the array, and telling the
    kernel to drop each chunk once read, holds only the array.
    """
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(f)
        if fortran:
            raise ValueError(f"{path}: Fortran order is not what materialize writes")
        array = np.empty(shape, dtype=dtype)
        view = memoryview(array.reshape(-1).view(np.uint8))
        fadvise = getattr(os, "posix_fadvise", None)
        offset = f.tell()
        for start in range(0, len(view), _CHUNK):
            chunk = view[start : start + _CHUNK]
            if f.readinto(chunk) != len(chunk):
                raise OSError(f"{path}: truncated")
            if fadvise is not None:
                fadvise(f.fileno(), offset + start, len(chunk), os.POSIX_FADV_DONTNEED)
    return array


def load(
    directory: pathlib.Path, like: Scenario, *, in_memory: bool = True
) -> Scenario:
    """Every cached scene as one `Scenario` with a leading axis of all rows.

    Args:
      directory: what `materialize` returned.
      like: a `template` -- supplies the tree structure; its rows must match.
      in_memory: read every leaf into RAM (one sequential read; right for a
        network volume), bypassing the page cache -- see `_read`. False
        memory-maps them instead, for caches bigger than RAM on fast local
        disk.
    """
    if not (directory / _DONE).exists():
        raise FileNotFoundError(
            f"no complete cache at {directory}; build it with `materialize`"
        )
    leaves = []
    for name, leaf in zip(_leaf_names(like), jax.tree.leaves(like), strict=True):
        path = directory / f"{name}.npy"
        array = _read(path) if in_memory else np.load(path, mmap_mode="r")
        if array.shape[1:] != leaf.shape[1:] or array.dtype != leaf.dtype:
            raise ValueError(
                f"cache leaf {name} is {array.dtype}{array.shape[1:]}, the pipeline "
                f"now makes {leaf.dtype}{leaf.shape[1:]}"
            )
        leaves.append(array)
    return jax.tree.unflatten(jax.tree.structure(like), leaves)


@attrs.frozen(kw_only=True, eq=False)
class Gather:
    """A batch of row indices -> those scenes, one vectorized take per leaf."""

    scenes: Scenario

    def __call__(self, rows: np.ndarray) -> Scenario:
        # Sorted: same batch, sequential reads.
        rows = np.sort(rows)
        return jax.tree.map(lambda x: np.take(x, rows, axis=0), self.scenes)


def dataset(
    scenes: Scenario,
    *,
    batch_size: int,
    seed: int,
    shard: Shard | None = None,
    num_threads: int = 2,
) -> grain.IterDataset[Scenario]:
    """Endless shuffled training batches from loaded scenes.

    Grain shuffles, repeats and batches row *indices* -- each epoch reshuffled
    by `seed`, sharded per process like `pipeline.build` -- and `Gather` turns
    each index batch into scenes with one numpy take per leaf. Batching the
    scenes themselves would be per-scene Python (~30k scenes/s on one core,
    about what a GPU eats); the take is ~10x that. `num_threads` gather ahead
    of the training loop; numpy releases the GIL while copying.
    """
    shard = Shard.current() if shard is None else shard
    count = jax.tree.leaves(scenes)[0].shape[0]
    ds = grain.MapDataset.range(count)[shard.index :: shard.count]
    ds = ds.seed(seed).shuffle(seed=seed).repeat()
    ds = ds.batch(batch_size, drop_remainder=True).map(Gather(scenes=scenes))
    return ds.to_iter_dataset(
        grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=16)
    )


def eval_dataset(
    scenes: Scenario, *, batch_size: int, shard: Shard | None = None
) -> grain.IterDataset[Scenario]:
    """Every loaded scene exactly once, in order: eval batches.

    The last batch is short; `Partitioner.eval_batches` pads and masks it.
    Re-iterable, so an eval callback can sweep it at every firing.
    """
    shard = Shard.current() if shard is None else shard
    count = jax.tree.leaves(scenes)[0].shape[0]
    ds = grain.MapDataset.range(count)[shard.index :: shard.count]
    ds = ds.batch(batch_size, drop_remainder=False).map(Gather(scenes=scenes))
    return ds.to_iter_dataset()

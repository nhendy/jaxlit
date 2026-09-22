"""Ingest CLI: raw dataset -> ArrayRecord shards of canonical `Scenario`s.

    uv run python -m data.ingest \\
        --dataset=argoverse --src=/data/av2/val --out=/data/ingested/av2-val

This is the one-time conversion the whole library is built around. Afterwards
nothing downstream knows which dataset the bytes came from -- `data.pipeline`
reads the shards the same way for all three.

Uses absl rather than argparse deliberately: `app.run()` parses flags, and Grain
reads absl flags at runtime (`grain_enable_multiprocess_worker_profiling` among
them). An entrypoint that never parses them dies the moment it enables
multiprocessing. See the repo `conftest.py` for the pytest equivalent.
"""

import time
from collections.abc import Iterator

from absl import app, flags, logging
from etils import epath

from data.ingest.writer import read_shards, write_shards
from data.schema import Scenario
from data.specs import ALL_SPECS, DatasetSpec

_DATASET = flags.DEFINE_enum(
    "dataset", None, sorted(ALL_SPECS), "Which dataset to ingest.", required=True
)
_SRC = flags.DEFINE_string(
    "src", None, "Raw dataset root (a local directory).", required=True
)
_OUT = flags.DEFINE_string("out", None, "Output directory.", required=True)
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 64, "ArrayRecord shards to write.")
_LIMIT = flags.DEFINE_integer(
    "limit", None, "Stop after this many scenarios (for smoke tests)."
)
_PREFIX = flags.DEFINE_string("prefix", None, "Shard name prefix; defaults to dataset.")
_VERIFY = flags.DEFINE_bool(
    "verify", True, "Read every record back after writing and re-validate it."
)
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 1, "Total workers splitting this split between them."
)
_WORKER_INDEX = flags.DEFINE_integer(
    "worker_index", 0, "This worker's index in [0, num_workers)."
)


def _source(dataset: str) -> tuple[DatasetSpec, object]:
    """Resolve a dataset name to its spec and `iter_scenarios`.

    Imported lazily so a missing adapter is a clear message about *that*
    dataset, not an import error at CLI start for the two you are not using.
    """
    spec = ALL_SPECS[dataset]
    if dataset == "av2":
        from data.sources import argoverse

        return spec, argoverse.iter_scenarios
    if dataset == "womd":
        from data.sources import womd

        return spec, womd.iter_scenarios
    if dataset == "nuscenes":
        raise NotImplementedError(
            "the nuScenes adapter is not written yet (build order step 6); it "
            "needs nuscenes-devkit and the v1.0-mini download"
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def _shard(
    scenarios: Iterator[Scenario], worker_index: int, num_workers: int
) -> Iterator[Scenario]:
    """Keep every `num_workers`-th scenario, offset by this worker's index.

    Strided rather than contiguous so every worker sees the same mix: the raw
    datasets arrive grouped (WOMD by segment, AV2 by sorted id), and handing
    worker 0 the first Nth of a sorted listing would give it a systematically
    different slice from worker 7.

    Each worker still reads the whole *listing*, which is cheap, and decodes
    only its own scenarios, which is not.
    """
    if not 0 <= worker_index < num_workers:
        raise app.UsageError(
            f"--worker_index {worker_index} outside [0, {num_workers})"
        )
    for i, scenario in enumerate(scenarios):
        if i % num_workers == worker_index:
            yield scenario


def _progress(
    scenarios: Iterator[Scenario], limit: int | None, every: int = 500
) -> Iterator[Scenario]:
    """Log throughput, and stop at `limit`.

    Wrapping the iterator rather than the writer keeps `write_shards` free of
    reporting concerns and lets the count reflect scenarios *read*, which is
    where the time actually goes.
    """
    started = time.perf_counter()
    for count, scenario in enumerate(scenarios, start=1):
        if limit is not None and count > limit:
            logging.info("stopping at --limit=%d", limit)
            return
        if count % every == 0:
            rate = count / (time.perf_counter() - started)
            logging.info("read %d scenarios (%.1f/s)", count, rate)
        yield scenario


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"unexpected positional arguments: {argv[1:]}")

    # epath, not pathlib: it handles gs:// as well as local paths, and etils is
    # already a dependency.
    # gs:// is refused rather than supported, and the reason is not obvious:
    # etils.epath implements gs:// on top of TensorFlow's filesystem, and on
    # this platform importing tensorflow into a process that has array_record
    # loaded segfaults (both are C++ extensions; they conflict). Since ingest
    # must write ArrayRecord, TensorFlow can never be in this process.
    #
    # Copy the shards down first -- `gcloud storage cp` streams them without
    # any Python involved -- then point --src at the local directory.
    if _SRC.value.startswith(("gs://", "s3://")):
        raise app.UsageError(
            f"--src must be a local directory, got {_SRC.value!r}.\n"
            "Object-store paths are not supported here: reading them would pull "
            "TensorFlow into a process that already holds array_record, which "
            "segfaults on this platform.\n"
            "Copy the shards down first:\n"
            f"  gcloud storage cp -r {_SRC.value} /local/scratch/"
        )
    src = epath.Path(_SRC.value)
    if not src.is_dir():
        raise app.UsageError(f"--src is not a directory: {src}")
    # Output always stages locally: ArrayRecordWriter is a C++ extension that
    # writes local files. rclone moves it to object storage afterwards.
    out = epath.Path(_OUT.value)

    spec, iter_scenarios = _source(_DATASET.value)
    num_workers, worker_index = _NUM_WORKERS.value, _WORKER_INDEX.value

    # The shard prefix carries the worker index, so N workers writing into the
    # same output directory cannot collide on a filename.
    prefix = _PREFIX.value or spec.name
    if num_workers > 1:
        prefix = f"{prefix}-w{worker_index:03d}"

    logging.info(
        "ingesting %s from %s -> %s (worker %d/%d)",
        spec.name,
        src,
        out,
        worker_index,
        num_workers,
    )
    started = time.perf_counter()
    scenarios = _shard(iter_scenarios(src, spec), worker_index, num_workers)
    paths = write_shards(
        _progress(scenarios, _LIMIT.value),
        out,
        num_shards=_NUM_SHARDS.value,
        prefix=prefix,
    )
    elapsed = time.perf_counter() - started

    written = sum(p.stat().length for p in paths)
    logging.info(
        "wrote %d shards, %.1f MB, in %.1fs",
        len(paths),
        written / (1024 * 1024),
        elapsed,
    )

    if _VERIFY.value:
        # Decoding re-runs every eincheck spec, so this is a real integrity
        # check rather than a byte count -- a truncated shard fails here.
        count = sum(1 for _ in read_shards(paths))
        logging.info("verified %d records decode and validate", count)
        if count == 0:
            raise app.UsageError(
                f"no scenarios found under {src}"
                + (
                    f" for worker {worker_index}/{num_workers}"
                    if num_workers > 1
                    else ""
                )
            )

    print(f"{out}")


if __name__ == "__main__":
    app.run(main)

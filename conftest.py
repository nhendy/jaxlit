"""Pytest configuration for the whole repo.

Grain reads absl flags at runtime (`grain_enable_multiprocess_worker_profiling`
among others). Those flags are normally parsed by `absl.app.run()`, which pytest
never calls -- so anything touching `mp_prefetch` dies with

    UnparsedFlagAccessError: Trying to access flag
    --grain_enable_multiprocess_worker_profiling before flags were parsed

Marking them parsed here leaves every flag at its default, which is what an
`absl.app.run()` entrypoint with no flags on the command line would do anyway.

The same applies outside pytest: a plain `python train.py` that never parses
flags will hit this the first time it enables Grain multiprocessing. Launch
training through `absl.app.run()`, or do the same two lines below.
"""

import grain.multiprocessing  # noqa: F401  -- registers the flags before parsing
from absl import flags


def pytest_configure(config: object) -> None:
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()

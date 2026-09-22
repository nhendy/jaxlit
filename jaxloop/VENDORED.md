# Vendored: google/jaxloop

Upstream: https://github.com/google/jaxloop
Commit:   d1bac45a77c4cf3b852c69ef9fe53e04161b0cbf (master)
Vendored: 2026-08-23
License:  Apache 2.0 (see LICENSE)

## Why this is vendored instead of installed

`pip install jaxloop` does **not** get you this library. The PyPI name is held by
an unrelated 996-byte demo package (v0.0.1, `github.com/jeffcarp/loop-docs`)
containing an empty `__init__.py` and an `actions.py` with `ActionOne`/`foo()`.
Google's jaxloop has never been published to PyPI, despite its README saying
`pip install -U jaxloop`.

Installing from git doesn't work either: upstream's `pyproject.toml` builds
`packages = ["jaxloop"]`, but that top-level directory is a leftover stub holding
only `__init__.py` — the real source lives in `src/jaxloop/`. A git install
therefore succeeds and installs nothing usable.

## What was copied

`src/jaxloop/` → this directory, plus upstream's top-level `jaxloop/__init__.py`
(which the wheel ships but which is missing from `src/jaxloop/`).

Excluded: `*_test.py` and `tests/`, matching upstream's own wheel `exclude` rule.

## Known-unimportable modules

These ship upstream but reference packages unavailable outside Google. They are
kept for reference; don't import them.

- `pipeline_loop.py` — imports `ml_metrics`
- `examples/resnet_dataset.py`, `examples/train_resnet.py` — import `google3`
- `examples/xm_launch.py` — imports `xmanager`
- `examples/resnet_pygrain_dataset.py` — imports `cv2`, `tensorflow`

## Note on the package namespace

`src/jaxloop/` has no `__init__.py` upstream, so submodules are not re-exported.
`import jaxloop as jl; jl.OuterLoop` does not work. Use the submodule path:

    from jaxloop.outer_loop import OuterLoop

## Refreshing

    git clone --depth 1 https://github.com/google/jaxloop /tmp/jaxloop-src
    rsync -a --delete --exclude='*_test.py' --exclude='tests/' \
        --exclude='__pycache__/' --exclude='VENDORED.md' --exclude='LICENSE' \
        /tmp/jaxloop-src/src/jaxloop/ ./jaxloop/
    cp /tmp/jaxloop-src/jaxloop/__init__.py ./jaxloop/__init__.py

Upstream's runtime dependencies are declared directly in this project's
`pyproject.toml`: `clu`, `optax`, `jaxtyping`, `absl-py`, `etils`, plus
`tensorflow` and `tensorboard` — those last two are not imported by jaxloop
itself, but `clu.metric_writers` imports them unconditionally via its TF summary
writer, so every loop module needs them present.

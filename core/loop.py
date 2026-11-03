"""`fit`: the one impure loop. Everything it calls is pure or a callback."""

import itertools
import time
from collections.abc import Iterable, Sequence
from typing import Any, cast

import jax
from absl import logging

from core import stats
from core.callbacks import Callback, Control, StepEvent, fires
from core.compile import compile_train, stack
from core.config import LoopConfig
from core.distributed import (
    Partitioner,
    Replicated,
    StateSharding,
    host_all,
    host_any,
    is_primary,
)
from core.hooks import StepHook
from core.metrics import Metrics
from core.task import Task
from core.types import jit


def fit[StateT, BatchT, MetricsT: Metrics](
    task: Task[StateT, BatchT, MetricsT, Any],
    *,
    config: LoopConfig,
    train: Iterable[BatchT],
    hooks: Sequence[StepHook[StateT, MetricsT]] = (),
    callbacks: Sequence[Callback[StateT, MetricsT, Any]] = (),
    state: StateT | None = None,
    partitioner: Partitioner | None = None,
    sharding: StateSharding[StateT] = Replicated(),
) -> StateT:
    """Train until `config.num_steps`, the data runs out, or a callback stops it.

    Args:
      task: what to train.
      config: step budget, steps per compiled call, seed.
      train: batches; consumed `config.steps_per_call` at a time. A task that
        generates its own data passes `itertools.repeat(NoBatch())`.
      hooks: pure post-step transforms, compiled into the step.
      callbacks: host-side side effects, run between compiled calls.
      state: resume from here instead of `task.init` -- typically the result
        of `core.callbacks.restore_latest`. Its buffers are donated to the
        first step: use the returned state, not this one, afterwards.
      partitioner: the device mesh. Defaults to `config.mesh` over every device.
        Callbacks that compute (`Evaluate`) must be given the same one.
      sharding: how the state is laid out on the mesh.

    Multi-host: every process calls `fit` with its own slice of each batch
    (`train` yields process-local batches; the global batch is their
    concatenation). Running out of data or a callback's STOP on any process
    ends the loop on all of them.

    Returns:
      The final state.
    """
    partitioner = (
        Partitioner.create(config.mesh) if partitioner is None else partitioner
    )
    batches = iter(train)
    k = config.steps_per_call
    if state is None:
        first = next(batches)
        rng = jax.random.key(config.seed)
        example = partitioner.to_global(first)
        abstract = jax.eval_shape(task.init, rng, example)
        shardings = partitioner.shardings(abstract, sharding)
        current: StateT = jit(task.init, out_shardings=shardings)(rng, example)
        batches = itertools.chain([first], batches)
    else:
        shardings = partitioner.shardings(state, sharding)
        current = cast(StateT, jax.device_put(state, shardings))

    run = compile_train(
        task,
        steps_per_call=k,
        hooks=hooks,
        inner_loop=config.inner_loop,
        partitioner=partitioner,
        state_shardings=shardings,
    )
    # Callback states are heterogeneous -- each callback has its own CbStateT --
    # so this is the one place they are held untyped. Each is only ever handed
    # back to the callback that made it.
    cb_states: list[Any] = [cb.begin(current) for cb in callbacks]

    step = int(task.step_of(current))
    logging.info("fit: steps %d -> %d, %d per call", step, config.num_steps, k)
    summary: stats.ModelSummary | None = None
    while step < config.num_steps:
        waited = time.perf_counter()
        chunk = list(itertools.islice(batches, k))
        data_seconds = time.perf_counter() - waited
        if not host_all(len(chunk) == k):
            logging.info("fit: train data exhausted at step %d", step)
            break
        if summary is None:
            summary = _summarize(task, current, chunk[0], partitioner)
        prepared = time.perf_counter()
        stacked = partitioner.to_global(stack(chunk), leading=1)
        prep_seconds = time.perf_counter() - prepared
        current, metrics = run(current, stacked)
        before, step = step, step + k
        event = StepEvent(
            step=step,
            num_steps=k,
            state=current,
            metrics=metrics,
            final=step >= config.num_steps,
            summary=summary,
            data_seconds=data_seconds,
            prep_seconds=prep_seconds,
        )

        stop = False
        for i, cb in enumerate(callbacks):
            cb_states[i] = cb.observe(cb_states[i], event)
            if fires(cb.trigger, before=before, after=step, final=event.final):
                cb_states[i], control = cb.fire(cb_states[i], event)
                stop |= control is Control.STOP
        if host_any(stop):
            logging.info("fit: stopped by a callback at step %d", step)
            break

    for cb, cb_state in zip(callbacks, cb_states, strict=True):
        cb.end(cb_state, current)
    return current


def _summarize[StateT, BatchT](
    task: Task[StateT, BatchT, Any, Any],
    state: StateT,
    local_batch: BatchT,
    partitioner: Partitioner,
) -> stats.ModelSummary:
    """Log the model table and step FLOPs once, before the first step runs.

    Traced, not run: costs one extra trace of the step, no compile.
    """
    table = task.describe(local_batch)
    if table is not None and is_primary():
        logging.info("model:\n%s", table)
    summary = stats.summarize(task, state, partitioner.to_global(local_batch))
    if is_primary():
        stats.log(summary)
    return summary

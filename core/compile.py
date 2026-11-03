"""Turning a task's train step into the program the loop actually runs.

The loop calls one compiled function per iteration, which runs `steps_per_call`
train steps under `lax.scan`. Host round-trips are paid once per K steps, and a
task that makes its own data on device (`NoBatch`) trains for K steps as a
single XLA program -- for RL, environment simulation included.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core import metrics as metrics_lib
from core.config import InnerLoop
from core.distributed import Partitioner
from core.hooks import StepHook, with_hooks
from core.metrics import Metrics
from core.task import Task
from core.types import jit


def stack[BatchT](batches: Sequence[BatchT]) -> BatchT:
    """K batches -> one batch with a leading axis of K, on host.

    The result has the same *type* as each input, with every leaf one rank
    higher; that is what `lax.scan` slices back apart.
    """
    return jax.tree.map(lambda *xs: np.stack(xs), *batches)


def compile_train[StateT, BatchT, MetricsT: Metrics](
    task: Task[StateT, BatchT, MetricsT, Any],
    *,
    steps_per_call: int,
    hooks: Sequence[StepHook[StateT, MetricsT]] = (),
    inner_loop: InnerLoop = InnerLoop.SCAN,
    partitioner: Partitioner,
    state_shardings: Any,
) -> Callable[[StateT, BatchT], tuple[StateT, MetricsT]]:
    """`(state, stacked batches) -> (state, metrics merged over the K steps)`, jitted.

    The state keeps `state_shardings` across the call (and its input buffers
    are donated, so the old state's memory is reused); stacked batches are
    split over the data axes; metrics come back replicated, so every process
    can read them.
    """
    if steps_per_call < 1:
        raise ValueError(f"steps_per_call must be >= 1, got {steps_per_call}")
    step = with_hooks(task.train_step, hooks)

    def scanned(state: StateT, batches: BatchT) -> tuple[StateT, MetricsT]:
        state, stacked = jax.lax.scan(step, state, batches, length=steps_per_call)
        return state, metrics_lib.reduce_stacked(stacked)

    def unrolled(state: StateT, batches: BatchT) -> tuple[StateT, MetricsT]:
        outs: list[MetricsT] = []
        for i in range(steps_per_call):
            state, m = step(state, jax.tree.map(lambda x: x[i], batches))
            outs.append(m)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *outs)
        return state, metrics_lib.reduce_stacked(stacked)

    match inner_loop:
        case InnerLoop.SCAN:
            many = scanned
        case InnerLoop.UNROLL:
            many = unrolled

    return jit(
        many,
        in_shardings=(state_shardings, partitioner.batch_sharding(leading=1)),
        out_shardings=(state_shardings, partitioner.replicated),
        donate_argnums=(0,),
    )
